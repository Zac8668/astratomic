//! Deterministic `KinematicActor` component, `ActorState` integer tick counters,
//! and AABB grid-sweep collision over [`ChunkWorld`].

use astratomic_sim::{
    atom::{Atom, AtomState, Materials},
    chunk::CHUNK_SIZE,
    consts::{FUEL_MAX, PLAYER_STEP_HEIGHT},
    particle::ParticlePool,
    world::{
        ChunkPos, ChunkWorld, DirtyRects, SimulationSettings, global_to_chunk, update_dirty_rects,
        update_dirty_rects_3x3,
    },
};
use bones_framework::prelude::*;

/// Inset epsilon used when testing perpendicular AABB axes and snapping to obstacle edges.
pub const SKIN: f32 = 0.001;

/// Default coyote-time window in simulation ticks.
pub const COYOTE_TIME_TICKS: u8 = 5;

/// Composable kinematic actor state in Bones world coordinates (`+Y` up, `-Y` down).
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct KinematicActor {
    /// Current center position in Bones world space (`+Y` up).
    pub pos: Vec2,
    /// Position at the start of the current tick (used for render interpolation and Verlet hooks).
    pub prev_pos: Vec2,
    /// Velocity in world units (atoms) per simulation tick (`+Y` up).
    pub vel: Vec2,
    /// Half-extents of the actor's AABB in atoms (e.g. `Vec2::new(3.0, 6.0)` for a 6x12 hitbox).
    pub half_size: Vec2,
    /// Maximum auto-step-up height in atoms when walking into small ledges (defaults to [`PLAYER_STEP_HEIGHT`]).
    pub step_height: i32,
    /// Actor weight / mass in simulation units (default 70.0). Influences impact impulse,
    /// particle splash velocity, and fluid drag.
    pub weight: f32,
    /// Whether the actor is currently resting on solid/powder ground.
    pub grounded: bool,
    /// Estimated upward surface normal (`+Y` up) for procedural leg/torso tilting.
    pub ground_normal: Vec2,
}

impl Default for KinematicActor {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            prev_pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            half_size: Vec2::new(3.0, 6.0),
            step_height: PLAYER_STEP_HEIGHT,
            weight: 70.0,
            grounded: false,
            ground_normal: Vec2::Y,
        }
    }
}

impl KinematicActor {
    pub fn new(pos: Vec2) -> Self {
        Self {
            pos,
            prev_pos: pos,
            ..Self::default()
        }
    }
}

/// Deterministic integer tick counters and fuel state for an actor.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ActorState {
    pub jetpack_fuel: f32,
    pub max_jetpack_fuel: f32,
    /// Integer tick cooldown between tool/weapon activations (decremented via `.saturating_sub(1)`).
    pub fire_cooldown_ticks: u16,
    /// Remaining ticks after leaving ground during which a jump is still allowed.
    pub coyote_ticks: u8,
    /// Consecutive ticks the actor's core has been buried inside solid/powder atoms.
    pub buried_ticks: u8,
    /// Cadence counter for footstep particles while walking on ground.
    pub footstep_ticks: u8,
    /// Whether the actor initiated a jump takeoff on the current tick.
    pub jumped_this_tick: bool,
}

impl Default for ActorState {
    fn default() -> Self {
        Self {
            jetpack_fuel: FUEL_MAX,
            max_jetpack_fuel: FUEL_MAX,
            fire_cooldown_ticks: 0,
            coyote_ticks: 0,
            buried_ticks: 0,
            footstep_ticks: 0,
            jumped_this_tick: false,
        }
    }
}

/// Converts a world-space integer cell `(wx, wy)` (`+Y` up, where cell `(wx, wy)` spans
/// `[wx, wx + 1] x [wy, wy + 1]`) into `ChunkWorld` global atom coordinates (`+Y` down).
#[inline]
pub fn world_cell_to_global_atom(wx: i32, wy: i32) -> IVec2 {
    ivec2(wx, -wy - 1)
}

/// Converts a `ChunkWorld` global atom coordinate `(gx, gy)` (`+Y` down) into the bottom-left
/// corner of its world-space unit cell `(wx, wy)` (`+Y` up).
#[inline]
pub fn global_atom_to_world_cell(global: IVec2) -> IVec2 {
    ivec2(global.x, -global.y - 1)
}

/// Returns `true` if the world cell `(wx, wy)` (`+Y` up) contains a solid or powder atom
/// (or lies in an unloaded chunk).
#[inline]
pub fn is_world_cell_solid(chunk_world: &ChunkWorld, wx: i32, wy: i32) -> bool {
    let chunk_pos = global_to_chunk(world_cell_to_global_atom(wx, wy));
    match chunk_world.get_atom(&chunk_pos) {
        Some(atom) => atom.is_solid() || atom.is_powder(),
        None => true,
    }
}

/// Tests whether any world cell intersecting the world-space AABB `[min_x, max_x] x [min_y, max_y]`
/// (`+Y` up) contains a solid or powder atom.
///
/// Uses `coord.floor() as i32` exclusively so negative coordinates never truncate toward zero.
#[inline]
pub fn aabb_has_solid(
    chunk_world: &ChunkWorld,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
) -> bool {
    if min_x >= max_x || min_y >= max_y {
        return false;
    }
    let x0 = min_x.floor() as i32;
    let x1 = max_x.floor() as i32;
    let y0 = min_y.floor() as i32;
    let y1 = max_y.floor() as i32;

    for wy in y0..=y1 {
        for wx in x0..=x1 {
            if is_world_cell_solid(chunk_world, wx, wy) {
                return true;
            }
        }
    }
    false
}

/// Tests whether any cell in columns `wx_start..=wx_end` and rows `wy_start..=wy_end` contains
/// a solid or powder atom.
#[inline]
pub fn cells_have_solid(
    chunk_world: &ChunkWorld,
    wx_start: i32,
    wx_end: i32,
    wy_start: i32,
    wy_end: i32,
) -> bool {
    for wy in wy_start..=wy_end {
        for wx in wx_start..=wx_end {
            if is_world_cell_solid(chunk_world, wx, wy) {
                return true;
            }
        }
    }
    false
}

/// Finds the first solid obstacle column in `start_wx..=end_wx` when sweeping horizontally.
///
/// If `moving_right` is true, scans `start_wx..=end_wx` ascending and returns the first column with solid atoms.
/// If `moving_right` is false, scans `start_wx..=end_wx` descending and returns the first column with solid atoms.
#[inline]
pub fn find_horizontal_hit_column(
    chunk_world: &ChunkWorld,
    start_wx: i32,
    end_wx: i32,
    wy_start: i32,
    wy_end: i32,
    moving_right: bool,
) -> Option<i32> {
    if moving_right {
        for wx in start_wx..=end_wx {
            for wy in wy_start..=wy_end {
                if is_world_cell_solid(chunk_world, wx, wy) {
                    return Some(wx);
                }
            }
        }
    } else {
        for wx in (start_wx..=end_wx).rev() {
            for wy in wy_start..=wy_end {
                if is_world_cell_solid(chunk_world, wx, wy) {
                    return Some(wx);
                }
            }
        }
    }
    None
}

/// Estimates the upward ground normal (`+Y` up) beneath an actor's feet by sampling surface
/// heights across the left and right edges of the footprint.
pub fn estimate_ground_normal(
    chunk_world: &ChunkWorld,
    pos: Vec2,
    half_size: Vec2,
    step_height: i32,
) -> Vec2 {
    let x_left = (pos.x - half_size.x + SKIN).floor() as i32;
    let x_right = (pos.x + half_size.x - SKIN).floor() as i32;
    let foot_wy = (pos.y - half_size.y).floor() as i32;
    let search_range = step_height.max(1) + 2;

    let sample_column_top = |wx: i32| -> Option<f32> {
        for wy in ((foot_wy - search_range)..=(foot_wy + search_range)).rev() {
            if is_world_cell_solid(chunk_world, wx, wy)
                && !is_world_cell_solid(chunk_world, wx, wy + 1)
            {
                return Some((wy + 1) as f32);
            }
        }
        None
    };

    if x_right > x_left
        && let (Some(y_l), Some(y_r)) = (sample_column_top(x_left), sample_column_top(x_right))
    {
        let dx = (x_right - x_left) as f32;
        let dy = y_r - y_l;
        let normal = Vec2::new(-dy, dx).normalize_or_zero();
        if normal.y > 0.0 {
            return normal;
        }
    }

    Vec2::Y
}

/// Finds a safe spawn position on top of the terrain surface at `world_x` (`+Y` up).
pub fn find_surface_spawn_pos(chunk_world: &ChunkWorld, world_x: f32, half_size: Vec2) -> Vec2 {
    if chunk_world.chunks.is_empty() {
        return Vec2::new(world_x, 220.0);
    }

    let min_chunk_y = chunk_world.chunks.keys().map(|c| c.y).min().unwrap_or(-6);
    let max_chunk_y = chunk_world.chunks.keys().map(|c| c.y).max().unwrap_or(6);

    // Convert global atom Y range to world cell Y range (`+Y` up).
    let top_wy = -(min_chunk_y * CHUNK_SIZE as i32) - 2;
    let bot_wy = -((max_chunk_y + 1) * CHUNK_SIZE as i32) + 2;
    let wx = world_x.floor() as i32;

    for wy in (bot_wy..=top_wy).rev() {
        if is_world_cell_solid(chunk_world, wx, wy) {
            let candidate_y = (wy + 1) as f32 + half_size.y + SKIN;
            let clear = !aabb_has_solid(
                chunk_world,
                world_x - half_size.x + SKIN,
                world_x + half_size.x - SKIN,
                candidate_y - half_size.y + SKIN,
                candidate_y + half_size.y - SKIN,
            );
            if clear {
                return Vec2::new(world_x, candidate_y);
            }
        }
    }

    Vec2::new(world_x, 260.0)
}

/// Returns the RGB preview/display color for an atom, falling back to its material color if unset.
#[inline]
pub fn atom_color(atom: &Atom) -> [u8; 4] {
    if atom.color != [0, 0, 0, 0] {
        atom.color
    } else {
        Atom::material_preview_color(atom.material_id)
    }
}

fn handle_horizontal_impact(
    chunk_world: &mut ChunkWorld,
    actor: &mut KinematicActor,
    hit_wx: i32,
    y0: i32,
    y1: i32,
    initial_vx: f32,
    materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    let speed = initial_vx.abs();
    if speed < 0.6 {
        return;
    }
    for wy in y0..=y1 {
        let cp = global_to_chunk(world_cell_to_global_atom(hit_wx, wy));
        if let Some(&atom) = chunk_world.get_atom(&cp) {
            if atom.is_air() || atom.is_object() {
                continue;
            }
            let outward_x = -initial_vx.signum();
            let g_pos = world_cell_to_global_atom(hit_wx, wy);
            let actor_weight = actor.weight.max(1.0);
            let mat_weight = materials
                .map(|m| m.get_from_atom(&atom).weight)
                .unwrap_or(2.0)
                .max(0.1);

            if atom.is_powder() {
                let speed_scale =
                    (speed * 0.4) * (actor_weight / (mat_weight * 35.0)).clamp(0.4, 2.5);

                if let Some(ref mut pool) = particle_pool {
                    let p_vx = outward_x * speed_scale * (0.8 + fastrand::f32() * 0.6);
                    let p_vy = -(0.5 + fastrand::f32() * 1.0) * speed_scale;

                    // Eject real powder matter atom
                    if let Some(ref mut dr) = dirty_rects {
                        pool.eject_from_grid_dirty(chunk_world, dr, g_pos.x, g_pos.y, [p_vx, p_vy]);
                    } else {
                        pool.eject_from_grid(chunk_world, g_pos.x, g_pos.y, [p_vx, p_vy]);
                    }

                    // Wake adjacent powder atoms
                    for n_wy in [wy - 1, wy + 1] {
                        let n_cp = global_to_chunk(world_cell_to_global_atom(hit_wx, n_wy));
                        if let Some(wake_atom) = chunk_world.get_mut_atom(n_cp) {
                            if wake_atom.is_powder() {
                                wake_atom.set_free_falling(true);
                                if let Some(ref mut dr) = dirty_rects {
                                    update_dirty_rects_3x3(&mut dr.current, n_cp);
                                    update_dirty_rects(&mut dr.render, n_cp);
                                }
                            }
                        }
                    }

                    // Extra powder particles spraying outward in material color (NO smoke)
                    let color = atom_color(&atom);
                    let extra = fastrand::usize(3..=6);
                    for _ in 0..extra {
                        let vx = outward_x * speed_scale * (0.7 + fastrand::f32() * 0.8);
                        let vy = -(0.4 + fastrand::f32() * 0.9) * speed_scale;
                        pool.spawn_fx(
                            [g_pos.x as f32 + 0.5, g_pos.y as f32 + 0.5],
                            [vx, vy],
                            color,
                            fastrand::u8(14..28),
                            true,
                        );
                    }
                }
                break;
            } else if atom.is_solid() {
                // Hard solids: bouncy rock chips in stone color (NO smoke)
                let speed_scale =
                    (speed * 0.38) * (actor_weight / (mat_weight * 45.0)).clamp(0.4, 2.0);

                if let Some(ref mut pool) = particle_pool {
                    let chip_count =
                        ((speed * 2.0 * (actor_weight / 60.0)).round() as usize).clamp(4, 10);
                    let color = atom_color(&atom);

                    for _ in 0..chip_count {
                        let p_vx = outward_x * (0.7 + fastrand::f32() * 0.9) * speed_scale;
                        let p_vy = -(0.4 + fastrand::f32() * 0.9) * speed_scale;
                        pool.spawn_fx(
                            [g_pos.x as f32 + 0.5, g_pos.y as f32 + 0.5],
                            [p_vx, p_vy],
                            color,
                            fastrand::u8(16..32),
                            true,
                        );
                    }
                }
                break;
            }
        }
    }
}

fn handle_vertical_floor_impact(
    chunk_world: &mut ChunkWorld,
    actor: &mut KinematicActor,
    x0: i32,
    x1: i32,
    hit_wy: i32,
    initial_vy: f32,
    materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    let impact_speed = initial_vy.abs();
    if impact_speed < 0.6 {
        return;
    }

    let mut powder_cells = Vec::new();
    let mut solid_cells = Vec::new();
    let mut fluid_cells = Vec::new();

    for wx in x0..=x1 {
        let cp = global_to_chunk(world_cell_to_global_atom(wx, hit_wy));
        if let Some(&atom) = chunk_world.get_atom(&cp) {
            if atom.is_powder() {
                powder_cells.push((wx, atom));
            } else if atom.is_solid() {
                solid_cells.push((wx, atom));
            } else if atom.is_fluid() {
                fluid_cells.push((wx, atom));
            }
        }
    }

    let actor_weight = actor.weight.max(1.0);

    if !powder_cells.is_empty() {
        // 1. Powders (Sand, Gravel, Dirt, Snow, etc.): Real Matter + Powder Sprays (NO smoke)
        let mut total_w = 0.0;
        for (_, atom) in &powder_cells {
            total_w += materials
                .map(|m| m.get_from_atom(atom).weight)
                .unwrap_or(1.5)
                .max(0.1);
        }
        let avg_mat_weight = total_w / powder_cells.len() as f32;
        let weight_ratio = (actor_weight / avg_mat_weight).clamp(0.2, 5.0);

        // Landing in powder absorbs horizontal momentum
        let horiz_absorption = (avg_mat_weight / actor_weight * 8.0).clamp(0.15, 0.70);
        actor.vel.x *= 1.0 - horiz_absorption;

        if let Some(ref mut pool) = particle_pool {
            let speed_scale =
                (impact_speed * 0.45) * (actor_weight / (avg_mat_weight * 40.0)).clamp(0.4, 2.5);

            // Eject real powder matter atoms into the air to dent the pile
            let eject_count = ((impact_speed * 1.2 * (weight_ratio * 0.4)).round() as usize)
                .clamp(2, 6)
                .min(powder_cells.len());

            for (wx, _) in powder_cells.iter().take(eject_count) {
                let g_pos = world_cell_to_global_atom(*wx, hit_wy);
                let rel_x = *wx as f32 + 0.5 - actor.pos.x;
                let outward_sign = if rel_x.abs() > 0.1 {
                    rel_x.signum()
                } else if fastrand::bool() {
                    1.0
                } else {
                    -1.0
                };
                let p_vx =
                    outward_sign * (0.6 + fastrand::f32() * 0.8) * speed_scale + actor.vel.x * 0.2;
                let p_vy = -(0.8 + fastrand::f32() * 0.8) * speed_scale;

                if let Some(ref mut dr) = dirty_rects {
                    pool.eject_from_grid_dirty(chunk_world, dr, g_pos.x, g_pos.y, [p_vx, p_vy]);
                } else {
                    pool.eject_from_grid(chunk_world, g_pos.x, g_pos.y, [p_vx, p_vy]);
                }
            }

            // Wake adjacent powder atoms under the feet so landing physically dents the pile
            for wx in x0..=x1 {
                let n_cp = global_to_chunk(world_cell_to_global_atom(wx, hit_wy - 1));
                if let Some(wake_atom) = chunk_world.get_mut_atom(n_cp) {
                    if wake_atom.is_powder() {
                        wake_atom.set_free_falling(true);
                        if let Some(ref mut dr) = dirty_rects {
                            update_dirty_rects_3x3(&mut dr.current, n_cp);
                            update_dirty_rects(&mut dr.render, n_cp);
                        }
                    }
                }
            }
            for wwx in [x0 - 1, x1 + 1] {
                let n_cp = global_to_chunk(world_cell_to_global_atom(wwx, hit_wy));
                if let Some(wake_atom) = chunk_world.get_mut_atom(n_cp) {
                    if wake_atom.is_powder() {
                        wake_atom.set_free_falling(true);
                        if let Some(ref mut dr) = dirty_rects {
                            update_dirty_rects_3x3(&mut dr.current, n_cp);
                            update_dirty_rects(&mut dr.render, n_cp);
                        }
                    }
                }
            }

            // Generous spray of bouncing powder particles (NO smoke)
            let sample_atom = powder_cells[0].1;
            let color = atom_color(&sample_atom);
            let center_gpos = world_cell_to_global_atom((x0 + x1) / 2, hit_wy);
            let spray_count = ((impact_speed * 2.2).round() as usize).clamp(6, 16);
            for _ in 0..spray_count {
                let p_vx = (fastrand::f32() * 2.0 - 1.0) * speed_scale * 1.5 + actor.vel.x * 0.2;
                let p_vy = -(0.6 + fastrand::f32() * 1.0) * speed_scale;
                pool.spawn_fx(
                    [
                        center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                        center_gpos.y as f32 + 0.5,
                    ],
                    [p_vx, p_vy],
                    color,
                    fastrand::u8(16..32),
                    true,
                );
            }
        }
    } else if !solid_cells.is_empty() {
        // 2. Hard Solids (Stone, Brick, Wood, Metal, etc.): Generous Bouncy Chips (NO smoke), NEVER eject
        let mut total_w = 0.0;
        for (_, atom) in &solid_cells {
            total_w += materials
                .map(|m| m.get_from_atom(atom).weight)
                .unwrap_or(2.5)
                .max(0.1);
        }
        let avg_mat_weight = total_w / solid_cells.len() as f32;
        let speed_scale =
            (impact_speed * 0.42) * (actor_weight / (avg_mat_weight * 40.0)).clamp(0.4, 2.2);

        // Minor friction on hard solid floors
        actor.vel.x *= 0.95;

        if let Some(ref mut pool) = particle_pool {
            let sample_atom = solid_cells[0].1;
            let color = atom_color(&sample_atom);
            let center_gpos = world_cell_to_global_atom((x0 + x1) / 2, hit_wy);

            // Generous shower of bouncy rock-chip VFX particles scattering and bouncing across the floor
            let chip_count =
                ((impact_speed * 2.8 * (actor_weight / 50.0)).round() as usize).clamp(6, 18);
            for _ in 0..chip_count {
                let p_vx = (fastrand::f32() * 2.0 - 1.0) * speed_scale * 1.6 + actor.vel.x * 0.25;
                let p_vy = -(0.5 + fastrand::f32() * 1.1) * speed_scale;
                pool.spawn_fx(
                    [
                        center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                        center_gpos.y as f32 + 0.5,
                    ],
                    [p_vx, p_vy],
                    color,
                    fastrand::u8(18..36),
                    true,
                );
            }
        }
    } else if !fluid_cells.is_empty() {
        // 3. Fluids on floor: Real Liquid Droplet Splash + Generous Droplet VFX
        let (_wx, sample_atom) = fluid_cells[0];
        let mat_weight = materials
            .map(|m| m.get_from_atom(&sample_atom).weight)
            .unwrap_or(1.0)
            .max(0.1);
        let speed_scale =
            (impact_speed * 0.45) * (actor_weight / (mat_weight * 40.0)).clamp(0.4, 2.2);

        if let Some(ref mut pool) = particle_pool {
            let center_gpos = world_cell_to_global_atom((x0 + x1) / 2, hit_wy);
            let is_lava = sample_atom.material_id == 6;
            let fx_color = if is_lava {
                [255, 180, 50, 240]
            } else {
                atom_color(&sample_atom)
            };

            // Eject real liquid droplets
            let eject_count = ((impact_speed * 0.8).round() as usize)
                .clamp(1, 3)
                .min(fluid_cells.len());
            for (fwx, _) in fluid_cells.iter().take(eject_count) {
                let g_pos = world_cell_to_global_atom(*fwx, hit_wy);
                let p_vx = (fastrand::f32() * 2.0 - 1.0) * speed_scale * 1.3 + actor.vel.x * 0.2;
                let p_vy = -(0.8 + fastrand::f32() * 0.9) * speed_scale;
                if let Some(ref mut dr) = dirty_rects {
                    pool.eject_from_grid_dirty(chunk_world, dr, g_pos.x, g_pos.y, [p_vx, p_vy]);
                } else {
                    pool.eject_from_grid(chunk_world, g_pos.x, g_pos.y, [p_vx, p_vy]);
                }
            }

            // Generous spray of bouncing splash droplets
            let splash_count = ((impact_speed * 2.6).round() as usize).clamp(8, 18);
            for _ in 0..splash_count {
                let p_vx = (fastrand::f32() * 2.0 - 1.0) * speed_scale * 1.4 + actor.vel.x * 0.2;
                let p_vy = -(0.7 + fastrand::f32() * 1.0) * speed_scale;
                pool.spawn_fx(
                    [
                        center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                        center_gpos.y as f32 + 0.5,
                    ],
                    [p_vx, p_vy],
                    fx_color,
                    fastrand::u8(14..28),
                    true,
                );
            }
        }
    }
}

fn handle_jump_takeoff_particles(
    chunk_world: &mut ChunkWorld,
    actor: &KinematicActor,
    x0: i32,
    x1: i32,
    foot_wy: i32,
    _materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    let Some(ref mut pool) = particle_pool else {
        return;
    };

    let mut powder_cells = Vec::new();
    let mut solid_cells = Vec::new();
    let mut fluid_cells = Vec::new();

    // Check cells directly beneath the feet
    for wy in [foot_wy, foot_wy - 1] {
        for wx in x0..=x1 {
            let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
            if let Some(&atom) = chunk_world.get_atom(&cp) {
                if atom.is_powder() {
                    powder_cells.push((wx, wy, atom));
                } else if atom.is_solid() {
                    solid_cells.push((wx, wy, atom));
                } else if atom.is_fluid() {
                    fluid_cells.push((wx, wy, atom));
                }
            }
        }
    }

    let center_wx = (x0 + x1) / 2;
    let center_gpos = world_cell_to_global_atom(center_wx, foot_wy);

    if !powder_cells.is_empty() {
        // Eject 1 real powder grain downwards/backwards
        let (pwx, pwy, sample_atom) = powder_cells[0];
        let g_pos = world_cell_to_global_atom(pwx, pwy);
        let color = atom_color(&sample_atom);
        let p_vx = (fastrand::f32() * 2.0 - 1.0) * 1.5 + actor.vel.x * 0.2;
        let p_vy = 0.3 + fastrand::f32() * 0.8; // pushed downward in ChunkWorld
        if let Some(ref mut dr) = dirty_rects {
            pool.eject_from_grid_dirty(chunk_world, dr, g_pos.x, g_pos.y, [p_vx, p_vy]);
        } else {
            pool.eject_from_grid(chunk_world, g_pos.x, g_pos.y, [p_vx, p_vy]);
        }

        // Spray 5..=10 powder grains outward and downward
        let count = fastrand::usize(5..=10);
        for _ in 0..count {
            let vx = (fastrand::f32() * 2.0 - 1.0) * 2.0 + actor.vel.x * 0.2;
            let vy = 0.2 + fastrand::f32() * 0.8;
            pool.spawn_fx(
                [
                    center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                    center_gpos.y as f32 + 0.5,
                ],
                [vx, vy],
                color,
                fastrand::u8(12..24),
                true,
            );
        }
    } else if !solid_cells.is_empty() {
        // Spray 4..=8 bouncy rock chips
        let (_, _, sample_atom) = solid_cells[0];
        let color = atom_color(&sample_atom);
        let count = fastrand::usize(4..=8);
        for _ in 0..count {
            let vx = (fastrand::f32() * 2.0 - 1.0) * 2.2 + actor.vel.x * 0.2;
            let vy = 0.2 + fastrand::f32() * 0.7;
            pool.spawn_fx(
                [
                    center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                    center_gpos.y as f32 + 0.5,
                ],
                [vx, vy],
                color,
                fastrand::u8(14..28),
                true,
            );
        }
    } else if !fluid_cells.is_empty() {
        // Spray 6..=12 fluid droplets
        let (_, _, sample_atom) = fluid_cells[0];
        let color = atom_color(&sample_atom);
        let count = fastrand::usize(6..=12);
        for _ in 0..count {
            let vx = (fastrand::f32() * 2.0 - 1.0) * 2.0 + actor.vel.x * 0.2;
            let vy = -(0.4 + fastrand::f32() * 0.8);
            pool.spawn_fx(
                [
                    center_gpos.x as f32 + 0.5 + (fastrand::f32() - 0.5) * 2.0,
                    center_gpos.y as f32 + 0.5,
                ],
                [vx, vy],
                color,
                fastrand::u8(12..24),
                true,
            );
        }
    }
}

fn handle_footstep_particles(
    chunk_world: &mut ChunkWorld,
    actor: &KinematicActor,
    x0: i32,
    x1: i32,
    foot_wy: i32,
    _materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    let Some(ref mut pool) = particle_pool else {
        return;
    };
    let dir = actor.vel.x.signum();
    let backward_x = -dir;
    let foot_wx = if dir > 0.0 { x0 } else { x1 };

    // Sample ground cell underneath the trailing foot
    let mut ground_sample = None;
    for wy in [foot_wy, foot_wy - 1] {
        let cp = global_to_chunk(world_cell_to_global_atom(foot_wx, wy));
        if let Some(&atom) = chunk_world.get_atom(&cp) {
            if !atom.is_air() && !atom.is_object() {
                ground_sample = Some((foot_wx, wy, atom));
                break;
            }
        }
    }

    let Some((g_wx, g_wy, atom)) = ground_sample else {
        return;
    };
    let g_pos = world_cell_to_global_atom(g_wx, g_wy);
    let color = atom_color(&atom);

    if atom.is_powder() {
        // Kicking powder: 2-3 powder grains flicked backward
        let count = fastrand::usize(2..=3);
        for _ in 0..count {
            let vx = backward_x * (0.6 + fastrand::f32() * 0.8) + actor.vel.x * 0.15;
            let vy = -(0.3 + fastrand::f32() * 0.5); // flicked slightly upward in ChunkWorld
            pool.spawn_fx(
                [g_pos.x as f32 + 0.5, g_pos.y as f32 + 0.5],
                [vx, vy],
                color,
                fastrand::u8(10..20),
                true,
            );
        }
        // 1 in 3 chance: physically kick 1 real powder grain out of the grid
        if fastrand::u8(0..3) == 0 {
            let vx = backward_x * (0.8 + fastrand::f32() * 0.8) + actor.vel.x * 0.2;
            let vy = -(0.4 + fastrand::f32() * 0.5);
            if let Some(ref mut dr) = dirty_rects {
                pool.eject_from_grid_dirty(chunk_world, dr, g_pos.x, g_pos.y, [vx, vy]);
            } else {
                pool.eject_from_grid(chunk_world, g_pos.x, g_pos.y, [vx, vy]);
            }
        }
    } else if atom.is_solid() {
        // Walking on stone/wood/solid: 1-2 tiny bouncy chips flicked backward
        let count = fastrand::usize(1..=2);
        for _ in 0..count {
            let vx = backward_x * (0.5 + fastrand::f32() * 0.7);
            let vy = -(0.2 + fastrand::f32() * 0.4);
            pool.spawn_fx(
                [g_pos.x as f32 + 0.5, g_pos.y as f32 + 0.5],
                [vx, vy],
                color,
                fastrand::u8(8..16),
                true,
            );
        }
    } else if atom.is_fluid() {
        // Walking in shallow water/liquid: 2-4 tiny splash droplets
        let count = fastrand::usize(2..=4);
        for _ in 0..count {
            let vx = backward_x * (0.5 + fastrand::f32() * 0.7);
            let vy = -(0.3 + fastrand::f32() * 0.5);
            pool.spawn_fx(
                [g_pos.x as f32 + 0.5, g_pos.y as f32 + 0.5],
                [vx, vy],
                color,
                fastrand::u8(8..16),
                true,
            );
        }
    }
}

fn handle_fluid_interaction(
    chunk_world: &mut ChunkWorld,
    actor: &mut KinematicActor,
    half_w: f32,
    half_h: f32,
    initial_vel: Vec2,
    materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    let body_x0 = (actor.pos.x - half_w + SKIN).floor() as i32;
    let body_x1 = (actor.pos.x + half_w - SKIN).floor() as i32;
    let body_y0 = (actor.pos.y - half_h + SKIN).floor() as i32;
    let body_y1 = (actor.pos.y + half_h - SKIN).floor() as i32;

    let total_body_cells = ((body_x1 - body_x0 + 1) * (body_y1 - body_y0 + 1)).max(1) as f32;
    let mut fluid_cells_count = 0;
    let mut total_fluid_weight = 0.0;
    let mut dominant_fluid_atom = None;

    for wy in body_y0..=body_y1 {
        for wx in body_x0..=body_x1 {
            let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
            if let Some(&atom) = chunk_world.get_atom(&cp) {
                if atom.is_fluid() {
                    fluid_cells_count += 1;
                    let mw = materials
                        .map(|m| m.get_from_atom(&atom).weight)
                        .unwrap_or(1.0)
                        .max(0.1);
                    total_fluid_weight += mw;
                    if dominant_fluid_atom.is_none() {
                        dominant_fluid_atom = Some((world_cell_to_global_atom(wx, wy), atom, mw));
                    }
                }
            }
        }
    }

    if fluid_cells_count > 0 {
        let submerged = (fluid_cells_count as f32 / total_body_cells).clamp(0.0, 1.0);
        let avg_mat_weight = (total_fluid_weight / fluid_cells_count as f32).max(0.1);
        let actor_weight = actor.weight.max(1.0);

        // Fluid drag slows the actor based on material weight vs actor weight and submerged ratio
        let drag_ratio = (avg_mat_weight / actor_weight) * 15.0;
        let drag = ((0.06 + drag_ratio * 0.12) * submerged).clamp(0.03, 0.45);
        actor.vel.x *= 1.0 - drag;
        actor.vel.y *= 1.0 - drag * 0.75;

        // Cap downward sinking velocity to simulate buoyancy and viscous resistance
        let terminal_sink = -(6.0 / (avg_mat_weight * 0.4 + 0.6));
        if actor.vel.y < terminal_sink {
            actor.vel.y = terminal_sink;
        }

        // Splash particles if moving through fluid with sufficient speed
        let speed = (initial_vel.x * initial_vel.x + initial_vel.y * initial_vel.y).sqrt();
        let is_moving = if actor.grounded {
            actor.vel.x.abs() > 0.4 || initial_vel.y < -0.8
        } else {
            speed > 0.8
        };
        if is_moving && speed > 0.8 {
            if let Some(ref mut pool) = particle_pool {
                let (sample_gpos, sample_atom, _) = dominant_fluid_atom.unwrap();
                let splash_speed =
                    (speed * 0.45) * (actor_weight / (avg_mat_weight * 40.0)).clamp(0.4, 2.2);

                let p_vx = (fastrand::f32() * 2.0 - 1.0) * splash_speed * 1.2;
                let p_vy = -(0.8 + fastrand::f32() * 0.8) * splash_speed;

                // 1. Eject real liquid droplet atom
                if let Some(ref mut dr) = dirty_rects {
                    pool.eject_from_grid_dirty(
                        chunk_world,
                        dr,
                        sample_gpos.x,
                        sample_gpos.y,
                        [p_vx, p_vy],
                    );
                } else {
                    pool.eject_from_grid(chunk_world, sample_gpos.x, sample_gpos.y, [p_vx, p_vy]);
                }

                // 2. Splash droplets in fluid color (bouncing, NO smoke)
                let count = ((speed * 1.8 * (actor_weight / (avg_mat_weight * 40.0))).round()
                    as usize)
                    .clamp(4, 12);
                let is_lava = sample_atom.material_id == 6;
                let fx_color = if is_lava {
                    [255, 180, 50, 240]
                } else {
                    atom_color(&sample_atom)
                };

                for _ in 0..count {
                    let fx_vx = (fastrand::f32() * 2.0 - 1.0) * splash_speed * 1.1;
                    let fx_vy = -(0.6 + fastrand::f32() * 0.8) * splash_speed;
                    let splash_gx =
                        (actor.pos.x + (fastrand::f32() * 2.0 - 1.0) * half_w).floor() as i32;
                    let splash_gy = (-actor.pos.y + half_h * 0.5).floor() as i32;
                    pool.spawn_fx(
                        [splash_gx as f32 + 0.5, splash_gy as f32 + 0.5],
                        [fx_vx, fx_vy],
                        fx_color,
                        fastrand::u8(14..26),
                        true,
                    );
                }
            }
        }
    }
}

/// Steps a single [`KinematicActor`] through [`ChunkWorld`].
pub fn step_kinematic_actor(
    chunk_world: &mut ChunkWorld,
    actor: &mut KinematicActor,
    state: Option<&mut ActorState>,
) {
    step_kinematic_actor_full(chunk_world, actor, state, None, None, None);
}

/// Steps a single [`KinematicActor`] through [`ChunkWorld`] with full material resistance,
/// powder impact absorption/particle spraying, and fluid drag/splashing.
pub fn step_kinematic_actor_full(
    chunk_world: &mut ChunkWorld,
    actor: &mut KinematicActor,
    mut state: Option<&mut ActorState>,
    materials: Option<&Materials>,
    mut particle_pool: Option<&mut ParticlePool>,
    mut dirty_rects: Option<&mut DirtyRects>,
) {
    // Save `pos` into `prev_pos` at the start of every tick for render interpolation & Verlet physics.
    actor.prev_pos = actor.pos;
    let initial_vel = actor.vel;
    let was_grounded = actor.grounded;

    let half_w = actor.half_size.x.max(0.5);
    let half_h = actor.half_size.y.max(0.5);
    let step_height = actor.step_height.max(0);
    let step_h_f32 = step_height as f32;

    // =========================================================================
    // Step A — Horizontal Sweep with Foot Inset & Auto-Step-Up
    // =========================================================================
    let dx = actor.vel.x;
    if dx.abs() > 0.0 {
        let substeps = dx.abs().ceil().max(1.0) as i32;
        let step_dx = dx / substeps as f32;
        let moving_right = step_dx > 0.0;

        for _ in 0..substeps {
            let new_x = actor.pos.x + step_dx;
            let bottom = actor.pos.y - half_h;
            let top = actor.pos.y + half_h;

            let upper_bottom = (bottom + step_h_f32).min(top - SKIN * 2.0) + SKIN;
            let upper_top = top - SKIN;
            let feet_bottom = bottom + SKIN;
            let feet_top = (bottom + step_h_f32).min(top) - SKIN;

            let upper_y0 = upper_bottom.floor() as i32;
            let upper_y1 = upper_top.floor() as i32;
            let feet_y0 = feet_bottom.floor() as i32;
            let feet_y1 = feet_top.floor() as i32;

            let (lead_wx_start, lead_wx_end) = if moving_right {
                let start = (actor.pos.x + half_w - SKIN).floor() as i32;
                let end = (new_x + half_w - SKIN).floor() as i32;
                (start, end)
            } else {
                let start = (new_x - half_w + SKIN).floor() as i32;
                let end = (actor.pos.x - half_w + SKIN).floor() as i32;
                (start, end)
            };

            let upper_clear =
                !cells_have_solid(chunk_world, lead_wx_start, lead_wx_end, upper_y0, upper_y1);

            if upper_clear {
                let feet_has_solid = step_height > 0
                    && cells_have_solid(chunk_world, lead_wx_start, lead_wx_end, feet_y0, feet_y1);

                if !feet_has_solid {
                    actor.pos.x = new_x;
                } else {
                    // Scan upward `h = 1..=step_height` to find the lowest `h` where the full AABB
                    // shifted by `(step_dx, h as f32)` is completely free of solid atoms.
                    let mut stepped_up = false;
                    for h in 1..=step_height {
                        let candidate_y = actor.pos.y + h as f32;
                        let cand_bottom = (candidate_y - half_h + SKIN).floor() as i32;
                        let cand_top = (candidate_y + half_h - SKIN).floor() as i32;
                        if !cells_have_solid(
                            chunk_world,
                            lead_wx_start,
                            lead_wx_end,
                            cand_bottom,
                            cand_top,
                        ) {
                            actor.pos.x = new_x;
                            actor.pos.y = candidate_y;
                            stepped_up = true;
                            break;
                        }
                    }

                    if !stepped_up {
                        let full_y0 = feet_y0;
                        let full_y1 = upper_y1;
                        if let Some(hit_wx) = find_horizontal_hit_column(
                            chunk_world,
                            lead_wx_start,
                            lead_wx_end,
                            full_y0,
                            full_y1,
                            moving_right,
                        ) {
                            if moving_right {
                                let snapped = (hit_wx as f32) - half_w - SKIN;
                                actor.pos.x = snapped.max(actor.pos.x);
                            } else {
                                let snapped = ((hit_wx + 1) as f32) + half_w + SKIN;
                                actor.pos.x = snapped.min(actor.pos.x);
                            }
                            handle_horizontal_impact(
                                chunk_world,
                                actor,
                                hit_wx,
                                full_y0,
                                full_y1,
                                initial_vel.x,
                                materials,
                                particle_pool.as_deref_mut(),
                                dirty_rects.as_deref_mut(),
                            );
                        }
                        actor.vel.x = 0.0;
                        break;
                    }
                }
            } else {
                let full_y0 = feet_y0;
                let full_y1 = upper_y1;
                if let Some(hit_wx) = find_horizontal_hit_column(
                    chunk_world,
                    lead_wx_start,
                    lead_wx_end,
                    full_y0,
                    full_y1,
                    moving_right,
                ) {
                    if moving_right {
                        let snapped = (hit_wx as f32) - half_w - SKIN;
                        actor.pos.x = snapped.max(actor.pos.x);
                    } else {
                        let snapped = ((hit_wx + 1) as f32) + half_w + SKIN;
                        actor.pos.x = snapped.min(actor.pos.x);
                    }
                    handle_horizontal_impact(
                        chunk_world,
                        actor,
                        hit_wx,
                        full_y0,
                        full_y1,
                        initial_vel.x,
                        materials,
                        particle_pool.as_deref_mut(),
                        dirty_rects.as_deref_mut(),
                    );
                }
                actor.vel.x = 0.0;
                break;
            }
        }
    }

    // =========================================================================
    // Step B — Vertical Sweep (with `SKIN` horizontal inset on both edges)
    // =========================================================================
    let dy = actor.vel.y;
    let inset_left = actor.pos.x - half_w + SKIN;
    let inset_right = actor.pos.x + half_w - SKIN;
    let x0 = inset_left.floor() as i32;
    let x1 = inset_right.floor() as i32;
    let mut hit_floor = false;

    if dy.abs() > 0.0 {
        let substeps = dy.abs().ceil().max(1.0) as i32;
        let step_dy = dy / substeps as f32;

        for _ in 0..substeps {
            let new_y = actor.pos.y + step_dy;

            if step_dy < 0.0 {
                // Falling (`-Y`) -> leading edge is bottom.
                let lead_wy_start = (new_y - half_h + SKIN).floor() as i32;
                let lead_wy_end = (actor.pos.y - half_h + SKIN).floor() as i32;

                let mut hit_wy = None;
                'search_down: for wy in (lead_wy_start..=lead_wy_end).rev() {
                    for wx in x0..=x1 {
                        if is_world_cell_solid(chunk_world, wx, wy) {
                            hit_wy = Some(wy);
                            break 'search_down;
                        }
                    }
                }

                if let Some(wy) = hit_wy {
                    let snapped = ((wy + 1) as f32) + half_h + SKIN;
                    actor.pos.y = snapped.min(actor.pos.y);
                    let is_landing =
                        (!was_grounded || initial_vel.y < -0.8) && initial_vel.y.abs() >= 0.6;
                    if is_landing {
                        handle_vertical_floor_impact(
                            chunk_world,
                            actor,
                            x0,
                            x1,
                            wy,
                            initial_vel.y,
                            materials,
                            particle_pool.as_deref_mut(),
                            dirty_rects.as_deref_mut(),
                        );
                    }
                    actor.vel.y = 0.0;
                    hit_floor = true;
                    break;
                } else {
                    actor.pos.y = new_y;
                }
            } else {
                // Jumping (`+Y`) -> leading edge is top.
                let lead_wy_start = (actor.pos.y + half_h - SKIN).floor() as i32;
                let lead_wy_end = (new_y + half_h - SKIN).floor() as i32;

                let mut hit_wy = None;
                'search_up: for wy in lead_wy_start..=lead_wy_end {
                    for wx in x0..=x1 {
                        if is_world_cell_solid(chunk_world, wx, wy) {
                            hit_wy = Some(wy);
                            break 'search_up;
                        }
                    }
                }

                if let Some(wy) = hit_wy {
                    let snapped = (wy as f32) - half_h - SKIN;
                    actor.pos.y = snapped.max(actor.pos.y);
                    actor.vel.y = 0.0;
                    break;
                } else {
                    actor.pos.y = new_y;
                }
            }
        }
    }

    // Also check a thin ground probe beneath the feet when `vel.y <= 0.0` so walking along flat
    // ground or stepping down 1 pixel maintains stable grounding.
    if !hit_floor && actor.vel.y <= 0.0 {
        let probe_bottom = (actor.pos.y - half_h) - (SKIN * 4.0);
        let probe_top = (actor.pos.y - half_h) - (SKIN * 0.5);
        if aabb_has_solid(
            chunk_world,
            inset_left,
            inset_right,
            probe_bottom,
            probe_top,
        ) {
            hit_floor = true;
        }
    }

    if hit_floor {
        actor.grounded = true;
        actor.ground_normal =
            estimate_ground_normal(chunk_world, actor.pos, actor.half_size, step_height);
        if let Some(ref mut st) = state {
            st.coyote_ticks = COYOTE_TIME_TICKS;
        }
    } else {
        actor.grounded = false;
        actor.ground_normal = Vec2::Y;
        if let Some(ref mut st) = state {
            st.coyote_ticks = st.coyote_ticks.saturating_sub(1);
        }
    }

    // Step C — Fluid Drag and Splash Particles
    handle_fluid_interaction(
        chunk_world,
        actor,
        half_w,
        half_h,
        initial_vel,
        materials,
        particle_pool.as_deref_mut(),
        dirty_rects.as_deref_mut(),
    );

    // Step D — Jump Takeoff Particles (pushing off the ground with feet)
    let takeoff_foot_wy = (actor.prev_pos.y - half_h).floor() as i32;
    let is_jump_takeoff = state.as_ref().is_some_and(|st| st.jumped_this_tick)
        || (was_grounded && initial_vel.y > 1.0);

    if let Some(ref mut st) = state {
        st.jumped_this_tick = false;
    }

    if is_jump_takeoff {
        handle_jump_takeoff_particles(
            chunk_world,
            actor,
            x0,
            x1,
            takeoff_foot_wy,
            materials,
            particle_pool.as_deref_mut(),
            dirty_rects.as_deref_mut(),
        );
    }

    // Step E — Footstep Particles While Walking On Ground
    let foot_wy = (actor.pos.y - half_h).floor() as i32;
    if actor.grounded && actor.vel.x.abs() > 0.4 {
        if let Some(ref mut st) = state {
            if st.footstep_ticks == 0 {
                handle_footstep_particles(
                    chunk_world,
                    actor,
                    x0,
                    x1,
                    foot_wy,
                    materials,
                    particle_pool.as_deref_mut(),
                    dirty_rects.as_deref_mut(),
                );
                st.footstep_ticks = 7;
            } else {
                st.footstep_ticks = st.footstep_ticks.saturating_sub(1);
            }
        } else {
            let prev_step = (actor.prev_pos.x / 6.0).floor() as i32;
            let curr_step = (actor.pos.x / 6.0).floor() as i32;
            if prev_step != curr_step {
                handle_footstep_particles(
                    chunk_world,
                    actor,
                    x0,
                    x1,
                    foot_wy,
                    materials,
                    particle_pool.as_deref_mut(),
                    dirty_rects.as_deref_mut(),
                );
            }
        }
    } else if let Some(ref mut st) = state {
        if st.footstep_ticks > 2 {
            st.footstep_ticks = 2;
        }
    }

    // Update `buried_ticks` if the actor's core is overlapped by falling powder/solid atoms.
    if let Some(st) = state {
        let core_buried = aabb_has_solid(
            chunk_world,
            actor.pos.x - (half_w * 0.5).max(0.5),
            actor.pos.x + (half_w * 0.5).max(0.5),
            actor.pos.y - (half_h * 0.5).max(0.5),
            actor.pos.y + (half_h * 0.5).max(0.5),
        );
        if core_buried {
            st.buried_ticks = st.buried_ticks.saturating_add(1);
        } else {
            st.buried_ticks = 0;
        }
    }
}

/// Deterministic [`KinematicActor`] grid sweep system reading and mutating [`ChunkWorld`].
pub fn kinematic_actor_movement_system(
    sim_settings: Option<Res<SimulationSettings>>,
    mut chunk_world: ResMut<ChunkWorld>,
    materials: Option<Res<Materials>>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
    mut dirty_rects: Option<ResMut<DirtyRects>>,
    mut rapier_physics: Option<ResMut<crate::rigidbody::RapierPhysics>>,
    entities: Res<Entities>,
    mut actors: CompMut<KinematicActor>,
    mut states: CompMut<ActorState>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    let obstacles = rapier_physics
        .as_ref()
        .map(|p| crate::rigidbody::collect_dynamic_obstacles(p))
        .unwrap_or_default();

    let mut actor_entities: Vec<Entity> = entities.iter_with(&actors).map(|(ent, _)| ent).collect();
    actor_entities.sort_unstable();

    for ent in actor_entities {
        let Some(actor) = actors.get_mut(ent) else {
            continue;
        };
        let state = states.get_mut(ent);
        let initial_vel = actor.vel;
        let prev_pos = actor.prev_pos;

        step_kinematic_actor_full(
            &mut chunk_world,
            actor,
            state,
            materials.as_deref(),
            particle_pool.as_deref_mut(),
            dirty_rects.as_deref_mut(),
        );

        if let Some(ref mut physics) = rapier_physics
            && !obstacles.is_empty()
        {
            crate::rigidbody::resolve_actor_dynamic_obstacles(
                &mut actor.pos,
                prev_pos,
                &mut actor.vel,
                &mut actor.grounded,
                actor.half_size,
                actor.step_height,
                actor.weight,
                initial_vel,
                &obstacles,
                physics,
            );
        }
    }
}

/// Tracks cells in [`ChunkWorld`] where temporary [`AtomState::Object`] atoms were placed
/// for [`KinematicActor`] footprints so that cellular automata particles cannot pass through them.
#[derive(HasSchema, Clone, Debug, Default)]
pub struct ActorSimulationStamp {
    pub stamped_atoms: Vec<(ChunkPos, Atom)>,
}

/// Stamps temporary [`AtomState::Object`] atoms across the footprints of all [`KinematicActor`] entities
/// before cellular automata simulation runs, preventing falling sand and liquids from passing through actors.
pub fn stamp_actor_objects_system(
    sim_settings: Option<Res<SimulationSettings>>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut stamp: ResMut<ActorSimulationStamp>,
    mut dirty_rects: ResMut<DirtyRects>,
    entities: Res<Entities>,
    actors: Comp<KinematicActor>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    stamp.stamped_atoms.clear();

    for (_ent, actor) in entities.iter_with(&actors) {
        let half_w = actor.half_size.x.max(0.5);
        let half_h = actor.half_size.y.max(0.5);

        let min_x = actor.pos.x - half_w + SKIN;
        let max_x = actor.pos.x + half_w - SKIN;
        let min_y = actor.pos.y - half_h + SKIN;
        let max_y = actor.pos.y + half_h - SKIN;

        let x0 = min_x.floor() as i32;
        let x1 = max_x.floor() as i32;
        let y0 = min_y.floor() as i32;
        let y1 = max_y.floor() as i32;

        for wy in y0..=y1 {
            for wx in x0..=x1 {
                let global = world_cell_to_global_atom(wx, wy);
                let chunk_pos = global_to_chunk(global);

                if let Some(atom) = chunk_world.get_atom(&chunk_pos) {
                    if atom.is_void() {
                        let original = *atom;
                        let object_atom = Atom::with_state(1, AtomState::Object, [0, 0, 0, 0]);
                        chunk_world.set_atom(chunk_pos, object_atom);
                        stamp.stamped_atoms.push((chunk_pos, original));
                        update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
                    }
                }
            }
        }
    }
}

/// Cleans up temporary [`AtomState::Object`] atoms after cellular automata simulation runs,
/// restoring the cells back to their original state and waking up surrounding cells.
pub fn clear_actor_objects_system(
    mut chunk_world: ResMut<ChunkWorld>,
    mut stamp: ResMut<ActorSimulationStamp>,
    mut dirty_rects: ResMut<DirtyRects>,
) {
    for (chunk_pos, original) in stamp.stamped_atoms.drain(..) {
        if let Some(atom) = chunk_world.get_atom(&chunk_pos) {
            if atom.is_object() {
                chunk_world.set_atom(chunk_pos, original);
                update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use astratomic_sim::{
        atom::{Atom, AtomState},
        chunk::Chunk,
    };

    fn world_with_flat_floor(floor_wy: i32) -> ChunkWorld {
        let mut world = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }
        let rock = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]);
        for wx in -64..64 {
            for wy in (floor_wy - 8)..=floor_wy {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, rock);
            }
        }
        world
    }

    #[test]
    fn left_wall_at_origin_and_negative_coords_never_sticks() {
        let mut world = world_with_flat_floor(-1);
        let rock = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]);

        // Build a vertical wall at wx = -1 (spanning world x in [-1.0, 0.0]).
        for wy in 0..20 {
            let cp = global_to_chunk(world_cell_to_global_atom(-1, wy));
            world.set_atom(cp, rock);
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(4.0, 10.0),
            prev_pos: Vec2::new(4.0, 10.0),
            vel: Vec2::new(-2.5, -1.5),
            half_size: Vec2::new(3.0, 6.0),
            step_height: PLAYER_STEP_HEIGHT,
            weight: 70.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };
        let mut state = ActorState::default();

        // Step repeatedly while pressing left against the wall at wx = -1:
        // actor must slide down smoothly to the floor at y = 6.001 without sticking on the wall!
        for _ in 0..10 {
            actor.vel.x = -2.5;
            actor.vel.y = -1.5;
            step_kinematic_actor(&mut world, &mut actor, Some(&mut state));
        }

        assert!(
            (actor.pos.x - (3.0 + SKIN)).abs() < 1e-4,
            "actor should snap cleanly to right face of wall wx=-1 at x = 3.001, got {}",
            actor.pos.x
        );
        assert!(
            actor.grounded,
            "sliding down against a left wall must not block vertical falling"
        );
        assert_eq!(state.coyote_ticks, COYOTE_TIME_TICKS);
    }

    #[test]
    fn auto_step_up_climbs_steps_up_to_step_height_and_blocks_taller_walls() {
        let mut world = world_with_flat_floor(-1);
        let rock = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]);

        // Place a `PLAYER_STEP_HEIGHT`-atom-tall ledge starting at wx = 5.
        for wx in 5..20 {
            for wy in 0..PLAYER_STEP_HEIGHT {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, rock);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(1.0, 6.0 + SKIN),
            prev_pos: Vec2::new(1.0, 6.0 + SKIN),
            vel: Vec2::new(2.5, 0.0),
            half_size: Vec2::new(3.0, 6.0),
            step_height: PLAYER_STEP_HEIGHT,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };

        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.x > 2.0 && actor.pos.y >= 6.0 + PLAYER_STEP_HEIGHT as f32,
            "actor should auto-step-up onto the {}-atom ledge (pos = {:?})",
            PLAYER_STEP_HEIGHT,
            actor.pos
        );

        // Place a `(PLAYER_STEP_HEIGHT + 1)`-atom-tall wall at wx = 12.
        for wy in PLAYER_STEP_HEIGHT..(PLAYER_STEP_HEIGHT * 2 + 1) {
            let cp = global_to_chunk(world_cell_to_global_atom(12, wy));
            world.set_atom(cp, rock);
        }

        actor.pos = Vec2::new(8.5, 6.0 + PLAYER_STEP_HEIGHT as f32 + SKIN);
        actor.vel = Vec2::new(2.5, 0.0);
        step_kinematic_actor(&mut world, &mut actor, None);
        assert_eq!(
            actor.vel.x, 0.0,
            "wall taller than PLAYER_STEP_HEIGHT must block horizontal motion"
        );
        assert!((actor.pos.x - (12.0 - 3.0 - SKIN)).abs() < 1e-4);
    }

    #[test]
    fn actor_overlapping_atoms_never_moves_in_reverse_direction() {
        let mut world = world_with_flat_floor(-1);
        let rock = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]);

        // Place solid atoms on the left at wx = 7, 8 (wy from 0 to 10).
        for wx in 7..=8 {
            for wy in 0..=10 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, rock);
            }
        }

        // Actor at x = 10.0, half_w = 3.0.
        // Left edge is at x = 7.001, so it overlaps wx = 7 and 8!
        let mut actor = KinematicActor {
            pos: Vec2::new(10.0, 6.0 + SKIN),
            prev_pos: Vec2::new(10.0, 6.0 + SKIN),
            vel: Vec2::new(2.0, 0.0), // Moving right (away from the atoms on the left)
            half_size: Vec2::new(3.0, 6.0),
            step_height: PLAYER_STEP_HEIGHT,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };

        // When moving right: actor MUST move right and NOT snap to the left!
        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.x > 10.0,
            "moving right away from atoms must increase pos.x, got pos.x = {}",
            actor.pos.x
        );

        // Reset to x = 10.0 and test moving left (into the atoms):
        actor.pos = Vec2::new(10.0, 6.0 + SKIN);
        actor.vel = Vec2::new(-2.0, 0.0);
        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.x <= 10.0,
            "moving left into atoms must never snap backwards to the right! got pos.x = {}",
            actor.pos.x
        );

        // Symmetrically: place atoms on the right at wx = 12, 13
        for wx in 12..=13 {
            for wy in 0..=10 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, rock);
            }
        }

        // Actor at x = 10.0. Right edge is at 12.999, overlapping wx = 12.
        actor.pos = Vec2::new(10.0, 6.0 + SKIN);
        actor.vel = Vec2::new(-2.0, 0.0); // Moving left (away from atoms on right)
        // Clear wx = 7, 8 so left is open
        for wx in 7..=8 {
            for wy in 0..=10 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, Atom::default());
            }
        }

        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.x < 10.0,
            "moving left away from atoms on the right must decrease pos.x, got pos.x = {}",
            actor.pos.x
        );

        // Reset to x = 10.0 and test moving right (into the atoms on right):
        actor.pos = Vec2::new(10.0, 6.0 + SKIN);
        actor.vel = Vec2::new(2.0, 0.0);
        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.x >= 10.0,
            "moving right into atoms must never snap backwards to the left! got pos.x = {}",
            actor.pos.x
        );
    }

    #[test]
    fn actor_jumping_with_atoms_around_feet_moves_upward() {
        let mut world = world_with_flat_floor(-1);
        let rock = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]);

        // Place atoms around the actor's feet (wy = 0..=2)
        for wx in -3..=3 {
            for wy in 0..=2 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, rock);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 6.0 + SKIN),
            prev_pos: Vec2::new(0.0, 6.0 + SKIN),
            vel: Vec2::new(0.0, 3.0),
            half_size: Vec2::new(3.0, 6.0),
            step_height: PLAYER_STEP_HEIGHT,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };

        step_kinematic_actor(&mut world, &mut actor, None);
        assert!(
            actor.pos.y > 6.0 + SKIN,
            "jumping upward with atoms at feet must increase pos.y, got {}",
            actor.pos.y
        );
    }

    #[test]
    fn actor_simulation_stamp_and_clear_cycle() {
        crate::plugin::register_physics_schemas();
        let world = World::new();
        let mut chunk_world = ChunkWorld::default();
        chunk_world.add_chunk(ivec2(0, 0), Chunk::default());
        chunk_world.add_chunk(ivec2(0, -1), Chunk::default());

        world.insert_resource(chunk_world);
        world.insert_resource(DirtyRects::default());
        world.insert_resource(ActorSimulationStamp::default());

        let ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(
            ent,
            KinematicActor {
                pos: Vec2::new(5.0, 5.0),
                prev_pos: Vec2::new(5.0, 5.0),
                vel: Vec2::ZERO,
                half_size: Vec2::new(2.0, 3.0),
                step_height: 3,
                weight: 70.0,
                grounded: false,
                ground_normal: Vec2::Y,
            },
        );

        world.run_system(stamp_actor_objects_system, ());

        {
            let stamp = world.resource::<ActorSimulationStamp>();
            assert!(
                !stamp.stamped_atoms.is_empty(),
                "should have stamped cells for the actor"
            );
            let cw = world.resource::<ChunkWorld>();
            let sample_cp = global_to_chunk(world_cell_to_global_atom(5, 5));
            let atom = cw.get_atom(&sample_cp).unwrap();
            assert!(
                atom.is_object(),
                "stamped cell must have AtomState::Object, got {:?}",
                atom.state()
            );
        }

        world.run_system(clear_actor_objects_system, ());

        {
            let stamp = world.resource::<ActorSimulationStamp>();
            assert!(stamp.stamped_atoms.is_empty(), "stamp should be drained");
            let cw = world.resource::<ChunkWorld>();
            let sample_cp = global_to_chunk(world_cell_to_global_atom(5, 5));
            let atom = cw.get_atom(&sample_cp).unwrap();
            assert!(
                atom.is_void(),
                "cleared cell must be restored to Void, got {:?}",
                atom.state()
            );
        }
    }

    #[test]
    fn falling_powder_cannot_penetrate_actor_when_stamped() {
        use astratomic_sim::{atom::Materials, ca::step_chunk_simulation, world::URect};

        crate::plugin::register_physics_schemas();
        astratomic_sim::plugin::register_sim_schemas();

        let world = World::new();
        let mut chunk_world = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                chunk_world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        world.insert_resource(chunk_world);
        world.insert_resource(DirtyRects::default());
        world.insert_resource(ActorSimulationStamp::default());
        world.insert_resource(Materials::default_builtin());

        // Actor at x = 5.0, y = 5.0, half_size = 2.0, 3.0 (spans y in 2.0..=8.0, x in 3.0..=7.0)
        let ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(
            ent,
            KinematicActor {
                pos: Vec2::new(5.0, 5.0),
                prev_pos: Vec2::new(5.0, 5.0),
                vel: Vec2::ZERO,
                half_size: Vec2::new(2.0, 3.0),
                step_height: 3,
                weight: 70.0,
                grounded: false,
                ground_normal: Vec2::Y,
            },
        );

        // Stamp Object atoms for the actor
        world.run_system(stamp_actor_objects_system, ());

        // Place sand atom at world cell (5, 8), directly above actor's head (wy = 7)
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        let sand_cp = global_to_chunk(world_cell_to_global_atom(5, 8));
        world.resource_mut::<ChunkWorld>().set_atom(sand_cp, sand);
        world
            .resource_mut::<DirtyRects>()
            .current
            .insert(sand_cp.chunk, URect::full_chunk());

        // Step simulation: sand tries to fall under gravity towards the actor's head (wy = 7)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut dr = world.resource_mut::<DirtyRects>();
            let mats = world.resource::<Materials>();
            step_chunk_simulation(&mut cw, &mut dr, &mats);
        }

        // Verify that the actor's body cells (wy in 2..=7) do NOT contain sand
        {
            let cw = world.resource::<ChunkWorld>();
            for wy in 2..=7 {
                let cp = global_to_chunk(world_cell_to_global_atom(5, wy));
                let atom = cw.get_atom(&cp).unwrap();
                assert!(
                    !atom.is_powder(),
                    "sand must not fall through the actor! Cell at wy = {} is powder",
                    wy
                );
            }
        }

        // Clear object atoms
        world.run_system(clear_actor_objects_system, ());

        let cw = world.resource::<ChunkWorld>();
        for wy in 2..=7 {
            let cp = global_to_chunk(world_cell_to_global_atom(5, wy));
            let atom = cw.get_atom(&cp).unwrap();
            assert!(atom.is_void(), "actor cells must be restored to Void");
        }
    }

    #[test]
    fn fluids_can_pass_through_actor_when_stamped() {
        use astratomic_sim::{atom::Materials, ca::step_chunk_simulation, world::URect};

        crate::plugin::register_physics_schemas();
        astratomic_sim::plugin::register_sim_schemas();

        let world = World::new();
        let mut chunk_world = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                chunk_world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        world.insert_resource(chunk_world);
        world.insert_resource(DirtyRects::default());
        world.insert_resource(ActorSimulationStamp::default());
        world.insert_resource(Materials::default_builtin());

        // Actor at x = 5.0, y = 5.0, half_size = 2.0, 3.0 (spans y in 2.0..=8.0, x in 3.0..=7.0)
        let ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(
            ent,
            KinematicActor {
                pos: Vec2::new(5.0, 5.0),
                prev_pos: Vec2::new(5.0, 5.0),
                vel: Vec2::ZERO,
                half_size: Vec2::new(2.0, 3.0),
                step_height: 3,
                weight: 70.0,
                grounded: false,
                ground_normal: Vec2::Y,
            },
        );

        // Stamp Object atoms for the actor
        world.run_system(stamp_actor_objects_system, ());

        // Place water atom at world cell (5, 8), directly above actor's head (wy = 7)
        let water = Atom::with_state(3, AtomState::Liquid, [40, 100, 220, 255]);
        let water_cp = global_to_chunk(world_cell_to_global_atom(5, 8));
        world.resource_mut::<ChunkWorld>().set_atom(water_cp, water);
        world
            .resource_mut::<DirtyRects>()
            .current
            .insert(water_cp.chunk, URect::full_chunk());

        // Step simulation: water falls under gravity and should enter the actor's stamped space at wy = 7
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut dr = world.resource_mut::<DirtyRects>();
            let mats = world.resource::<Materials>();
            step_chunk_simulation(&mut cw, &mut dr, &mats);
        }

        // Verify that the water has entered the actor's footprint (wy in 2..=7)
        {
            let cw = world.resource::<ChunkWorld>();
            let mut found_water_inside = false;
            for wy in 2..=7 {
                let cp = global_to_chunk(world_cell_to_global_atom(5, wy));
                if let Some(atom) = cw.get_atom(&cp)
                    && atom.is_liquid()
                {
                    found_water_inside = true;
                    break;
                }
            }
            assert!(
                found_water_inside,
                "fluid (liquid) must be able to enter and pass through actor's footprint"
            );
        }

        // Clear object atoms
        world.run_system(clear_actor_objects_system, ());
    }

    #[test]
    fn powder_impact_slows_horizontal_momentum_and_sprays_particles() {
        let mut world = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Build a sand floor at wy = 0
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        for wx in -10..=10 {
            let cp = global_to_chunk(world_cell_to_global_atom(wx, 0));
            world.set_atom(cp, sand);
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(4.0, -3.0), // Falling at speed while running horizontally
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            None,
            Some(&materials),
            Some(&mut pool),
            None,
        );

        // Ground collision should stop vertical motion and powder should absorb horizontal momentum
        assert!(actor.grounded);
        assert_eq!(actor.vel.y, 0.0);
        assert!(
            actor.vel.x < 4.0,
            "powder landing must absorb horizontal speed, got vel.x = {}",
            actor.vel.x
        );
        assert!(
            !pool.is_empty(),
            "landing on powder at speed must spray particles into ParticlePool"
        );
    }

    #[test]
    fn fluid_drag_slows_actor_and_sprays_splashes_based_on_weight() {
        let mut world = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Fill region wy in 0..=10 with water
        let water = Atom::with_state(3, AtomState::Liquid, [40, 100, 220, 255]);
        for wy in 0..=10 {
            for wx in -10..=10 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, water);
            }
        }

        let materials = Materials::default_builtin();

        // Light actor (weight = 20.0) vs Heavy actor (weight = 100.0)
        let mut light_actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(3.0, -2.0),
            half_size: Vec2::new(2.0, 3.0),
            step_height: 2,
            weight: 20.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let mut heavy_actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(3.0, -2.0),
            half_size: Vec2::new(2.0, 3.0),
            step_height: 2,
            weight: 100.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let mut light_pool = ParticlePool::new();
        let mut heavy_pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut light_actor,
            None,
            Some(&materials),
            Some(&mut light_pool),
            None,
        );
        step_kinematic_actor_full(
            &mut world,
            &mut heavy_actor,
            None,
            Some(&materials),
            Some(&mut heavy_pool),
            None,
        );

        // Fluid drag must slow down both actors
        assert!(light_actor.vel.x < 3.0);
        assert!(heavy_actor.vel.x < 3.0);

        // Light actor experiences greater drag than heavy actor
        assert!(
            light_actor.vel.x < heavy_actor.vel.x,
            "light actor ({:.2}) should experience more fluid drag than heavy actor ({:.2})",
            light_actor.vel.x,
            heavy_actor.vel.x
        );

        // Both plunge events must spawn splash particles
        assert!(
            !light_pool.is_empty(),
            "fluid plunge must spawn splash particles"
        );
        assert!(
            !heavy_pool.is_empty(),
            "fluid plunge must spawn splash particles"
        );
    }

    #[test]
    fn sand_landing_ejects_real_matter_and_dents_floor() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Fill floor wy = 0 and wy = -1 with sand
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, sand);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(0.0, -3.5), // Falling fast onto sand
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 80.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            None,
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        assert!(actor.grounded);
        assert!(!pool.is_empty(), "landing on sand must create particles");

        // Verify that at least one particle in the pool is REAL MATTER (life_5bit == 0)
        let real_matter_count = pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() == 0)
            .count();
        assert!(
            real_matter_count > 0,
            "landing on sand must eject at least 1 real matter particle (life_5bit == 0), got {}",
            real_matter_count
        );

        // Verify that the sand floor was physically dented (at least one cell at wy = 0 became air)
        let mut air_cells_at_foot = 0;
        for wx in -2..=2 {
            let cp = global_to_chunk(world_cell_to_global_atom(wx, 0));
            if let Some(atom) = world.get_atom(&cp) {
                if atom.is_air() {
                    air_cells_at_foot += 1;
                }
            }
        }
        assert!(
            air_cells_at_foot > 0,
            "sand floor should be physically dented by ejecting powder atoms, got {} air cells",
            air_cells_at_foot
        );

        // Verify that neighbor powder cells beneath feet (wy = -1) were awakened (set_free_falling(true))
        let n_cp = global_to_chunk(world_cell_to_global_atom(0, -1));
        let wake_atom = world.get_atom(&n_cp).unwrap();
        assert!(
            wake_atom.is_free_falling(),
            "powder atoms under feet should be awakened to free-falling"
        );
    }

    #[test]
    fn stone_landing_spawns_vfx_only_chips_and_dust_without_grid_mutation() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Hard solid stone floor at wy = 0
        let stone = Atom::with_state(1, AtomState::Solid, [130, 130, 130, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, stone);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(0.0, -3.5),
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            None,
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        assert!(actor.grounded);
        assert!(
            !pool.is_empty(),
            "landing on stone should spawn VFX chips and dust"
        );

        // Verify ALL particles in pool are VFX-only (life_5bit > 0)
        for p in pool.particles() {
            assert!(
                p.atom.life_5bit() > 0,
                "stone landing particles must strictly be VFX particles (life_5bit > 0), got life_5bit = {}",
                p.atom.life_5bit()
            );
        }

        // Verify NOT A SINGLE stone cell at wy = 0 was converted to air (stone floor never damaged)
        for wx in -10..=10 {
            let cp = global_to_chunk(world_cell_to_global_atom(wx, 0));
            let atom = world.get_atom(&cp).unwrap();
            assert!(
                atom.is_solid(),
                "stone cells must never be converted to Air by actor landing! Cell at wx = {} is {:?}",
                wx,
                atom.state()
            );
        }
    }

    #[test]
    fn fluid_splash_ejects_real_liquid_matter() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Fill region wy in 0..=5 with water
        let water = Atom::with_state(3, AtomState::Liquid, [40, 100, 220, 255]);
        for wy in 0..=5 {
            for wx in -10..=10 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, water);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 3.0),
            prev_pos: Vec2::new(0.0, 3.0),
            vel: Vec2::new(2.5, -2.0),
            half_size: Vec2::new(2.0, 3.0),
            step_height: 2,
            weight: 75.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            None,
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        assert!(!pool.is_empty(), "fluid splash must spawn particles");

        // Verify that real liquid matter droplet was ejected into the pool
        let real_liquid_count = pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() == 0)
            .count();
        assert!(
            real_liquid_count > 0,
            "fluid plunge must eject at least 1 real liquid atom (life_5bit == 0), got {}",
            real_liquid_count
        );

        // Verify that VFX mist/foam particles are also present (life_5bit > 0)
        let vfx_count = pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() > 0)
            .count();
        assert!(
            vfx_count > 0,
            "fluid plunge must spawn VFX mist/foam particles alongside real droplets, got {}",
            vfx_count
        );
    }

    #[test]
    fn ejected_sand_matter_particle_reinserts_into_chunk_world() {
        use astratomic_sim::particle::step_particle_simulation;

        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Fill floor wy = 0 and wy = -1 with sand across the entire width of chunks
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        let mut initial_sand_count = 0;
        for wx in -128..=128 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, sand);
                initial_sand_count += 1;
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(0.0, -3.5),
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 80.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        // 1. Landing impact: pulls real matter sand atoms out of grid into pool
        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            None,
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        let real_particles = pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() == 0)
            .count();
        assert!(real_particles > 0, "must have ejected real matter");

        // Sand count in world is now decreased
        let mut sand_in_world = 0;
        for cy in -2..=2 {
            for cx in -2..=2 {
                if let Some(chunk) = world.get_chunk(ivec2(cx, cy)) {
                    for atom in chunk.atoms().iter() {
                        if atom.is_powder() {
                            sand_in_world += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            sand_in_world + real_particles,
            initial_sand_count,
            "matter strictly conserved during ejection"
        );

        // 2. Simulate particles until the ejected matter re-inserts
        for _ in 0..100 {
            step_particle_simulation(&mut pool, &mut world, Some(&mut dirty_rects), 1);
            if pool.particles().iter().all(|p| p.atom.life_5bit() > 0) {
                break;
            }
        }

        // 3. Verify that all real matter particles have re-inserted into ChunkWorld
        let remaining_real = pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() == 0)
            .count();
        assert_eq!(
            remaining_real, 0,
            "all real matter particles must re-insert into ChunkWorld"
        );

        let mut final_sand_in_world = 0;
        for cy in -2..=2 {
            for cx in -2..=2 {
                if let Some(chunk) = world.get_chunk(ivec2(cx, cy)) {
                    for atom in chunk.atoms().iter() {
                        if atom.is_powder() {
                            final_sand_in_world += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            final_sand_in_world, initial_sand_count,
            "matter strictly conserved after re-insertion"
        );
    }

    #[test]
    fn walking_on_ground_spawns_footstep_particles_backward() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Sand floor at wy = 0
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, sand);
            }
        }

        // Grounded actor moving right (vel.x = 2.0)
        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(2.0, 0.0),
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };
        let mut state = ActorState {
            footstep_ticks: 0, // Ready to step
            ..ActorState::default()
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            Some(&mut state),
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        assert!(
            !pool.is_empty(),
            "walking on ground must spawn footstep particles"
        );
        assert_eq!(
            state.footstep_ticks, 7,
            "footstep counter should reset to 7 ticks"
        );

        // Footstep particles should flick backward (p_vx < 0.0 when moving right)
        for p in pool.particles() {
            assert!(
                p.vel[0] < 0.0,
                "particles kicked by footsteps must travel backward opposite to movement, got vx = {}",
                p.vel[0]
            );
        }
    }

    #[test]
    fn jumping_takeoff_spawns_particles_from_ground() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Stone floor at wy = 0
        let stone = Atom::with_state(1, AtomState::Solid, [130, 130, 130, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, stone);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(0.0, 4.5), // Jump impulse upward
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: false,
            ground_normal: Vec2::Y,
        };
        let mut state = ActorState {
            jumped_this_tick: true, // Initiated jump
            ..ActorState::default()
        };

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        step_kinematic_actor_full(
            &mut world,
            &mut actor,
            Some(&mut state),
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );

        assert!(
            !pool.is_empty(),
            "jumping takeoff must kick particles from ground"
        );
        assert!(
            !state.jumped_this_tick,
            "jumped_this_tick flag must be cleared"
        );
    }

    #[test]
    fn actor_stopped_on_ground_never_emits_particles() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Sand floor at wy = 0
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, sand);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::ZERO,
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };
        let mut state = ActorState::default();
        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        // Simulate 60 ticks of standing still with gravity applied each tick
        for _ in 0..60 {
            actor.vel.y -= 0.36; // Gravity accumulator
            step_kinematic_actor_full(
                &mut world,
                &mut actor,
                Some(&mut state),
                Some(&materials),
                Some(&mut pool),
                Some(&mut dirty_rects),
            );
            assert!(actor.grounded, "actor must remain grounded");
            assert!(
                pool.is_empty(),
                "stopped actor with state on sand floor must NEVER emit particles, found {}",
                pool.particles().len()
            );
        }

        // Also test an actor WITHOUT ActorState (e.g. NPC / mob)
        let mut npc_actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::ZERO,
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };

        for _ in 0..60 {
            npc_actor.vel.y -= 0.36;
            step_kinematic_actor_full(
                &mut world,
                &mut npc_actor,
                None,
                Some(&materials),
                Some(&mut pool),
                Some(&mut dirty_rects),
            );
            assert!(npc_actor.grounded, "npc actor must remain grounded");
            assert!(
                pool.is_empty(),
                "stopped npc actor on sand floor must NEVER emit particles, found {}",
                pool.particles().len()
            );
        }
    }

    #[test]
    fn actor_stopped_in_shallow_water_never_emits_splash_particles() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Stone floor at wy = 0, water layer at wy = 1..=2
        let stone = Atom::with_state(1, AtomState::Solid, [130, 130, 130, 255]);
        let water = Atom::with_state(3, AtomState::Liquid, [40, 100, 220, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, stone);
            }
            for wy in 1..=2 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, water);
            }
        }

        let mut actor = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::ZERO,
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };
        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        // Simulate 30 ticks of standing still in puddle
        for _ in 0..30 {
            actor.vel.y -= 0.36;
            step_kinematic_actor_full(
                &mut world,
                &mut actor,
                None,
                Some(&materials),
                Some(&mut pool),
                Some(&mut dirty_rects),
            );
            assert!(
                pool.is_empty(),
                "stopped actor in shallow water must NEVER emit splash particles, found {}",
                pool.particles().len()
            );
        }
    }

    #[test]
    fn npc_without_state_jumping_and_walking_spawns_particles() {
        let mut world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Stone floor at wy = 0
        let stone = Atom::with_state(8, AtomState::Solid, [130, 130, 130, 255]);
        for wx in -10..=10 {
            for wy in -2..=0 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                world.set_atom(cp, stone);
            }
        }

        let materials = Materials::default_builtin();
        let mut pool = ParticlePool::new();

        // 1. NPC Jump takeoff from ground (state == None)
        let mut npc = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(0.0, 4.0), // Upward jump velocity
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true, // Grounded at start of tick
            ground_normal: Vec2::Y,
        };

        step_kinematic_actor_full(
            &mut world,
            &mut npc,
            None,
            Some(&materials),
            Some(&mut pool),
            Some(&mut dirty_rects),
        );
        assert!(
            !pool.is_empty(),
            "npc without state must spawn takeoff particles when jumping from ground"
        );
        pool.clear();

        // 2. NPC Walking on ground across 6 pixels (state == None)
        let mut walking_npc = KinematicActor {
            pos: Vec2::new(0.0, 5.0),
            prev_pos: Vec2::new(0.0, 5.0),
            vel: Vec2::new(1.5, 0.0),
            half_size: Vec2::new(2.0, 4.0),
            step_height: 2,
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        };

        // Walk 6 ticks (9 pixels)
        for _ in 0..6 {
            walking_npc.vel = Vec2::new(1.5, -0.36);
            step_kinematic_actor_full(
                &mut world,
                &mut walking_npc,
                None,
                Some(&materials),
                Some(&mut pool),
                Some(&mut dirty_rects),
            );
        }
        assert!(
            !pool.is_empty(),
            "npc walking on ground must spawn footstep particles"
        );
    }
}
