//! Deterministic `KinematicActor` component, `ActorState` integer tick counters,
//! and AABB grid-sweep collision over [`ChunkWorld`].

use astratomic_sim::{
    chunk::CHUNK_SIZE,
    consts::{FUEL_MAX, PLAYER_STEP_HEIGHT},
    world::{ChunkWorld, SimulationSettings, global_to_chunk},
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
}

impl Default for ActorState {
    fn default() -> Self {
        Self {
            jetpack_fuel: FUEL_MAX,
            max_jetpack_fuel: FUEL_MAX,
            fire_cooldown_ticks: 0,
            coyote_ticks: 0,
            buried_ticks: 0,
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

/// Steps a single [`KinematicActor`] (and optional [`ActorState`]) through [`ChunkWorld`]
/// using the exact Step A (horizontal sweep + foot inset + auto-step-up) and Step B (vertical sweep)
/// rules.
pub fn step_kinematic_actor(
    chunk_world: &ChunkWorld,
    actor: &mut KinematicActor,
    mut state: Option<&mut ActorState>,
) {
    // Save `pos` into `prev_pos` at the start of every tick for render interpolation & Verlet physics.
    actor.prev_pos = actor.pos;

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

        for _ in 0..substeps {
            let new_x = actor.pos.x + step_dx;
            let bottom = actor.pos.y - half_h;
            let top = actor.pos.y + half_h;

            let new_left = new_x - half_w + SKIN;
            let new_right = new_x + half_w - SKIN;

            // Test upper body from `bottom + step_height` to `top` first.
            let upper_bottom = (bottom + step_h_f32).min(top - SKIN * 2.0) + SKIN;
            let upper_top = top - SKIN;
            let upper_clear =
                !aabb_has_solid(chunk_world, new_left, new_right, upper_bottom, upper_top);

            if upper_clear {
                // Check the feet region (`bottom .. bottom + step_height`).
                let feet_top = (bottom + step_h_f32).min(top) - SKIN;
                let feet_has_solid = step_height > 0
                    && aabb_has_solid(chunk_world, new_left, new_right, bottom + SKIN, feet_top);

                if !feet_has_solid {
                    actor.pos.x = new_x;
                } else {
                    // Scan upward `h = 1..=step_height` to find the lowest `h` where the full AABB
                    // shifted by `(step_dx, h as f32)` is completely free of solid atoms.
                    let mut stepped_up = false;
                    for h in 1..=step_height {
                        let candidate_y = actor.pos.y + h as f32;
                        let cand_bottom = candidate_y - half_h + SKIN;
                        let cand_top = candidate_y + half_h - SKIN;
                        if !aabb_has_solid(
                            chunk_world,
                            new_left,
                            new_right,
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
                        snap_horizontal_obstacle(chunk_world, actor, new_x, step_dx > 0.0);
                        break;
                    }
                }
            } else {
                snap_horizontal_obstacle(chunk_world, actor, new_x, step_dx > 0.0);
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
    let mut hit_floor = false;

    if dy.abs() > 0.0 {
        let substeps = dy.abs().ceil().max(1.0) as i32;
        let step_dy = dy / substeps as f32;

        for _ in 0..substeps {
            let new_y = actor.pos.y + step_dy;
            let new_bottom = new_y - half_h + SKIN;
            let new_top = new_y + half_h - SKIN;

            if !aabb_has_solid(chunk_world, inset_left, inset_right, new_bottom, new_top) {
                actor.pos.y = new_y;
            } else {
                if step_dy < 0.0 {
                    // Falling (`-Y`) -> floor hit.
                    let wy = new_bottom.floor() as i32;
                    actor.pos.y = ((wy + 1) as f32) + half_h + SKIN;
                    actor.vel.y = 0.0;
                    hit_floor = true;
                } else {
                    // Jumping (`+Y`) -> ceiling hit.
                    let wy = new_top.floor() as i32;
                    actor.pos.y = (wy as f32) - half_h - SKIN;
                    actor.vel.y = 0.0;
                }
                break;
            }
        }
    }

    // Also check a thin ground probe beneath the feet when `vel.y <= 0.0` so walking along flat
    // ground or stepping down 1 pixel maintains stable grounding.
    if !hit_floor && actor.vel.y <= 0.0 {
        let probe_bottom = (actor.pos.y - half_h) - (SKIN * 4.0);
        let probe_top = (actor.pos.y - half_h) - (SKIN * 0.5);
        if aabb_has_solid(chunk_world, inset_left, inset_right, probe_bottom, probe_top) {
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

/// Snaps `actor.pos.x` to the exact obstacle face (`±SKIN`) using `.floor() as i32` and zeroes `vel.x`.
fn snap_horizontal_obstacle(
    chunk_world: &ChunkWorld,
    actor: &mut KinematicActor,
    attempted_x: f32,
    moving_right: bool,
) {
    let half_w = actor.half_size.x.max(0.5);
    let half_h = actor.half_size.y.max(0.5);
    let bottom = actor.pos.y - half_h + SKIN;
    let top = actor.pos.y + half_h - SKIN;
    let y0 = bottom.floor() as i32;
    let y1 = top.floor() as i32;

    if moving_right {
        let start_wx = (actor.pos.x + half_w - SKIN).floor() as i32;
        let end_wx = (attempted_x + half_w - SKIN).floor() as i32;
        let mut hit_wx = end_wx;
        'search_right: for wx in start_wx..=end_wx {
            for wy in y0..=y1 {
                if is_world_cell_solid(chunk_world, wx, wy) {
                    hit_wx = wx;
                    break 'search_right;
                }
            }
        }
        actor.pos.x = (hit_wx as f32) - half_w - SKIN;
    } else {
        let start_wx = (attempted_x - half_w + SKIN).floor() as i32;
        let end_wx = (actor.pos.x - half_w + SKIN).floor() as i32;
        let mut hit_wx = start_wx;
        'search_left: for wx in (start_wx..=end_wx).rev() {
            for wy in y0..=y1 {
                if is_world_cell_solid(chunk_world, wx, wy) {
                    hit_wx = wx;
                    break 'search_left;
                }
            }
        }
        actor.pos.x = ((hit_wx + 1) as f32) + half_w + SKIN;
    }

    actor.vel.x = 0.0;
}

/// Deterministic [`KinematicActor`] grid sweep system reading `Res<ChunkWorld>`.
pub fn kinematic_actor_movement_system(
    sim_settings: Option<Res<SimulationSettings>>,
    chunk_world: Res<ChunkWorld>,
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

    let mut actor_entities: Vec<Entity> = entities.iter_with(&actors).map(|(ent, _)| ent).collect();
    actor_entities.sort_unstable();

    for ent in actor_entities {
        let Some(actor) = actors.get_mut(ent) else {
            continue;
        };
        let state = states.get_mut(ent);
        step_kinematic_actor(&chunk_world, actor, state);
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
            grounded: false,
            ground_normal: Vec2::Y,
        };
        let mut state = ActorState::default();

        // Step repeatedly while pressing left against the wall at wx = -1:
        // actor must slide down smoothly to the floor at y = 6.001 without sticking on the wall!
        for _ in 0..10 {
            actor.vel.x = -2.5;
            actor.vel.y = -1.5;
            step_kinematic_actor(&world, &mut actor, Some(&mut state));
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
            grounded: true,
            ground_normal: Vec2::Y,
        };

        step_kinematic_actor(&world, &mut actor, None);
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
        step_kinematic_actor(&world, &mut actor, None);
        assert_eq!(
            actor.vel.x, 0.0,
            "wall taller than PLAYER_STEP_HEIGHT must block horizontal motion"
        );
        assert!((actor.pos.x - (12.0 - 3.0 - SKIN)).abs() < 1e-4);
    }
}
