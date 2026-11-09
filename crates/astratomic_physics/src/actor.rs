//! Deterministic `KinematicActor` component, `ActorState` integer tick counters,
//! and AABB grid-sweep collision over [`ChunkWorld`].

use astratomic_sim::{
    atom::{Atom, AtomState},
    chunk::CHUNK_SIZE,
    consts::{FUEL_MAX, PLAYER_STEP_HEIGHT},
    world::{
        ChunkPos, ChunkWorld, DirtyRects, SimulationSettings, global_to_chunk,
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

/// Steps a single [`KinematicActor`] (and optional [`ActorState`]) through [`ChunkWorld`]
/// using the exact Step A (horizontal sweep + foot inset + auto-step-up) and Step B (vertical sweep)
/// rules.
///
/// Leading-edge sweeps and non-inverting clamps guarantee that overlapping atoms inside an actor's
/// body never cause reverse-direction movement or teleports.
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
                    && cells_have_solid(
                        chunk_world,
                        lead_wx_start,
                        lead_wx_end,
                        feet_y0,
                        feet_y1,
                    );

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
            grounded: true,
            ground_normal: Vec2::Y,
        };

        // When moving right: actor MUST move right and NOT snap to the left!
        step_kinematic_actor(&world, &mut actor, None);
        assert!(
            actor.pos.x > 10.0,
            "moving right away from atoms must increase pos.x, got pos.x = {}",
            actor.pos.x
        );

        // Reset to x = 10.0 and test moving left (into the atoms):
        actor.pos = Vec2::new(10.0, 6.0 + SKIN);
        actor.vel = Vec2::new(-2.0, 0.0);
        step_kinematic_actor(&world, &mut actor, None);
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

        step_kinematic_actor(&world, &mut actor, None);
        assert!(
            actor.pos.x < 10.0,
            "moving left away from atoms on the right must decrease pos.x, got pos.x = {}",
            actor.pos.x
        );

        // Reset to x = 10.0 and test moving right (into the atoms on right):
        actor.pos = Vec2::new(10.0, 6.0 + SKIN);
        actor.vel = Vec2::new(2.0, 0.0);
        step_kinematic_actor(&world, &mut actor, None);
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
            grounded: true,
            ground_normal: Vec2::Y,
        };

        step_kinematic_actor(&world, &mut actor, None);
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
                grounded: false,
                ground_normal: Vec2::Y,
            },
        );

        world.run_system(stamp_actor_objects_system, ());

        {
            let stamp = world.resource::<ActorSimulationStamp>();
            assert!(!stamp.stamped_atoms.is_empty(), "should have stamped cells for the actor");
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
        world.resource_mut::<DirtyRects>().current.insert(sand_cp.chunk, URect::full_chunk());

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
}
