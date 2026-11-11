//! Deterministic, multiplayer-ready particle simulation resource and system for `bones_ecs`.

use std::sync::Arc;

use bones_framework::prelude::*;
use bones_schema::HasSchema;

use crate::{
    atom::Atom,
    world::{ChunkWorld, DirtyRects, global_to_chunk, update_dirty_rects, update_dirty_rects_3x3},
};

/// Reuses `FLAG_SLIDE_LEFT` while in flight to indicate that an ephemeral FX particle bounces.
pub const FLAG_PARTICLE_BOUNCE: u8 = 1 << 2;

/// Particle actively steers and accelerates toward its target actor's center position each tick.
pub const PARTICLE_FOLLOW_ACTOR: u8 = 1 << 0;

/// Particle skips Bresenham/DDA ChunkWorld raycast collision and flies straight through solid terrain.
pub const PARTICLE_IGNORE_WORLD_COLLISION: u8 = 1 << 1;

/// Controls what happens when a following particle reaches its target actor's pickup radius:
/// - If unset (`0`) -> Despawn: The particle is absorbed and removed from ParticlePool without writing to ChunkWorld.
/// - If set (`1`) -> Re-insert: The particle places its Atom back into ChunkWorld at its current grid cell.
pub const PARTICLE_REINSERT_ON_TOUCH: u8 = 1 << 2;

/// Maximum number of active particles permitted in the particle pool.
pub const MAX_PARTICLES: usize = 16_384;

/// Downward gravity acceleration per simulation tick applied to particles (+Y is down).
pub const PARTICLE_GRAVITY: f32 = 0.25;

/// Air drag multiplier applied to particle velocity each tick.
pub const PARTICLE_AIR_DRAG: f32 = 0.995;

/// Maximum velocity magnitude in pixels per tick.
pub const MAX_PARTICLE_SPEED: f32 = 24.0;

/// Compact 32-byte POD particle representation stored in a flat array.
#[derive(Clone, Copy, Default, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, HasSchema)]
#[schema(opaque)]
#[repr(C)]
pub struct Particle {
    /// Sub-pixel world coordinates (x, y) in simulation pixel space.
    pub pos: [f32; 2],
    /// Velocity in simulation pixels per tick (vx, vy).
    pub vel: [f32; 2],
    /// The 8-byte [`Atom`] (preserves `material_id`, RGBA8 color, flags, and 5-bit life).
    pub atom: Atom,
    /// Target actor ID (`0` = None, `1..=255` = actor/player ID).
    pub target_actor_id: u8,
    /// Behavior flags (`PARTICLE_FOLLOW_ACTOR`, `PARTICLE_IGNORE_WORLD_COLLISION`, `PARTICLE_REINSERT_ON_TOUCH`).
    pub flags: u8,
    /// Explicit padding to ensure 8-byte aligned block (total size 32 bytes) with zero compiler padding.
    pub _padding: [u8; 6],
}

const _: [(); 32] = [(); std::mem::size_of::<Particle>()];

impl Particle {
    /// Create a standard particle with default zero flags and target.
    #[inline]
    pub fn new(pos: [f32; 2], vel: [f32; 2], atom: Atom) -> Self {
        Self {
            pos,
            vel,
            atom,
            target_actor_id: 0,
            flags: 0,
            _padding: [0; 6],
        }
    }

    /// Create a particle targeting an actor with specific behavior flags.
    #[inline]
    pub fn new_homing(
        pos: [f32; 2],
        vel: [f32; 2],
        atom: Atom,
        target_actor_id: u8,
        flags: u8,
    ) -> Self {
        Self {
            pos,
            vel,
            atom,
            target_actor_id,
            flags: flags | PARTICLE_FOLLOW_ACTOR,
            _padding: [0; 6],
        }
    }
}

/// Target actor center position and pickup radius for homing particles.
#[derive(Clone, Copy, Debug, PartialEq, HasSchema, Default)]
#[schema(opaque)]
#[repr(C)]
pub struct ParticleActorTarget {
    /// Deterministic actor ID (e.g. `1..=4` for players).
    pub actor_id: u8,
    /// Center position in simulation pixel space (`+Y` down).
    pub center_pos: [f32; 2],
    /// Pickup contact radius in simulation pixels (e.g. `6.0`).
    pub pickup_radius: f32,
}

/// Lookup resource mapping actor IDs to positions for homing particles.
#[derive(Clone, Debug, PartialEq, HasSchema, Default)]
#[schema(opaque)]
pub struct ParticleActorTargets {
    pub targets: Vec<ParticleActorTarget>,
}

impl ParticleActorTargets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_target(&mut self, actor_id: u8, center_pos: [f32; 2], pickup_radius: f32) {
        if let Some(t) = self.targets.iter_mut().find(|t| t.actor_id == actor_id) {
            t.center_pos = center_pos;
            t.pickup_radius = pickup_radius;
        } else {
            self.targets.push(ParticleActorTarget {
                actor_id,
                center_pos,
                pickup_radius,
            });
        }
    }

    pub fn clear(&mut self) {
        self.targets.clear();
    }
}

/// Simplified collision proxy for an active dynamic rigidbody used to prevent particles from solidifying inside it.
#[derive(Clone, Debug, Default, HasSchema)]
#[schema(opaque)]
pub struct ParticleRigidBodyObstacle {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub pos: [f32; 2],
    pub sin_a: f32,
    pub cos_a: f32,
    pub width: u16,
    pub height: u16,
    pub atoms: Arc<Vec<Atom>>,
}

impl ParticleRigidBodyObstacle {
    /// Tests if world cell `(wx, wy)` contains a solid atom in this rigidbody.
    #[inline]
    pub fn sample_rigidbody_at_world_cell(&self, wx: i32, wy: i32) -> Option<Atom> {
        let dx = (wx as f32 + 0.5) - self.pos[0];
        let dy = (wy as f32 + 0.5) - self.pos[1];
        let half_w = self.width as f32 * 0.5;
        let half_h = self.height as f32 * 0.5;
        let local_x = (dx * self.cos_a + dy * self.sin_a + half_w).floor() as i32;
        let local_y = (-dx * self.sin_a + dy * self.cos_a + half_h).floor() as i32;

        if local_x >= 0 && local_x < self.width as i32 && local_y >= 0 && local_y < self.height as i32 {
            let idx = (local_y as usize) * (self.width as usize) + (local_x as usize);
            self.atoms.get(idx).copied()
        } else {
            None
        }
    }
}

/// Resource holding active dynamic rigidbody obstacle proxies for particle collision safety.
#[derive(Clone, Debug, Default, HasSchema)]
#[schema(opaque)]
pub struct ParticleRigidBodyObstacles {
    pub obstacles: Vec<ParticleRigidBodyObstacle>,
}

/// Unified pool resource handling both matter-carrying atoms and ephemeral FX particles.
///
/// Backed by an `Arc<Vec<Particle>>` with Copy-on-Write semantics so rollback snapshots
/// are zero-copy when idle.
#[derive(Clone, HasSchema, Default, Debug)]
pub struct ParticlePool {
    pub active: Arc<Vec<Particle>>,
}

impl ParticlePool {
    /// Create a new empty particle pool.
    pub fn new() -> Self {
        Self {
            active: Arc::new(Vec::new()),
        }
    }

    /// Push a particle to the pool if below [`MAX_PARTICLES`].
    pub fn push_particle(&mut self, particle: Particle) {
        let list = Arc::make_mut(&mut self.active);
        if list.len() < MAX_PARTICLES {
            list.push(particle);
        }
    }

    /// Eject a matter-carrying atom from [`ChunkWorld`] into the particle pool.
    ///
    /// Reads `atom` at `(gx, gy)`. If not air or an immovable object, replaces it with
    /// [`Atom::AIR`] and spawns a real-matter particle (`life_5bit() == 0`).
    pub fn eject_from_grid(
        &mut self,
        world: &mut ChunkWorld,
        gx: i32,
        gy: i32,
        vel: [f32; 2],
    ) -> bool {
        let chunk_pos = global_to_chunk(ivec2(gx, gy));
        let Some(&existing) = world.get_atom(&chunk_pos) else {
            return false;
        };
        if existing.is_air() || existing.is_object() {
            return false;
        }

        let mut atom = existing;
        if !world.set_atom(chunk_pos, Atom::AIR) {
            return false;
        }

        atom.set_free_falling(true);
        atom.set_life_5bit(0);
        atom.vx = 0;
        atom.vy = 0;

        self.push_particle(Particle::new([gx as f32 + 0.5, gy as f32 + 0.5], vel, atom));
        true
    }

    /// Eject a matter-carrying atom and spawn it as a homing particle targeting an actor.
    pub fn eject_homing_from_grid(
        &mut self,
        world: &mut ChunkWorld,
        gx: i32,
        gy: i32,
        vel: [f32; 2],
        target_actor_id: u8,
        flags: u8,
    ) -> bool {
        let chunk_pos = global_to_chunk(ivec2(gx, gy));
        let Some(&existing) = world.get_atom(&chunk_pos) else {
            return false;
        };
        if existing.is_air() || existing.is_object() {
            return false;
        }

        let mut atom = existing;
        if !world.set_atom(chunk_pos, Atom::AIR) {
            return false;
        }

        atom.set_free_falling(true);
        atom.set_life_5bit(0);
        atom.vx = 0;
        atom.vy = 0;

        self.push_particle(Particle::new_homing(
            [gx as f32 + 0.5, gy as f32 + 0.5],
            vel,
            atom,
            target_actor_id,
            flags,
        ));
        true
    }

    /// Eject a homing particle and mark dirty rects.
    pub fn eject_homing_from_grid_dirty(
        &mut self,
        world: &mut ChunkWorld,
        dirty_rects: &mut DirtyRects,
        gx: i32,
        gy: i32,
        vel: [f32; 2],
        target_actor_id: u8,
        flags: u8,
    ) -> bool {
        let success = self.eject_homing_from_grid(world, gx, gy, vel, target_actor_id, flags);
        if success {
            let chunk_pos = global_to_chunk(ivec2(gx, gy));
            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
        }
        success
    }

    /// Eject a matter-carrying atom and mark dirty rects.
    pub fn eject_from_grid_dirty(
        &mut self,
        world: &mut ChunkWorld,
        dirty_rects: &mut DirtyRects,
        gx: i32,
        gy: i32,
        vel: [f32; 2],
    ) -> bool {
        let success = self.eject_from_grid(world, gx, gy, vel);
        if success {
            let chunk_pos = global_to_chunk(ivec2(gx, gy));
            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
        }
        success
    }

    /// Eject a matter-carrying atom from (orig_gx, orig_gy) in the grid, but spawn it
    /// as a particle at (spawn_gx, spawn_gy), marking dirty rects if provided.
    pub fn eject_atom_at_pos_dirty(
        &mut self,
        world: &mut ChunkWorld,
        mut dirty_rects: Option<&mut DirtyRects>,
        orig_gx: i32,
        orig_gy: i32,
        spawn_gx: i32,
        spawn_gy: i32,
        vel: [f32; 2],
    ) -> bool {
        let chunk_pos = global_to_chunk(ivec2(orig_gx, orig_gy));
        let Some(&existing) = world.get_atom(&chunk_pos) else {
            return false;
        };
        if existing.is_air() || existing.is_object() {
            return false;
        }

        let mut atom = existing;
        if !world.set_atom(chunk_pos, Atom::AIR) {
            return false;
        }

        if let Some(ref mut dr) = dirty_rects {
            update_dirty_rects_3x3(&mut dr.current, chunk_pos);
            update_dirty_rects(&mut dr.render, chunk_pos);
        }

        atom.set_free_falling(true);
        atom.set_life_5bit(0);
        atom.vx = 0;
        atom.vy = 0;

        self.push_particle(Particle::new(
            [spawn_gx as f32 + 0.5, spawn_gy as f32 + 0.5],
            vel,
            atom,
        ));
        true
    }

    /// Spawn an ephemeral visual effects particle.
    pub fn spawn_fx(
        &mut self,
        pos: [f32; 2],
        vel: [f32; 2],
        color: [u8; 4],
        lifetime_ticks: u8,
        bounces: bool,
    ) {
        let mut atom = Atom {
            color,
            material_id: 0,
            flags: if bounces { FLAG_PARTICLE_BOUNCE } else { 0 },
            vx: 0,
            vy: 0,
        };
        atom.set_life_5bit(lifetime_ticks.clamp(1, 31));
        self.push_particle(Particle::new(pos, vel, atom));
    }

    /// Returns the number of active particles.
    #[inline]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Returns `true` if there are no active particles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Clear all active particles.
    pub fn clear(&mut self) {
        if !self.active.is_empty() {
            self.active = Arc::new(Vec::new());
        }
    }

    /// Read-only slice view of all active particles.
    #[inline]
    pub fn particles(&self) -> &[Particle] {
        &self.active
    }

    /// Mutable slice view of all active particles via Copy-on-Write (`Arc::make_mut`).
    #[inline]
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        Arc::make_mut(&mut self.active).as_mut_slice()
    }
}

/// Pure integer Bresenham line iterator stepping from `start` to `end` (skipping `start`).
#[derive(Debug, Clone, Copy)]
pub struct Bresenham {
    curr_x: i32,
    curr_y: i32,
    end_x: i32,
    end_y: i32,
    dx: i32,
    dy: i32,
    sx: i32,
    sy: i32,
    err: i32,
}

impl Bresenham {
    /// Create a new Bresenham line iterator from `start` to `end`.
    pub fn new(start: (i32, i32), end: (i32, i32)) -> Self {
        let dx = (end.0 - start.0).abs();
        let dy = -(end.1 - start.1).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let err = dx + dy;
        Self {
            curr_x: start.0,
            curr_y: start.1,
            end_x: end.0,
            end_y: end.1,
            dx,
            dy,
            sx,
            sy,
            err,
        }
    }
}

impl Iterator for Bresenham {
    type Item = (i32, i32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_x == self.end_x && self.curr_y == self.end_y {
            return None;
        }
        let e2 = 2 * self.err;
        if e2 >= self.dy {
            self.err += self.dy;
            self.curr_x += self.sx;
        }
        if e2 <= self.dx {
            self.err += self.dx;
            self.curr_y += self.sy;
        }
        Some((self.curr_x, self.curr_y))
    }
}

/// Re-insert a matter-carrying atom back into [`ChunkWorld`] at or near `(lx, ly)`.
///
/// If `(lx, ly)` is air, the atom is written directly. If already occupied, searches
/// deterministic rings (`radius = 1..=2`) in fixed offset order for the first air cell
/// to guarantee matter conservation.
pub fn reinsert_particle_atom(
    world: &mut ChunkWorld,
    lx: i32,
    ly: i32,
    vel: [f32; 2],
    atom: Atom,
    current_tick: u64,
) -> bool {
    reinsert_particle_atom_with_dirty(world, None, lx, ly, vel, atom, current_tick)
}

/// Re-insert a matter-carrying atom back into [`ChunkWorld`] and update dirty rects if provided.
pub fn reinsert_particle_atom_with_dirty(
    world: &mut ChunkWorld,
    mut dirty_rects: Option<&mut DirtyRects>,
    lx: i32,
    ly: i32,
    vel: [f32; 2],
    mut atom: Atom,
    current_tick: u64,
) -> bool {
    atom.set_free_falling(true);
    atom.mark_updated(current_tick);
    atom.vx = (vel[0] * 0.5).round().clamp(-6.0, 6.0) as i8;
    atom.vy = 0;

    // 1. Check primary target cell (lx, ly)
    let primary_pos = global_to_chunk(ivec2(lx, ly));
    if let Some(target) = world.get_atom(&primary_pos) {
        if target.is_air() {
            world.set_atom(primary_pos, atom);
            if let Some(ref mut dr) = dirty_rects {
                update_dirty_rects_3x3(&mut dr.current, primary_pos);
                update_dirty_rects(&mut dr.render, primary_pos);
            }
            return true;
        }
    }

    // 2. Search deterministic neighbor rings radius = 1..=2
    for radius in 1i32..=2 {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dy.abs()) == radius {
                    let nx = lx + dx;
                    let ny = ly + dy;
                    let npos = global_to_chunk(ivec2(nx, ny));
                    if let Some(target) = world.get_atom(&npos) {
                        if target.is_air() {
                            world.set_atom(npos, atom);
                            if let Some(ref mut dr) = dirty_rects {
                                update_dirty_rects_3x3(&mut dr.current, npos);
                                update_dirty_rects(&mut dr.render, npos);
                            }
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

/// Advance all particles in `pool` by one deterministic simulation tick, taking actor targets for homing.
pub fn step_particle_simulation_with_targets(
    pool: &mut ParticlePool,
    world: &mut ChunkWorld,
    dirty_rects: Option<&mut DirtyRects>,
    current_tick: u64,
    actor_targets: &[ParticleActorTarget],
) {
    step_particle_simulation_full(pool, world, dirty_rects, current_tick, actor_targets, &[]);
}

/// Advance all particles in `pool` by one deterministic simulation tick, taking actor targets and rigidbody obstacles.
pub fn step_particle_simulation_full(
    pool: &mut ParticlePool,
    world: &mut ChunkWorld,
    mut dirty_rects: Option<&mut DirtyRects>,
    current_tick: u64,
    actor_targets: &[ParticleActorTarget],
    rigidbody_obstacles: &[ParticleRigidBodyObstacle],
) {
    if pool.active.is_empty() {
        return;
    }

    let list = Arc::make_mut(&mut pool.active);

    list.retain_mut(|p| {
        // 1. Ephemeral FX Lifetime Check
        let life = p.atom.life_5bit();
        let is_fx = life > 0;
        if is_fx {
            if life == 1 {
                return false; // Expired in mid-air -> despawn
            }
            p.atom.set_life_5bit(life - 1);
        }

        // 2. Homing vs Standard Physics
        let is_homing_requested = (p.flags & PARTICLE_FOLLOW_ACTOR) != 0 && p.target_actor_id != 0;
        let target = if is_homing_requested {
            actor_targets
                .iter()
                .find(|t| t.actor_id == p.target_actor_id)
        } else {
            None
        };

        if let Some(target) = target {
            let target_pos = target.center_pos;
            let pickup_radius = target.pickup_radius;

            let to_target = [target_pos[0] - p.pos[0], target_pos[1] - p.pos[1]];
            let dist_sq = to_target[0] * to_target[0] + to_target[1] * to_target[1];
            let dist = dist_sq.sqrt();

            // Actor Touch Check at current position:
            if dist <= pickup_radius {
                if (p.flags & PARTICLE_REINSERT_ON_TOUCH) != 0 {
                    reinsert_particle_atom_with_dirty(
                        world,
                        dirty_rects.as_deref_mut(),
                        p.pos[0].floor() as i32,
                        p.pos[1].floor() as i32,
                        p.vel,
                        p.atom,
                        current_tick,
                    );
                }
                return false; // Despawn on touch or re-inserted into grid
            }

            // Homing Steering Physics matching original codebase (PartState::Follow):
            // Directly steer toward target nozzle and smoothly accelerate
            if dist > 1e-4 {
                let dir = [to_target[0] / dist, to_target[1] / dist];
                let cur_mag = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1])
                    .sqrt()
                    .clamp(0.0, 6.0);
                let new_mag = cur_mag + 0.5;
                p.vel[0] = dir[0] * new_mag;
                p.vel[1] = dir[1] * new_mag;
            }
        } else {
            // Standard Gravity & Air Drag (+Y is down):
            p.vel[1] += PARTICLE_GRAVITY;
            p.vel[0] *= PARTICLE_AIR_DRAG;
            p.vel[1] *= PARTICLE_AIR_DRAG;
        }

        let speed_sq = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1];
        if speed_sq > MAX_PARTICLE_SPEED * MAX_PARTICLE_SPEED {
            let speed = speed_sq.sqrt();
            let scale = MAX_PARTICLE_SPEED / speed;
            p.vel[0] *= scale;
            p.vel[1] *= scale;
        }

        let new_pos = [p.pos[0] + p.vel[0], p.pos[1] + p.vel[1]];

        // Actor Touch Check at new_pos to prevent tunneling past the actor when moving fast:
        if let Some(target) = target {
            let new_to_target = [
                target.center_pos[0] - new_pos[0],
                target.center_pos[1] - new_pos[1],
            ];
            let new_dist_sq =
                new_to_target[0] * new_to_target[0] + new_to_target[1] * new_to_target[1];
            if new_dist_sq <= target.pickup_radius * target.pickup_radius {
                if (p.flags & PARTICLE_REINSERT_ON_TOUCH) != 0 {
                    reinsert_particle_atom_with_dirty(
                        world,
                        dirty_rects.as_deref_mut(),
                        new_pos[0].floor() as i32,
                        new_pos[1].floor() as i32,
                        p.vel,
                        p.atom,
                        current_tick,
                    );
                }
                return false;
            }
        }

        // 3. World Collision Check (PARTICLE_IGNORE_WORLD_COLLISION / Noclip):
        if (p.flags & PARTICLE_IGNORE_WORLD_COLLISION) != 0 {
            p.pos = new_pos;
            return true;
        }

        // 4. Bresenham / Integer DDA Line Trace
        let start = (p.pos[0].floor() as i32, p.pos[1].floor() as i32);
        let end = (new_pos[0].floor() as i32, new_pos[1].floor() as i32);

        let mut last_free = start;
        for (cx, cy) in Bresenham::new(start, end) {
            let Some(atom) = world.get_atom_global(cx, cy) else {
                // In an unloaded chunk -> drop particle
                return false;
            };

            if !atom.is_air() {
                let (lx, ly) = last_free;
                if !is_fx {
                    // Safety net: check if (lx, ly) is currently covered by any PixelRigidBody
                    let wx = lx;
                    let wy = -ly - 1;
                    let mut covered_by_rb = false;
                    let mut deflect_dir = [0.0f32, 0.0f32];

                    for obs in rigidbody_obstacles {
                        let cx = wx as f32 + 0.5;
                        let cy = wy as f32 + 0.5;
                        if cx >= obs.min_x && cx <= obs.max_x && cy >= obs.min_y && cy <= obs.max_y {
                            if let Some(rb_atom) = obs.sample_rigidbody_at_world_cell(wx, wy) {
                                if !rb_atom.is_air() && !rb_atom.is_void() {
                                    covered_by_rb = true;
                                    let dx = cx - obs.pos[0];
                                    let dy = cy - obs.pos[1];
                                    let sx = if dx.abs() > 0.1 { dx.signum() } else { 1.0 };
                                    let sy = if dy.abs() > 0.1 { dy.signum() } else { 1.0 };
                                    deflect_dir = [sx, -sy]; // In sim coords, Y is inverted
                                    break;
                                }
                            }
                        }
                    }

                    if covered_by_rb {
                        // Push particle outward by 1.5 pixels and keep it flying for 1 more tick
                        p.pos[0] += deflect_dir[0] * 1.5;
                        p.pos[1] += deflect_dir[1] * 1.5;
                        p.vel[0] = deflect_dir[0] * 2.5;
                        p.vel[1] = deflect_dir[1] * 2.5;
                        return true;
                    }

                    // Case A: Real Matter -> re-insert into grid
                    reinsert_particle_atom_with_dirty(
                        world,
                        dirty_rects.as_deref_mut(),
                        lx,
                        ly,
                        p.vel,
                        p.atom,
                        current_tick,
                    );
                    return false;
                } else if (p.atom.flags & FLAG_PARTICLE_BOUNCE) != 0 {
                    // Case B: Ephemeral FX with FLAG_PARTICLE_BOUNCE
                    if cx != lx {
                        p.vel[0] = -p.vel[0] * 0.55;
                    }
                    if cy != ly {
                        p.vel[1] = -p.vel[1] * 0.55;
                    }
                    p.pos = [lx as f32 + 0.5, ly as f32 + 0.5];
                    return true;
                } else {
                    // Case C: Ephemeral FX without bounce
                    return false;
                }
            }

            last_free = (cx, cy);
        }

        // 5. Unobstructed Move
        p.pos = new_pos;
        true
    });
}

/// Advance all particles in `pool` by one deterministic simulation tick (without homing targets).
pub fn step_particle_simulation(
    pool: &mut ParticlePool,
    world: &mut ChunkWorld,
    dirty_rects: Option<&mut DirtyRects>,
    current_tick: u64,
) {
    step_particle_simulation_full(pool, world, dirty_rects, current_tick, &[], &[]);
}

/// Bones ECS system advancing the [`ParticlePool`] simulation.
pub fn simulate_particles_system(
    mut pool: ResMut<ParticlePool>,
    mut world: ResMut<ChunkWorld>,
    mut dirty_rects: Option<ResMut<DirtyRects>>,
    actor_targets: Option<Res<ParticleActorTargets>>,
    rigidbody_obstacles: Option<Res<ParticleRigidBodyObstacles>>,
) {
    let current_tick = world.tick;
    let targets = actor_targets
        .as_ref()
        .map(|t| t.targets.as_slice())
        .unwrap_or(&[]);
    let obstacles = rigidbody_obstacles
        .as_ref()
        .map(|o| o.obstacles.as_slice())
        .unwrap_or(&[]);
    step_particle_simulation_full(
        &mut pool,
        &mut world,
        dirty_rects.as_deref_mut(),
        current_tick,
        targets,
        obstacles,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{atom::Materials, chunk::Chunk, plugin::register_sim_schemas};

    #[test]
    fn test_particle_data_layout() {
        assert_eq!(std::mem::size_of::<Particle>(), 32);
        assert_eq!(std::mem::align_of::<Particle>(), 4);

        let zeroed = Particle::default();
        let bytes: [u8; 32] = bytemuck::cast(zeroed);
        assert_eq!(bytes, [0u8; 32]);

        let p = Particle {
            pos: [12.5, -45.75],
            vel: [1.25, -2.5],
            atom: Atom {
                color: [255, 128, 64, 255],
                material_id: 2,
                flags: FLAG_PARTICLE_BOUNCE,
                vx: -3,
                vy: 5,
            },
            target_actor_id: 3,
            flags: PARTICLE_FOLLOW_ACTOR | PARTICLE_IGNORE_WORLD_COLLISION,
            _padding: [0; 6],
        };
        let cast_bytes: [u8; 32] = bytemuck::cast(p);
        let back: Particle = bytemuck::cast(cast_bytes);
        assert_eq!(p, back);
    }

    #[test]
    fn test_life_5bit_roundtrip() {
        let mut atom = Atom::default();
        assert_eq!(atom.life_5bit(), 0);

        atom.set_life_5bit(15);
        assert_eq!(atom.life_5bit(), 15);

        atom.set_life_5bit(31);
        assert_eq!(atom.life_5bit(), 31);

        // Clamps to 31
        atom.set_life_5bit(50);
        assert_eq!(atom.life_5bit(), 31);

        // Ensure bit 2 (FLAG_PARTICLE_BOUNCE) is preserved
        atom.flags |= FLAG_PARTICLE_BOUNCE;
        atom.set_life_5bit(7);
        assert_eq!(atom.life_5bit(), 7);
        assert_ne!(atom.flags & FLAG_PARTICLE_BOUNCE, 0);

        atom.set_life_5bit(0);
        assert_eq!(atom.life_5bit(), 0);
        assert_ne!(atom.flags & FLAG_PARTICLE_BOUNCE, 0);
    }

    #[test]
    fn test_bresenham_line() {
        // Zero length
        let pts: Vec<_> = Bresenham::new((5, 5), (5, 5)).collect();
        assert!(pts.is_empty());

        // Horizontal right
        let pts: Vec<_> = Bresenham::new((0, 0), (3, 0)).collect();
        assert_eq!(pts, vec![(1, 0), (2, 0), (3, 0)]);

        // Horizontal left
        let pts: Vec<_> = Bresenham::new((3, 0), (0, 0)).collect();
        assert_eq!(pts, vec![(2, 0), (1, 0), (0, 0)]);

        // Vertical down
        let pts: Vec<_> = Bresenham::new((0, 0), (0, 3)).collect();
        assert_eq!(pts, vec![(0, 1), (0, 2), (0, 3)]);

        // Vertical up
        let pts: Vec<_> = Bresenham::new((0, 3), (0, 0)).collect();
        assert_eq!(pts, vec![(0, 2), (0, 1), (0, 0)]);

        // Diagonal
        let pts: Vec<_> = Bresenham::new((0, 0), (2, 2)).collect();
        assert_eq!(pts, vec![(1, 1), (2, 2)]);
    }

    fn setup_test_world() -> (ChunkWorld, DirtyRects, Materials) {
        let mut world = ChunkWorld::default();
        let dirty = DirtyRects::default();
        let materials = Materials::default_builtin();

        for cx in -1..=1 {
            for cy in -1..=1 {
                world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }
        (world, dirty, materials)
    }

    #[test]
    fn test_eject_from_grid() {
        let (mut world, mut dirty, materials) = setup_test_world();
        let mut pool = ParticlePool::new();

        // Paint sand at (10, 10)
        world.paint_at_global(ivec2(10, 10), 2, false, &materials, &mut dirty);
        assert!(!world.get_atom_global(10, 10).unwrap().is_air());

        // Eject sand into particle pool
        let ejected = pool.eject_from_grid(&mut world, 10, 10, [2.0, -4.0]);
        assert!(ejected);
        assert_eq!(pool.len(), 1);

        // Grid cell is now AIR
        assert!(world.get_atom_global(10, 10).unwrap().is_air());

        let p = pool.particles()[0];
        assert_eq!(p.pos, [10.5, 10.5]);
        assert_eq!(p.vel, [2.0, -4.0]);
        assert_eq!(p.atom.material_id, 2);
        assert_eq!(p.atom.life_5bit(), 0);

        // Attempting to eject AIR returns false
        assert!(!pool.eject_from_grid(&mut world, 10, 10, [0.0, 0.0]));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_spawn_fx_and_decay() {
        let (mut world, mut dirty, _) = setup_test_world();
        let mut pool = ParticlePool::new();

        pool.spawn_fx([0.0, 0.0], [0.0, 0.0], [255, 0, 0, 255], 3, false);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.particles()[0].atom.life_5bit(), 3);

        // Tick 1: life goes from 3 to 2
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 1);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.particles()[0].atom.life_5bit(), 2);

        // Tick 2: life goes from 2 to 1
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 2);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.particles()[0].atom.life_5bit(), 1);

        // Tick 3: life was 1 -> despawns in mid-air
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 3);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_ephemeral_fx_impact_with_and_without_bounce() {
        let (mut world, mut dirty, materials) = setup_test_world();
        let mut pool = ParticlePool::new();

        // Solid rock at y = 10
        world.paint_at_global(ivec2(5, 10), 8, false, &materials, &mut dirty);

        // 1. Ephemeral FX without bounce hitting solid ground -> despawns
        pool.spawn_fx([5.5, 8.5], [0.0, 2.0], [255, 255, 255, 255], 20, false);
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 1);
        assert_eq!(pool.len(), 0);

        // 2. Ephemeral FX WITH bounce hitting solid ground -> bounces
        pool.spawn_fx([5.5, 8.5], [0.0, 2.0], [255, 255, 255, 255], 20, true);
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 2);
        assert_eq!(pool.len(), 1);
        let p = pool.particles()[0];
        // Snapped to center of last_free cell (y = 9)
        assert_eq!(p.pos, [5.5, 9.5]);
        // Bounced velocity in Y is negative
        assert!(p.vel[1] < 0.0);
    }

    #[test]
    fn test_real_matter_reinsertion_and_conservation() {
        let (mut world, mut dirty, materials) = setup_test_world();
        let mut pool = ParticlePool::new();

        // Floor of solid rock at y = 20
        for x in 0..=30 {
            world.paint_at_global(ivec2(x, 20), 8, false, &materials, &mut dirty);
        }

        // Spawn a real matter particle (sand, material_id = 2) at (10.5, 18.5) moving downward
        let mut sand_atom =
            Atom::with_state(2, crate::atom::AtomState::Powder, [230, 197, 92, 255]);
        sand_atom.set_life_5bit(0); // 0 = real matter
        pool.push_particle(Particle::new([10.5, 18.5], [0.0, 2.0], sand_atom));

        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 1);
        // Settled into grid, removed from pool
        assert_eq!(pool.len(), 0);

        // Verify sand placed at (10, 19)
        let reinserted = world.get_atom_global(10, 19).unwrap();
        assert_eq!(reinserted.material_id, 2);
        assert!(reinserted.is_free_falling());
        assert_eq!(reinserted.vx, 0);

        // Multiple particles landing on the exact same spot:
        // Spawn 3 sand particles directed at (10.5, 18.5)
        for _ in 0..3 {
            pool.push_particle(Particle::new([10.5, 18.5], [0.0, 2.0], sand_atom));
        }
        step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), 2);
        assert_eq!(pool.len(), 0);

        // Count sand atoms around x in 8..=12, y in 17..=19
        let mut sand_count = 0;
        for x in 8..=12 {
            for y in 17..=19 {
                if world
                    .get_atom_global(x, y)
                    .is_some_and(|a| a.material_id == 2)
                {
                    sand_count += 1;
                }
            }
        }
        // Total 4 sand atoms (1 initial + 3 new), none lost!
        assert_eq!(sand_count, 4);
    }

    #[test]
    fn test_particles_in_bones_ecs_stage() {
        register_sim_schemas();

        let mut session = Session::default();
        session.world.init_resource::<ChunkWorld>();
        session.world.init_resource::<DirtyRects>();
        session.world.init_resource::<ParticlePool>();

        // Populate chunks
        {
            let mut world = session.world.resource_mut::<ChunkWorld>();
            for cx in -1..=1 {
                for cy in -1..=1 {
                    world.add_chunk(ivec2(cx, cy), Chunk::default());
                }
            }
        }

        // Spawn 2 FX particles
        {
            let mut pool = session.world.resource_mut::<ParticlePool>();
            pool.spawn_fx([0.0, 0.0], [1.0, 1.0], [255, 255, 255, 255], 5, false);
            pool.spawn_fx([10.0, 10.0], [-1.0, 0.0], [200, 200, 200, 255], 2, false);
        }

        session.add_system_to_stage(CoreStage::Update, simulate_particles_system);

        // Step 1
        session.world.run_system(simulate_particles_system, ());
        {
            let pool = session.world.resource::<ParticlePool>();
            assert_eq!(pool.len(), 2);
            assert_eq!(pool.particles()[0].atom.life_5bit(), 4);
            assert_eq!(pool.particles()[1].atom.life_5bit(), 1);
        }

        // Step 2: second particle expires (life was 1)
        session.world.run_system(simulate_particles_system, ());
        {
            let pool = session.world.resource::<ParticlePool>();
            assert_eq!(pool.len(), 1);
            assert_eq!(pool.particles()[0].atom.life_5bit(), 3);
        }
    }

    #[test]
    fn test_homing_particle_reaches_actor_and_despawns() {
        let (mut world, mut dirty, _) = setup_test_world();
        let mut pool = ParticlePool::new();

        let mut sand_atom =
            Atom::with_state(2, crate::atom::AtomState::Powder, [230, 197, 92, 255]);
        sand_atom.set_life_5bit(0);

        // Spawn particle at (10.0, 10.0) targeting actor 1 at (10.0, 30.0) with outward pop velocity
        pool.push_particle(Particle {
            pos: [10.0, 10.0],
            vel: [-2.0, -1.0], // Initial outward pop velocity
            atom: sand_atom,
            target_actor_id: 1,
            flags: PARTICLE_FOLLOW_ACTOR | PARTICLE_IGNORE_WORLD_COLLISION,
            _padding: [0; 6],
        });

        let targets = vec![ParticleActorTarget {
            actor_id: 1,
            center_pos: [10.0, 30.0],
            pickup_radius: 6.0,
        }];

        // Simulate ticks until particle reaches actor and despawns
        for tick in 1..=100 {
            step_particle_simulation_with_targets(
                &mut pool,
                &mut world,
                Some(&mut dirty),
                tick,
                &targets,
            );
            if pool.is_empty() {
                break;
            }
        }

        // Particle must have reached actor and despawned without placing sand in the world
        assert_eq!(pool.len(), 0);
        let atom_at_target = world.get_atom_global(10, 30).unwrap();
        assert!(atom_at_target.is_air());
    }

    #[test]
    fn test_homing_particle_reinserts_into_world_on_touch() {
        let (mut world, mut dirty, _) = setup_test_world();
        let mut pool = ParticlePool::new();

        let mut sand_atom =
            Atom::with_state(2, crate::atom::AtomState::Powder, [230, 197, 92, 255]);
        sand_atom.set_life_5bit(0);

        // Spawn particle with PARTICLE_REINSERT_ON_TOUCH targeting actor 2
        pool.push_particle(Particle {
            pos: [10.0, 10.0],
            vel: [0.0, 0.0],
            atom: sand_atom,
            target_actor_id: 2,
            flags: PARTICLE_FOLLOW_ACTOR
                | PARTICLE_IGNORE_WORLD_COLLISION
                | PARTICLE_REINSERT_ON_TOUCH,
            _padding: [0; 6],
        });

        let targets = vec![ParticleActorTarget {
            actor_id: 2,
            center_pos: [10.0, 25.0],
            pickup_radius: 6.0,
        }];

        for tick in 1..=100 {
            step_particle_simulation_with_targets(
                &mut pool,
                &mut world,
                Some(&mut dirty),
                tick,
                &targets,
            );
            if pool.is_empty() {
                break;
            }
        }

        assert_eq!(pool.len(), 0);
        // Sand must have re-inserted into ChunkWorld near the actor (y ~ 20..=30)
        let mut found_sand = false;
        for dy in -8..=8 {
            for dx in -8..=8 {
                if let Some(atom) = world.get_atom_global(10 + dx, 25 + dy) {
                    if atom.material_id == 2 {
                        found_sand = true;
                        break;
                    }
                }
            }
        }
        assert!(
            found_sand,
            "sand must be re-inserted into ChunkWorld when PARTICLE_REINSERT_ON_TOUCH is set"
        );
    }

    #[test]
    fn test_noclip_particle_flies_through_solid_terrain() {
        let (mut world, mut dirty, materials) = setup_test_world();
        let mut pool = ParticlePool::new();

        // Solid rock wall from y = 10 to y = 20 at x in -5..=25
        for x in -5..=25 {
            for y in 10..=20 {
                world.paint_at_global(ivec2(x, y), 8, false, &materials, &mut dirty);
            }
        }

        let mut sand_atom =
            Atom::with_state(2, crate::atom::AtomState::Powder, [230, 197, 92, 255]);
        sand_atom.set_life_5bit(0);

        // Particle with PARTICLE_IGNORE_WORLD_COLLISION homing through the rock wall toward actor on the other side
        pool.push_particle(Particle {
            pos: [10.0, 5.0],
            vel: [0.0, 2.0],
            atom: sand_atom,
            target_actor_id: 1,
            flags: PARTICLE_FOLLOW_ACTOR | PARTICLE_IGNORE_WORLD_COLLISION,
            _padding: [0; 6],
        });

        let targets = vec![ParticleActorTarget {
            actor_id: 1,
            center_pos: [10.0, 28.0],
            pickup_radius: 4.0,
        }];

        // Advance 2 ticks: particle should be inside the solid wall without colliding
        for tick in 1..=2 {
            step_particle_simulation_with_targets(
                &mut pool,
                &mut world,
                Some(&mut dirty),
                tick,
                &targets,
            );
        }

        assert_eq!(pool.len(), 1);
        let p = pool.particles()[0];
        // Must have passed into/through rock wall without being blocked
        assert!(p.pos[1] > 6.0);
    }
}
