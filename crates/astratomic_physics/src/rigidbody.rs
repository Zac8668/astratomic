//! Deterministic dynamic rigidbodies backed by raw `rapier2d` (`enhanced-determinism`),
//! lazy Awake-AABB chunk terrain colliders, pixel-backed dynamic bodies (`PixelRigidBody`),
//! and two-way coupling with `ChunkWorld`, `ParticlePool`, and `KinematicActor`.

use std::collections::{BTreeMap, BTreeSet};

pub use bones_framework::prelude::Vec2;
use bones_framework::prelude::*;
use rapier2d::prelude::{
    CCDSolver, ColliderBuilder, ColliderSet, DefaultBroadPhase, ImpulseJointSet,
    IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline,
    RigidBodyBuilder, RigidBodySet, SharedShape, SoftBodySet,
};
pub use rapier2d::prelude::{ColliderHandle, RigidBodyHandle, Group, InteractionGroups};
pub use rapier2d::geometry::InteractionTestMode;

/// Helper struct providing `CollisionGroups::new(memberships, filter)`.
pub struct CollisionGroups;
impl CollisionGroups {
    #[inline]
    pub const fn new(memberships: Group, filter: Group) -> InteractionGroups {
        InteractionGroups::new(memberships, filter, InteractionTestMode::And)
    }
}

/// Rapier collision layer for static chunk polyline colliders.
pub const GROUP_TERRAIN: Group = Group::GROUP_1;
/// Rapier collision layer for dynamic `PixelRigidBody` entities (crates, boulders).
pub const GROUP_DYNAMIC_BODY: Group = Group::GROUP_2;
/// Rapier collision layer for `KinematicActor` entities (players, NPCs, enemies).
pub const GROUP_ACTOR: Group = Group::GROUP_3;

/// Tuning multiplier for Archimedes buoyancy physics on submerged rigidbodies.
pub const BUOYANCY_MULTIPLIER: f32 = 1.6;

use astratomic_sim::{
    atom::{Atom, AtomState, Materials},
    chunk::Chunk,
    consts::CHUNK_SIZE,
    particle::{
        FLAG_PARTICLE_BOUNCE, ParticlePool, ParticleRigidBodyObstacle,
        ParticleRigidBodyObstacles,
    },
    world::{ChunkPos, ChunkWorld, DirtyRects, SimulationSettings, global_to_chunk, update_dirty_rects, update_dirty_rects_3x3},
};

use crate::actor::world_cell_to_global_atom;

/// Converts a [`bones_framework::prelude::Vec2`] into a [`rapier2d::math::Vec2`].
#[inline]
pub fn to_rapier_vec(v: Vec2) -> rapier2d::math::Vec2 {
    rapier2d::math::Vec2::new(v.x, v.y)
}

/// Converts a [`rapier2d::math::Vec2`] into a [`bones_framework::prelude::Vec2`].
#[inline]
pub fn from_rapier_vec(v: rapier2d::math::Vec2) -> Vec2 {
    vec2(v.x, v.y)
}

/// Packs an [`Entity`] into a deterministic 128-bit integer for Rapier's user data.
#[inline]
pub fn entity_to_user_data(ent: Entity) -> u128 {
    ((ent.index() as u128) << 32) | (ent.generation() as u128)
}

/// Unpacks an [`Entity`] from Rapier's 128-bit user data.
#[inline]
pub fn user_data_to_entity(user_data: u128) -> Entity {
    Entity::new((user_data >> 32) as u32, (user_data & 0xFFFF_FFFF) as u32)
}

/// Deterministic 64-bit noise generator returning a float in `[0.0, 1.0)`.
#[inline]
pub fn deterministic_noise(seed: u64, index: u32) -> f32 {
    let mut h = seed.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h as u32 as f32) / (u32::MAX as f32)
}

/// Simple deterministic RNG wrapper based on 64-bit splitmix hash.
#[derive(Clone, Copy, Debug)]
pub struct SimpleRng {
    seed: u64,
    counter: u32,
}

impl SimpleRng {
    #[inline]
    pub fn new(seed: u64, salt: u32) -> Self {
        Self { seed, counter: salt }
    }

    #[inline]
    pub fn f32(&mut self) -> f32 {
        self.counter = self.counter.wrapping_add(1);
        deterministic_noise(self.seed, self.counter)
    }

    #[inline]
    pub fn f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.f32() * (max - min)
    }

    #[inline]
    pub fn bool(&mut self) -> bool {
        self.f32() > 0.5
    }

    #[inline]
    pub fn u8_range(&mut self, min: u8, max: u8) -> u8 {
        let v = min as f32 + self.f32() * ((max - min) as f32 + 0.999);
        (v.floor() as u8).clamp(min, max)
    }
}

/// Steps outward from `(start_wx, start_wy)` along `(side_sign, up_sign)` until exiting the
/// `PixelRigidBody` hull into empty air in [`ChunkWorld`].
pub fn find_hull_exit_pos(
    pixel_body: &PixelRigidBody,
    body_pos: Vec2,
    sin_a: f32,
    cos_a: f32,
    chunk_world: &ChunkWorld,
    start_wx: i32,
    start_wy: i32,
    side_sign: f32,
    up_sign: f32,
) -> (i32, i32) {
    let step_x = side_sign as i32;
    let step_y = up_sign as i32;
    let max_steps = (pixel_body.width.max(pixel_body.height) as i32) + 6;

    let mut curr_wx = start_wx;
    let mut curr_wy = start_wy;

    for _ in 0..max_steps {
        curr_wx += step_x;
        curr_wy += step_y;

        let rb_sample = pixel_body.sample_rigidbody_at_world_cell(curr_wx, curr_wy, body_pos, sin_a, cos_a);
        let outside_rb = rb_sample.is_none() || rb_sample.unwrap().is_air();

        if outside_rb {
            let cp = global_to_chunk(world_cell_to_global_atom(curr_wx, curr_wy));
            let is_air_in_world = match chunk_world.get_atom(&cp) {
                Some(a) => a.is_air() || a.is_void(),
                None => true,
            };
            if is_air_in_world {
                return (curr_wx, curr_wy);
            }
        }
    }

    // Fallback: step purely along up_sign from the hull edge to clear the body
    let fallback_wx = start_wx + step_x * 2;
    let mut fallback_wy = start_wy + step_y * 4;
    for _ in 0..max_steps {
        fallback_wy += step_y;
        let rb_sample = pixel_body.sample_rigidbody_at_world_cell(fallback_wx, fallback_wy, body_pos, sin_a, cos_a);
        let outside_rb = rb_sample.is_none() || rb_sample.unwrap().is_air();
        if outside_rb {
            return (fallback_wx, fallback_wy);
        }
    }

    (curr_wx, curr_wy)
}

/// Rollback-ready raw `rapier2d` physics engine wrapped in a single `bones_ecs` resource.
///
/// Holds all Rapier simulation state without `bevy_rapier2d`, implementing `Clone` for
/// O(1) session rollback snapshots.
#[derive(HasSchema)]
#[schema(opaque)]
pub struct RapierPhysics {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub integration_parameters: IntegrationParameters,
    pub pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub soft_bodies: SoftBodySet,
    pub ccd_solver: CCDSolver,
    /// Deterministic mapping of active chunk coordinates to their Rapier static terrain collider.
    pub chunk_colliders: BTreeMap<(i32, i32), ColliderHandle>,
    /// Global gravity vector in world coordinates (`+Y` up).
    pub gravity: rapier2d::math::Vec2,
}

impl Clone for RapierPhysics {
    fn clone(&self) -> Self {
        Self {
            bodies: self.bodies.clone(),
            colliders: self.colliders.clone(),
            integration_parameters: self.integration_parameters,
            // `PhysicsPipeline` is stateless scratch memory in Rapier.
            pipeline: PhysicsPipeline::new(),
            island_manager: self.island_manager.clone(),
            broad_phase: self.broad_phase.clone(),
            narrow_phase: self.narrow_phase.clone(),
            impulse_joints: self.impulse_joints.clone(),
            multibody_joints: self.multibody_joints.clone(),
            soft_bodies: self.soft_bodies.clone(),
            ccd_solver: self.ccd_solver.clone(),
            chunk_colliders: self.chunk_colliders.clone(),
            gravity: self.gravity,
        }
    }
}

impl Default for RapierPhysics {
    fn default() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd_solver: CCDSolver::new(),
            chunk_colliders: BTreeMap::new(),
            gravity: rapier2d::math::Vec2::new(0.0, -600.0),
        }
    }
}

impl RapierPhysics {
    /// Removes a dynamic or kinematic [`RigidBody`] from the physics simulation along with its attached colliders.
    pub fn remove_body(
        &mut self,
        handle: RigidBodyHandle,
    ) -> Option<rapier2d::dynamics::RigidBody> {
        self.bodies.remove(
            handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            true,
        )
    }

    /// Removes a [`Collider`] from the physics simulation.
    pub fn remove_collider(
        &mut self,
        handle: ColliderHandle,
    ) -> Option<rapier2d::geometry::Collider> {
        self.colliders.remove(
            handle,
            &mut self.island_manager,
            &mut self.bodies,
            &mut self.soft_bodies,
            true,
        )
    }

    /// Clears and removes all static chunk terrain colliders from the collider set.
    pub fn clear_chunk_colliders(&mut self) {
        let handles: Vec<ColliderHandle> = self.chunk_colliders.values().copied().collect();
        for h in handles {
            self.remove_collider(h);
        }
        self.chunk_colliders.clear();
    }
}

/// Component storing a handle to a Rapier [`RigidBody`] alongside previous and current
/// transform state for tick-boundary render interpolation.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq, Default)]
pub struct PhysicsBodyHandle {
    pub raw_handle: [u32; 2],
    pub prev_pos: Vec2,
    pub pos: Vec2,
    pub prev_angle: f32,
    pub angle: f32,
}

impl PhysicsBodyHandle {
    /// Constructs a new [`PhysicsBodyHandle`].
    pub fn new(handle: RigidBodyHandle, pos: Vec2, angle: f32) -> Self {
        let (idx, generation) = handle.into_raw_parts();
        Self {
            raw_handle: [idx, generation],
            prev_pos: pos,
            pos,
            prev_angle: angle,
            angle,
        }
    }

    /// Reconstructs the [`RigidBodyHandle`] from its raw index and generation.
    #[inline]
    pub fn handle(&self) -> RigidBodyHandle {
        RigidBodyHandle::from_raw_parts(self.raw_handle[0], self.raw_handle[1])
    }
}

/// Component linking a [`KinematicActor`] entity to its Rapier rigid body and collider handles.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq, Default)]
pub struct ActorPhysicsHandle {
    pub body_raw: [u32; 2],
    pub collider_raw: [u32; 2],
}

impl ActorPhysicsHandle {
    /// Constructs a new [`ActorPhysicsHandle`].
    pub fn new(body: RigidBodyHandle, collider: ColliderHandle) -> Self {
        let (b_idx, b_gen) = body.into_raw_parts();
        let (c_idx, c_gen) = collider.into_raw_parts();
        Self {
            body_raw: [b_idx, b_gen],
            collider_raw: [c_idx, c_gen],
        }
    }

    /// Reconstructs the [`RigidBodyHandle`] from its raw parts.
    #[inline]
    pub fn body(&self) -> RigidBodyHandle {
        RigidBodyHandle::from_raw_parts(self.body_raw[0], self.body_raw[1])
    }

    /// Reconstructs the [`ColliderHandle`] from its raw parts.
    #[inline]
    pub fn collider(&self) -> ColliderHandle {
        ColliderHandle::from_raw_parts(self.collider_raw[0], self.collider_raw[1])
    }
}

/// Dynamic pixel-backed physical object composed of a 2D grid of [`Atom`]s.
#[derive(HasSchema, Clone, Debug, Default)]
#[schema(opaque)]
pub struct PixelRigidBody {
    pub width: u16,
    pub height: u16,
    pub atoms: std::sync::Arc<Vec<Atom>>,
    pub prev_pos: Vec2,
    pub pos: Vec2,
    pub prev_angle: f32,
    pub angle: f32,
    pub image_handle: Option<Handle<Image>>,
    pub render_cid: Option<Cid>,
    pub last_linvel: Vec2,
    pub prev_submerged_liquid_count: usize,
}

impl PixelRigidBody {
    /// Creates a new solid rectangular [`PixelRigidBody`].
    pub fn new_box(width: u16, height: u16, atom: Atom, pos: Vec2) -> Self {
        let count = (width as usize) * (height as usize);
        let atoms = std::sync::Arc::new(vec![atom; count]);
        Self {
            width,
            height,
            atoms,
            prev_pos: pos,
            pos,
            prev_angle: 0.0,
            angle: 0.0,
            image_handle: None,
            render_cid: None,
            last_linvel: Vec2::ZERO,
            prev_submerged_liquid_count: 0,
        }
    }

    /// Creates a styled wooden crate with darker edge borders and a diagonal cross brace.
    pub fn new_crate(width: u16, height: u16, pos: Vec2) -> Self {
        let w = width as usize;
        let h = height as usize;
        let mut atoms = Vec::with_capacity(w * h);

        let border_atom = Atom::with_state(12, AtomState::Solid, [101, 67, 33, 255]); // Dark wood
        let inner_atom = Atom::with_state(12, AtomState::Solid, [156, 102, 50, 255]); // Light wood
        let brace_atom = Atom::with_state(12, AtomState::Solid, [120, 80, 40, 255]); // Medium wood

        for y in 0..h {
            for x in 0..w {
                let is_border = x == 0 || x == w - 1 || y == 0 || y == h - 1;
                let is_cross = x == y || (x + y + 1) == w;
                if is_border {
                    atoms.push(border_atom);
                } else if is_cross {
                    atoms.push(brace_atom);
                } else {
                    atoms.push(inner_atom);
                }
            }
        }

        Self {
            width,
            height,
            atoms: std::sync::Arc::new(atoms),
            prev_pos: pos,
            pos,
            prev_angle: 0.0,
            angle: 0.0,
            image_handle: None,
            render_cid: None,
            last_linvel: Vec2::ZERO,
            prev_submerged_liquid_count: 0,
        }
    }

    /// Projects an integer world cell `(wx, wy)` backward into the rigidbody's local grid
    /// and returns the [`Atom`] at that position if it lies within the body bounds.
    #[inline]
    pub fn sample_rigidbody_at_world_cell(
        &self,
        wx: i32,
        wy: i32,
        center_pos: Vec2,
        sin_a: f32,
        cos_a: f32,
    ) -> Option<Atom> {
        let dx = (wx as f32 + 0.5) - center_pos.x;
        let dy = (wy as f32 + 0.5) - center_pos.y;
        let half_w = self.width as f32 * 0.5;
        let half_h = self.height as f32 * 0.5;
        let local_x = (dx * cos_a + dy * sin_a + half_w).floor() as i32;
        let local_y = (-dx * sin_a + dy * cos_a + half_h).floor() as i32;

        if local_x >= 0 && local_x < self.width as i32 && local_y >= 0 && local_y < self.height as i32 {
            let idx = (local_y as usize) * (self.width as usize) + (local_x as usize);
            self.atoms.get(idx).copied()
        } else {
            None
        }
    }

    /// Computes the total mass of the object by summing the material weights of all non-air atoms.
    pub fn total_mass(&self, materials: Option<&Materials>) -> f32 {
        let mut total: f32 = 0.0;
        for atom in self.atoms.iter() {
            if !atom.is_air() {
                let weight = materials
                    .and_then(|m| m.0.get(atom.material_id as usize))
                    .map(|m| m.weight)
                    .unwrap_or_else(|| match atom.material_id {
                        12 => 0.5,
                        8 => 2.5,
                        _ => 1.0,
                    })
                    .max(0.1);
                total += weight;
            }
        }
        total.max(1.0)
    }

    /// Generates a dynamic [`Image`] texture from the pixel atoms.
    pub fn to_image(&self) -> Image {
        let mut img = image::RgbaImage::new(self.width as u32, self.height as u32);
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let atom = self.atoms[y * self.width as usize + x];
                if !atom.is_air() {
                    img.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgba([
                            atom.color[0],
                            atom.color[1],
                            atom.color[2],
                            atom.color[3].max(255),
                        ]),
                    );
                } else {
                    img.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 0, 0]));
                }
            }
        }
        Image::Data(image::DynamicImage::ImageRgba8(img))
    }
}

/// Tracks cells in [`ChunkWorld`] where temporary [`AtomState::Object`] atoms were placed
/// for active dynamic [`PixelRigidBody`] entities before cellular automata simulation runs.
#[derive(HasSchema, Clone, Debug, Default)]
pub struct RigidbodySimulationStamp {
    pub stamped_atoms: Vec<(ChunkPos, Atom)>,
}

/// Builds compound cuboid shapes for a [`PixelRigidBody`] using 2D greedy rectangular meshing.
pub fn build_pixel_body_shapes(body: &PixelRigidBody) -> Vec<(rapier2d::math::Pose2, SharedShape)> {
    let w = body.width as usize;
    let h = body.height as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut visited = vec![false; w * h];
    let mut shapes = Vec::new();

    let half_w = body.width as f32 * 0.5;
    let half_h = body.height as f32 * 0.5;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || body.atoms[idx].is_air() {
                continue;
            }

            // 1. Scan horizontal run
            let mut x_end = x + 1;
            while x_end < w && !visited[y * w + x_end] && !body.atoms[y * w + x_end].is_air() {
                x_end += 1;
            }

            // 2. Scan vertical expansion
            let mut y_end = y + 1;
            'vert: while y_end < h {
                for rx in x..x_end {
                    let r_idx = y_end * w + rx;
                    if visited[r_idx] || body.atoms[r_idx].is_air() {
                        break 'vert;
                    }
                }
                y_end += 1;
            }

            // 3. Mark visited
            for vy in y..y_end {
                for vx in x..x_end {
                    visited[vy * w + vx] = true;
                }
            }

            // 4. Construct local cuboid
            let rect_w = (x_end - x) as f32;
            let rect_h = (y_end - y) as f32;
            let local_x = (x as f32 + x_end as f32) * 0.5 - half_w;
            let local_y = (y as f32 + y_end as f32) * 0.5 - half_h;

            shapes.push((
                rapier2d::math::Pose2::new(rapier2d::math::Vec2::new(local_x, local_y), 0.0),
                SharedShape::cuboid(rect_w * 0.5, rect_h * 0.5),
            ));
        }
    }

    shapes
}

/// Padded size for the binary chunk grid (`CHUNK_SIZE + 2 = 66`).
pub const PADDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;

/// Builds a despeckled `66 x 66` binary grid with a 1-cell false outer border.
///
/// Isolated single solid pixels (with 0 solid 4-neighbors) are despeckled to false so
/// stray pixels never generate tiny diamond colliders.
pub fn despeckle_and_pad_chunk(chunk: &Chunk) -> [bool; PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE] {
    let mut raw = [false; CHUNK_SIZE * CHUNK_SIZE];
    for (i, atom) in chunk.grid.atoms.iter().enumerate() {
        raw[i] = atom.is_solid();
    }

    let mut padded = [false; PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE];
    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let idx = ly * CHUNK_SIZE + lx;
            if !raw[idx] {
                continue;
            }
            let has_up = ly > 0 && raw[(ly - 1) * CHUNK_SIZE + lx];
            let has_down = ly + 1 < CHUNK_SIZE && raw[(ly + 1) * CHUNK_SIZE + lx];
            let has_left = lx > 0 && raw[ly * CHUNK_SIZE + (lx - 1)];
            let has_right = lx + 1 < CHUNK_SIZE && raw[ly * CHUNK_SIZE + (lx + 1)];

            if has_up || has_down || has_left || has_right {
                padded[(ly + 1) * PADDED_CHUNK_SIZE + (lx + 1)] = true;
            }
        }
    }

    padded
}

/// Quantizes half-integer coordinates to deterministic integer keys.
#[inline]
pub fn quantize_pt(pt: [f32; 2]) -> (i32, i32) {
    ((pt[0] * 2.0).round() as i32, (pt[1] * 2.0).round() as i32)
}

/// Extracts closed polygon loops from a 2D padded binary grid of arbitrary dimensions using Marching Squares.
///
/// Directed boundary segments are generated such that solid matter is consistently on the right
/// side of each directed edge. Saddle cases 5 and 10 are resolved deterministically without branching.
pub fn extract_marching_squares_loops_grid(
    padded: &[bool],
    padded_w: usize,
    padded_h: usize,
) -> Vec<Vec<[f32; 2]>> {
    let mut segments = BTreeMap::new();

    let grid_w = padded_w.saturating_sub(1);
    let grid_h = padded_h.saturating_sub(1);
    let max_x = padded_w.saturating_sub(2) as f32;
    let max_y = padded_h.saturating_sub(2) as f32;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let tl = padded[gy * padded_w + gx];
            let tr = padded[gy * padded_w + (gx + 1)];
            let br = padded[(gy + 1) * padded_w + (gx + 1)];
            let bl = padded[(gy + 1) * padded_w + gx];

            let mask = (tl as u8) | ((tr as u8) << 1) | ((br as u8) << 2) | ((bl as u8) << 3);
            if mask == 0 || mask == 15 {
                continue;
            }

            let top = [gx as f32 - 0.5, gy as f32 - 1.0];
            let right = [gx as f32, gy as f32 - 0.5];
            let bottom = [gx as f32 - 0.5, gy as f32];
            let left = [gx as f32 - 1.0, gy as f32 - 0.5];

            let mut add_seg = |p_start: [f32; 2], p_end: [f32; 2]| {
                let k_start = quantize_pt(p_start);
                let k_end = quantize_pt(p_end);
                if k_start != k_end {
                    segments.insert(k_start, (p_start, k_end, p_end));
                }
            };

            match mask {
                1 => add_seg(top, left),
                2 => add_seg(right, top),
                3 => add_seg(right, left),
                4 => add_seg(bottom, right),
                5 => {
                    add_seg(top, left);
                    add_seg(bottom, right);
                }
                6 => add_seg(bottom, top),
                7 => add_seg(bottom, left),
                8 => add_seg(left, bottom),
                9 => add_seg(top, bottom),
                10 => {
                    add_seg(right, top);
                    add_seg(left, bottom);
                }
                11 => add_seg(right, bottom),
                12 => add_seg(left, right),
                13 => add_seg(top, right),
                14 => add_seg(left, top),
                _ => {}
            }
        }
    }

    let mut raw_loops = Vec::new();
    while let Some((&start_key, &(p_start, next_key, _))) = segments.iter().next() {
        let mut loop_pts = vec![p_start];
        segments.remove(&start_key);

        let mut curr_key = next_key;
        let mut safety = 0;
        while curr_key != start_key && safety < 16384 {
            safety += 1;
            if let Some((pt, next, _)) = segments.remove(&curr_key) {
                loop_pts.push(pt);
                curr_key = next;
            } else {
                break;
            }
        }

        if loop_pts.len() >= 3 {
            // Clamp loop vertices to [0.0, max] and deduplicate consecutive identical vertices
            let mut clamped_loop: Vec<[f32; 2]> = Vec::with_capacity(loop_pts.len());
            for pt in loop_pts {
                let cx = pt[0].clamp(0.0, max_x);
                let cy = pt[1].clamp(0.0, max_y);
                if let Some(&last) = clamped_loop.last() {
                    if (last[0] - cx).abs() < 1e-4 && (last[1] - cy).abs() < 1e-4 {
                        continue;
                    }
                }
                clamped_loop.push([cx, cy]);
            }
            if clamped_loop.len() > 1 {
                let first = clamped_loop[0];
                let last = *clamped_loop.last().unwrap();
                if (last[0] - first[0]).abs() < 1e-4 && (last[1] - first[1]).abs() < 1e-4 {
                    clamped_loop.pop();
                }
            }
            if clamped_loop.len() >= 3 {
                raw_loops.push(clamped_loop);
            }
        }
    }

    raw_loops
}

/// Extracts closed polygon loops from a `66 x 66` padded binary grid using Marching Squares.
///
/// Directed boundary segments are generated such that solid terrain is consistently on the right
/// side of each directed edge. Saddle cases 5 and 10 are resolved deterministically without branching.
pub fn extract_marching_squares_loops(
    padded: &[bool; PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE],
) -> Vec<Vec<[f32; 2]>> {
    extract_marching_squares_loops_grid(padded, PADDED_CHUNK_SIZE, PADDED_CHUNK_SIZE)
}

/// Computes perpendicular distance from point `p` to line segment `ab`.
#[inline]
pub fn perpendicular_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-6 {
        let px = p[0] - a[0];
        let py = p[1] - a[1];
        return (px * px + py * py).sqrt();
    }
    let cross = (p[0] - a[0]) * dy - (p[1] - a[1]) * dx;
    cross.abs() / len_sq.sqrt()
}

/// Ramer-Douglas-Peucker (RDP) polyline simplification for open vertex chains.
pub fn rdp_simplify(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let a = points[0];
    let b = points[points.len() - 1];

    let mut max_dist: f32 = 0.0;
    let mut index: usize = 0;

    for i in 1..points.len() - 1 {
        let dist = perpendicular_distance(points[i], a, b);
        if dist > max_dist {
            max_dist = dist;
            index = i;
        }
    }

    if max_dist > epsilon {
        let mut left = rdp_simplify(&points[..=index], epsilon);
        let right = rdp_simplify(&points[index..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![a, b]
    }
}

/// Filters out degenerate loops.
///
/// If a thin 1-pixel feature simplifies down to a 2-vertex line segment, it is kept as long as its
/// length is >= 2.0 pixels (discarding tiny sub-pixel crumbs). Polygons with 3+ vertices require
/// a total perimeter >= 4.0 pixels.
pub fn filter_degenerate_loop(pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if pts.len() < 2 {
        return Vec::new();
    }
    if pts.len() == 2 {
        let dx = pts[1][0] - pts[0][0];
        let dy = pts[1][1] - pts[0][1];
        if (dx * dx + dy * dy).sqrt() < 2.0 {
            return Vec::new();
        }
        return pts;
    }

    let mut perimeter: f32 = 0.0;
    for i in 0..pts.len() {
        let next = (i + 1) % pts.len();
        let dx = pts[next][0] - pts[i][0];
        let dy = pts[next][1] - pts[i][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    if perimeter < 4.0 {
        return Vec::new();
    }

    pts
}

/// Simplifies a closed polygon loop using Ramer-Douglas-Peucker (RDP).
///
/// Splits the loop at its two furthest vertices into two open chains, runs RDP with `epsilon`,
/// and recombines them into a simplified closed loop.
pub fn simplify_closed_loop(loop_pts: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
    if loop_pts.len() < 3 {
        return Vec::new();
    }

    let mut max_dist_sq: f32 = 0.0;
    let mut split_idx: usize = 0;
    for i in 1..loop_pts.len() {
        let dx = loop_pts[i][0] - loop_pts[0][0];
        let dy = loop_pts[i][1] - loop_pts[0][1];
        let d_sq = dx * dx + dy * dy;
        if d_sq > max_dist_sq {
            max_dist_sq = d_sq;
            split_idx = i;
        }
    }

    if split_idx == 0 || split_idx == loop_pts.len() - 1 {
        let mut open = loop_pts.to_vec();
        open.push(loop_pts[0]);
        let mut simp = rdp_simplify(&open, epsilon);
        simp.pop();
        return filter_degenerate_loop(simp);
    }

    let chain1 = &loop_pts[0..=split_idx];
    let mut chain2 = loop_pts[split_idx..].to_vec();
    chain2.push(loop_pts[0]);

    let mut simp1 = rdp_simplify(chain1, epsilon);
    let simp2 = rdp_simplify(&chain2, epsilon);

    simp1.pop();
    simp1.extend(simp2);
    simp1.pop();

    filter_degenerate_loop(simp1)
}

/// Extracts and simplifies all closed terrain contour loops for a chunk in local `[0.0, 64.0]` coordinates.
pub fn build_chunk_terrain_loops(chunk: &Chunk) -> Vec<Vec<[f32; 2]>> {
    let padded = despeckle_and_pad_chunk(chunk);
    let raw_loops = extract_marching_squares_loops(&padded);
    let mut simplified_loops = Vec::with_capacity(raw_loops.len());

    for r_loop in raw_loops {
        let simplified = simplify_closed_loop(&r_loop, 1.4);
        if !simplified.is_empty() {
            simplified_loops.push(simplified);
        }
    }

    simplified_loops
}

/// Builds a single static [`SharedShape::polyline`] for chunk `(cx, cy)` using Marching Squares + RDP.
/// Returns `None` if the chunk has no solid collision geometry.
pub fn build_chunk_terrain_polyline(chunk: &Chunk, cx: i32, cy: i32) -> Option<SharedShape> {
    let loops = build_chunk_terrain_loops(chunk);
    if loops.is_empty() {
        return None;
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for poly_loop in loops {
        let base_idx = vertices.len() as u32;
        let n = poly_loop.len() as u32;

        for pt in poly_loop {
            let world_x = (cx * CHUNK_SIZE as i32) as f32 + pt[0];
            let world_y = -(cy * CHUNK_SIZE as i32) as f32 - pt[1];
            vertices.push(rapier2d::math::Vec2::new(world_x, world_y));
        }

        if n == 2 {
            indices.push([base_idx, base_idx + 1]);
        } else {
            for i in 0..n {
                let next_i = (i + 1) % n;
                indices.push([base_idx + i, base_idx + next_i]);
            }
        }
    }

    if indices.is_empty() {
        None
    } else {
        Some(SharedShape::polyline(vertices, Some(indices)))
    }
}

/// Builds a despeckled padded binary grid for a [`PixelRigidBody`] with a 1-cell false outer border.
pub fn despeckle_and_pad_pixel_body(body: &PixelRigidBody) -> (usize, usize, Vec<bool>) {
    let w = body.width as usize;
    let h = body.height as usize;
    let padded_w = w + 2;
    let padded_h = h + 2;
    let mut padded = vec![false; padded_w * padded_h];

    if w == 0 || h == 0 {
        return (padded_w, padded_h, padded);
    }

    let is_solid = |x: usize, y: usize| -> bool {
        let idx = y * w + x;
        if idx < body.atoms.len() {
            let atom = body.atoms[idx];
            !atom.is_air() && !atom.is_void()
        } else {
            false
        }
    };

    for y in 0..h {
        for x in 0..w {
            if !is_solid(x, y) {
                continue;
            }
            let has_up = y > 0 && is_solid(x, y - 1);
            let has_down = y + 1 < h && is_solid(x, y + 1);
            let has_left = x > 0 && is_solid(x - 1, y);
            let has_right = x + 1 < w && is_solid(x + 1, y);

            if has_up || has_down || has_left || has_right || (w <= 2 && h <= 2) {
                padded[(y + 1) * padded_w + (x + 1)] = true;
            }
        }
    }

    (padded_w, padded_h, padded)
}

/// Default Ramer-Douglas-Peucker simplification tolerance for dynamic pixel rigidbodies.
/// A lower epsilon preserves more fine pixel details, corners, and small notches.
pub const RIGIDBODY_RDP_EPSILON: f32 = 0.7;

/// Extracts and simplifies closed contour loops for a [`PixelRigidBody`] in local pixel coordinates.
pub fn build_pixel_body_loops(body: &PixelRigidBody) -> Vec<Vec<[f32; 2]>> {
    let (padded_w, padded_h, padded) = despeckle_and_pad_pixel_body(body);
    let raw_loops = extract_marching_squares_loops_grid(&padded, padded_w, padded_h);
    let mut simplified_loops = Vec::with_capacity(raw_loops.len());

    for r_loop in raw_loops {
        let simplified = simplify_closed_loop(&r_loop, RIGIDBODY_RDP_EPSILON);
        if !simplified.is_empty() {
            simplified_loops.push(simplified);
        }
    }

    simplified_loops
}

/// Builds a dynamic [`SharedShape`] for a [`PixelRigidBody`] using Marching Squares + RDP polyline simplification.
///
/// Loops are extracted, simplified via RDP, and decomposed into convex polygons via convex decomposition.
/// Falls back to convex hull or cuboid if decomposition yields no shapes.
pub fn build_pixel_body_rdp_shape(body: &PixelRigidBody) -> SharedShape {
    let loops = build_pixel_body_loops(body);
    let half_w = body.width as f32 * 0.5;
    let half_h = body.height as f32 * 0.5;

    if !loops.is_empty() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for poly_loop in &loops {
            let base_idx = vertices.len() as u32;
            let n = poly_loop.len() as u32;
            if n < 3 {
                continue;
            }

            for pt in poly_loop {
                let local_x = pt[0] - half_w;
                let local_y = pt[1] - half_h;
                vertices.push(rapier2d::math::Vec2::new(local_x, local_y));
            }

            for i in 0..n {
                let next_i = (i + 1) % n;
                indices.push([base_idx + i, base_idx + next_i]);
            }
        }

        if !vertices.is_empty() && !indices.is_empty() {
            let decomp = SharedShape::convex_decomposition(&vertices, &indices);
            if let Some(compound) = decomp.as_compound() {
                if !compound.shapes().is_empty() {
                    return decomp;
                }
            }
            if let Some(hull) = SharedShape::convex_hull(&vertices) {
                return hull;
            }
        }
    }

    // Fallback: cuboid of body dimensions
    SharedShape::cuboid(half_w.max(0.5), half_h.max(0.5))
}

/// Outer radius around dynamic rigidbodies within which static terrain colliders are maintained.
///
/// Chunks with no dynamic rigidbodies within this radius have their terrain colliders removed.
pub const TERRAIN_COLLIDER_RADIUS: f32 = 96.0;

/// Inner radius around awake dynamic rigidbodies within which terrain colliders are actively created/rebuilt.
pub const TERRAIN_AWAKE_RADIUS: f32 = 48.0;

/// Computes the shortest Euclidean distance from a world point to a chunk's bounding box.
#[inline]
pub fn distance_to_chunk(pos: Vec2, cx: i32, cy: i32) -> f32 {
    let min_x = (cx * CHUNK_SIZE as i32) as f32;
    let max_x = min_x + CHUNK_SIZE as f32;
    let max_y = -(cy * CHUNK_SIZE as i32) as f32; // world top
    let min_y = max_y - CHUNK_SIZE as f32; // world bottom

    let dx = (min_x - pos.x).max(0.0).max(pos.x - max_x);
    let dy = (min_y - pos.y).max(0.0).max(pos.y - max_y);
    (dx * dx + dy * dy).sqrt()
}

/// Synchronizes terrain colliders based on dynamic rigidbody positions.
///
/// Terrain colliders are created and maintained only within [`TERRAIN_COLLIDER_RADIUS`] of dynamic
/// rigidbodies. Active rebuilding of dirty chunks only occurs on the smaller [`TERRAIN_AWAKE_RADIUS`]
/// around awake bodies. If there are no rigidbodies close to a chunk, its collider is removed.
pub fn sync_awake_terrain_colliders(physics: &mut RapierPhysics, world: &mut ChunkWorld) {
    // 1. Collect all dynamic rigidbodies with their position and awake state
    let mut dynamic_bodies = Vec::new();
    for (_handle, body) in physics.bodies.iter() {
        if body.is_dynamic() {
            let pos = from_rapier_vec(body.translation());
            let is_awake = !body.is_sleeping();
            dynamic_bodies.push((pos, is_awake));
        }
    }

    // 2. If there are NO dynamic rigidbodies in the world, remove all terrain colliders
    if dynamic_bodies.is_empty() {
        if !physics.chunk_colliders.is_empty() {
            physics.clear_chunk_colliders();
        }
        return;
    }

    // 3. Find all chunks within TERRAIN_COLLIDER_RADIUS of ANY dynamic body,
    // and identify which chunks are within TERRAIN_AWAKE_RADIUS of an AWAKE body.
    let mut chunks_in_collider_radius = BTreeSet::new();
    let mut chunks_to_rebuild = BTreeSet::new();

    for &(pos, is_awake) in &dynamic_bodies {
        let min_x = pos.x - TERRAIN_COLLIDER_RADIUS;
        let max_x = pos.x + TERRAIN_COLLIDER_RADIUS;
        let min_y = pos.y - TERRAIN_COLLIDER_RADIUS;
        let max_y = pos.y + TERRAIN_COLLIDER_RADIUS;

        let cx_min = (min_x / CHUNK_SIZE as f32).floor() as i32;
        let cx_max = (max_x / CHUNK_SIZE as f32).floor() as i32;
        let cy_min = ((-max_y) / CHUNK_SIZE as f32).floor() as i32;
        let cy_max = ((-min_y) / CHUNK_SIZE as f32).floor() as i32;

        for cy in cy_min..=cy_max {
            for cx in cx_min..=cx_max {
                let dist = distance_to_chunk(pos, cx, cy);
                if dist <= TERRAIN_COLLIDER_RADIUS {
                    chunks_in_collider_radius.insert((cx, cy));
                    // Active creation/rebuilding occurs within the smaller awake radius
                    if is_awake && dist <= TERRAIN_AWAKE_RADIUS {
                        chunks_to_rebuild.insert((cx, cy));
                    }
                }
            }
        }
    }

    // 4. Remove any existing chunk colliders that no longer have a rigidbody within TERRAIN_COLLIDER_RADIUS
    let active_chunk_keys: Vec<(i32, i32)> = physics.chunk_colliders.keys().copied().collect();
    for (cx, cy) in active_chunk_keys {
        if !chunks_in_collider_radius.contains(&(cx, cy)) {
            if let Some(handle) = physics.chunk_colliders.remove(&(cx, cy)) {
                physics.remove_collider(handle);
            }
        }
    }

    // 5. Build/rebuild chunk colliders for chunks within the awake radius that are dirty or missing
    for (cx, cy) in chunks_to_rebuild {
        if let Some(chunk) = world.chunks.get_mut(&ivec2(cx, cy)) {
            if chunk.collider_dirty || !physics.chunk_colliders.contains_key(&(cx, cy)) {
                if let Some(old_handle) = physics.chunk_colliders.remove(&(cx, cy)) {
                    physics.remove_collider(old_handle);
                }

                if let Some(shape) = build_chunk_terrain_polyline(chunk, cx, cy) {
                    let collider = ColliderBuilder::new(shape)
                        .collision_groups(CollisionGroups::new(GROUP_TERRAIN, GROUP_DYNAMIC_BODY))
                        .friction(0.6)
                        .restitution(0.1)
                        .build();
                    let new_handle = physics.colliders.insert(collider);
                    physics.chunk_colliders.insert((cx, cy), new_handle);
                }

                chunk.collider_dirty = false;
            }
        }
    }
}

/// Spawns a new dynamic [`PixelRigidBody`] into components and Rapier state.
pub fn spawn_pixel_rigidbody_with_components(
    ent: Entity,
    physics: &mut RapierPhysics,
    materials: Option<&Materials>,
    body: PixelRigidBody,
    handles: &mut CompMut<PhysicsBodyHandle>,
    pixel_bodies: &mut CompMut<PixelRigidBody>,
    transforms: &mut CompMut<Transform>,
) {
    let shape = build_pixel_body_rdp_shape(&body);
    let total_mass = body.total_mass(materials);

    let rb = RigidBodyBuilder::dynamic()
        .translation(to_rapier_vec(body.pos))
        .rotation(body.angle)
        .additional_mass(total_mass)
        .linear_damping(0.1)
        .angular_damping(0.5)
        .ccd_enabled(true)
        .build();

    let rb_handle = physics.bodies.insert(rb);

    let collider = ColliderBuilder::new(shape)
        .collision_groups(CollisionGroups::new(
            GROUP_DYNAMIC_BODY,
            GROUP_TERRAIN | GROUP_DYNAMIC_BODY | GROUP_ACTOR,
        ))
        .mass(total_mass)
        .friction(0.6)
        .restitution(0.2)
        .build();

    let col_handle = {
        let RapierPhysics {
            ref mut colliders,
            ref mut bodies,
            ..
        } = *physics;
        colliders.insert_with_parent(collider, rb_handle, bodies)
    };

    let user_data = entity_to_user_data(ent);
    if let Some(body_mut) = physics.bodies.get_mut(rb_handle) {
        body_mut.user_data = user_data;
    }
    if let Some(col_mut) = physics.colliders.get_mut(col_handle) {
        col_mut.user_data = user_data;
    }

    let handle_comp = PhysicsBodyHandle::new(rb_handle, body.pos, body.angle);

    handles.insert(ent, handle_comp);
    pixel_bodies.insert(ent, body);

    let mut transform = Transform::default();
    transform.translation.x = handle_comp.pos.x;
    transform.translation.y = handle_comp.pos.y;
    transform.translation.z = 1.0;
    transform.rotation = Quat::from_rotation_z(handle_comp.angle);
    transforms.insert(ent, transform);
}

/// Spawns a new dynamic [`PixelRigidBody`] in `world` and initializes its Rapier rigid body and collider.
pub fn spawn_pixel_rigidbody(
    world: &mut World,
    materials: Option<&Materials>,
    body: PixelRigidBody,
) -> Entity {
    let ent = world.resource_mut::<Entities>().create();
    let mut physics = world.resource_mut::<RapierPhysics>();
    let mut handles = world.component_mut::<PhysicsBodyHandle>();
    let mut pixel_bodies = world.component_mut::<PixelRigidBody>();
    let mut transforms = world.component_mut::<Transform>();
    spawn_pixel_rigidbody_with_components(
        ent,
        &mut physics,
        materials,
        body,
        &mut handles,
        &mut pixel_bodies,
        &mut transforms,
    );
    ent
}

/// Steps [`RapierPhysics`] forward deterministically, updating all dynamic bodies and
/// performing two-way coupling with [`ChunkWorld`], [`ParticlePool`], and [`KinematicActor`](crate::actor::KinematicActor).
pub fn step_rapier_physics_system(
    mut physics: ResMut<RapierPhysics>,
    mut chunk_world: ResMut<ChunkWorld>,
    materials: Option<Res<Materials>>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
    mut dirty_rects: Option<ResMut<DirtyRects>>,
    entities: Res<Entities>,
    mut handles: CompMut<PhysicsBodyHandle>,
    mut pixel_bodies: CompMut<PixelRigidBody>,
    mut actors: CompMut<crate::actor::KinematicActor>,
    mut actor_handles: CompMut<ActorPhysicsHandle>,
    mut rb_obstacles: Option<ResMut<ParticleRigidBodyObstacles>>,
) {
    // 1. Sync terrain colliders for all awake dynamic bodies
    sync_awake_terrain_colliders(&mut physics, &mut chunk_world);

    let mats = materials.as_deref();
    let tick = chunk_world.tick;
    let dt = physics.integration_parameters.dt;
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 60.0 };

    // Step A (Pre-Rapier Sync): Ensure every KinematicActor has a Rapier body and collider,
    // and sync its position and desired velocity into Rapier.
    let mut stale_actor_handles = Vec::new();
    for (ent, handle) in entities.iter_with(&actor_handles) {
        if !entities.is_alive(ent) || !actors.contains(ent) {
            stale_actor_handles.push((ent, *handle));
        }
    }
    for (ent, handle) in stale_actor_handles {
        physics.remove_body(handle.body());
        actor_handles.remove(ent);
    }

    let mut ordered_actor_entities: Vec<Entity> =
        entities.iter_with(&actors).map(|(ent, _)| ent).collect();
    ordered_actor_entities.sort_by_key(|e| (e.index(), e.generation()));

    for ent in &ordered_actor_entities {
        if let Some(actor) = actors.get(*ent) {
            let needs_creation = match actor_handles.get(*ent) {
                Some(h) => {
                    !physics.bodies.contains(h.body()) || !physics.colliders.contains(h.collider())
                }
                None => true,
            };

            let handle = if needs_creation {
                let rb = RigidBodyBuilder::kinematic_velocity_based()
                    .translation(to_rapier_vec(actor.pos))
                    .build();
                let rb_h = physics.bodies.insert(rb);
                let collider = ColliderBuilder::cuboid(actor.half_size.x, actor.half_size.y)
                    .collision_groups(CollisionGroups::new(GROUP_ACTOR, GROUP_DYNAMIC_BODY))
                    .friction(0.6)
                    .build();
                let col_h = {
                    let RapierPhysics {
                        ref mut colliders,
                        ref mut bodies,
                        ..
                    } = *physics;
                    colliders.insert_with_parent(collider, rb_h, bodies)
                };
                let udata = entity_to_user_data(*ent);
                if let Some(b) = physics.bodies.get_mut(rb_h) {
                    b.user_data = udata;
                }
                if let Some(c) = physics.colliders.get_mut(col_h) {
                    c.user_data = udata;
                }
                let h = ActorPhysicsHandle::new(rb_h, col_h);
                actor_handles.insert(*ent, h);
                h
            } else {
                *actor_handles.get(*ent).unwrap()
            };

            if let Some(rb) = physics.bodies.get_mut(handle.body()) {
                rb.set_translation(to_rapier_vec(actor.pos), true);
                rb.set_linvel(to_rapier_vec(actor.vel * inv_dt), true);
            }
        }
    }

    // 2. Pre-step unified PixelRigidBody <-> Grid Overlap Pass (Buoyancy + Sand Impediment & Displacement)
    let g = physics.gravity.y.abs();
    let up_sign = if physics.gravity.y != 0.0 {
        -physics.gravity.y.signum()
    } else {
        1.0
    };
    let mut ordered_entities: Vec<Entity> =
        entities.iter_with(&handles).map(|(ent, _)| ent).collect();
    ordered_entities.sort_by_key(|e| (e.index(), e.generation()));

    for (sample_idx, ent) in ordered_entities.iter().enumerate() {
        if let Some(handle) = handles.get(*ent)
            && let Some(pixel_body) = pixel_bodies.get_mut(*ent)
            && let Some(rb) = physics.bodies.get_mut(handle.handle())
        {
            if !rb.is_dynamic() {
                continue;
            }
            if rb.is_sleeping() && rb.gravity_scale() >= 0.99 && pixel_body.prev_submerged_liquid_count == 0 {
                continue;
            }

            let body_pos = from_rapier_vec(rb.translation());
            let body_angle = rb.rotation().angle();
            let linvel = from_rapier_vec(rb.linvel());
            let angvel = rb.angvel();
            let (sin_a, cos_a) = body_angle.sin_cos();

            let half_w = pixel_body.width as f32 * 0.5;
            let half_h = pixel_body.height as f32 * 0.5;
            let body_mass = rb.mass().max(0.1);
            let total_pixels = pixel_body.atoms.iter().filter(|a| !a.is_air()).count().max(1);
            let mass_per_pixel = body_mass / (total_pixels as f32);

            let mut submerged_liquid_count: usize = 0;
            let mut real_liquid_splashes = 0;
            let mut vfx_liquid_splashes = 0;
            let mut powder_impediment: f32 = 0.0;
            let mut particles_thrown_this_frame = 0;

            let corners = [
                Vec2::new(-half_w, -half_h),
                Vec2::new(half_w, -half_h),
                Vec2::new(half_w, half_h),
                Vec2::new(-half_w, half_h),
            ];

            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for c in corners {
                let wx = body_pos.x + c.x * cos_a - c.y * sin_a;
                let wy = body_pos.y + c.x * sin_a + c.y * cos_a;
                min_x = min_x.min(wx);
                max_x = max_x.max(wx);
                min_y = min_y.min(wy);
                max_y = max_y.max(wy);
            }

            let min_wx = min_x.floor() as i32;
            let max_wx = max_x.ceil() as i32;
            let min_wy = min_y.floor() as i32;
            let max_wy = max_y.ceil() as i32;

            const SAND_DISPLACE_SPEED: f32 = 1.5;

            for wy in min_wy..=max_wy {
                for wx in min_wx..=max_wx {
                    let Some(rb_atom) = pixel_body.sample_rigidbody_at_world_cell(wx, wy, body_pos, sin_a, cos_a) else {
                        continue;
                    };
                    if rb_atom.is_air() {
                        continue;
                    }

                    let chunk_pos = global_to_chunk(world_cell_to_global_atom(wx, wy));
                    let Some(world_atom) = chunk_world.get_atom(&chunk_pos).copied() else {
                        continue;
                    };
                    if world_atom.is_air() {
                        continue;
                    }

                    let cell_world = Vec2::new(wx as f32 + 0.5, wy as f32 + 0.5);
                    let r = cell_world - body_pos;
                    let point_vel = linvel + Vec2::new(-angvel * r.y, angvel * r.x);
                    let speed = point_vel.length() * dt;

                    let rb_weight = mats
                        .and_then(|m| m.0.get(rb_atom.material_id as usize))
                        .map(|m| m.weight)
                        .unwrap_or_else(|| match rb_atom.material_id {
                            12 => 0.5,
                            8 => 2.5,
                            _ => 1.0,
                        })
                        .max(0.05);

                    let world_weight = mats
                        .and_then(|m| m.0.get(world_atom.material_id as usize))
                        .map(|m| m.weight)
                        .unwrap_or_else(|| match world_atom.material_id {
                            3 => 1.0,
                            5 => 2.2,
                            9 => 1.1,
                            2 => 1.5,
                            4 => 2.0,
                            _ => 1.0,
                        })
                        .max(0.05);

                    if world_atom.is_liquid() {
                        // Case A: Liquid (Noita Archimedes Buoyancy at exact pixel)
                        submerged_liquid_count += 1;
                        let density_ratio = (world_weight / rb_weight) * BUOYANCY_MULTIPLIER;
                        let pixel_buoyancy_impulse = up_sign * (mass_per_pixel * g * dt) * density_ratio;
                        rb.apply_impulse_at_point(
                            to_rapier_vec(Vec2::new(0.0, pixel_buoyancy_impulse)),
                            to_rapier_vec(cell_world),
                            true,
                        );

                        // Detect Surface / Waterline Pixels (is_near_surface):
                        let up_step = up_sign as i32;
                        let cp_above1 = global_to_chunk(world_cell_to_global_atom(wx, wy + up_step));
                        let cp_above2 = global_to_chunk(world_cell_to_global_atom(wx, wy + 2 * up_step));
                        let is_near_surface = match chunk_world.get_atom(&cp_above1) {
                            Some(a) => a.is_air() || a.is_void(),
                            None => true,
                        } || match chunk_world.get_atom(&cp_above2) {
                            Some(a) => a.is_air() || a.is_void(),
                            None => true,
                        };

                        if is_near_surface && speed > 1.4 {
                            let mut rng = SimpleRng::new(
                                tick,
                                (sample_idx * 1000 + (wx.abs() as usize * 31 + wy.abs() as usize)) as u32,
                            );
                            let dx = (wx as f32 + 0.5) - body_pos.x;
                            let side_sign = if dx.abs() > 0.5 {
                                dx.signum()
                            } else if rng.bool() {
                                1.0
                            } else {
                                -1.0
                            };
                            let speed_scale = speed.clamp(2.0, 10.0);
                            let vx = side_sign * rng.f32_range(0.35, 0.85) * speed_scale + point_vel.x * 0.25;
                            let vy = up_sign * rng.f32_range(0.45, 0.95) * speed_scale;
                            let splash_sim_vel = [vx, -vy];

                            // Real Matter Splash (capped at 6 per body per tick)
                            if real_liquid_splashes < 6 && rng.f32() < 0.35 {
                                let (exit_wx, exit_wy) = find_hull_exit_pos(
                                    pixel_body,
                                    body_pos,
                                    sin_a,
                                    cos_a,
                                    &chunk_world,
                                    wx,
                                    wy,
                                    side_sign,
                                    up_sign,
                                );
                                let orig_gx = wx;
                                let orig_gy = -wy - 1;
                                let spawn_gx = exit_wx;
                                let spawn_gy = -exit_wy - 1;

                                if let Some(ref mut pool) = particle_pool {
                                    if pool.eject_atom_at_pos_dirty(
                                        &mut chunk_world,
                                        dirty_rects.as_deref_mut(),
                                        orig_gx,
                                        orig_gy,
                                        spawn_gx,
                                        spawn_gy,
                                        splash_sim_vel,
                                    ) {
                                        real_liquid_splashes += 1;
                                    }
                                }
                            }

                            // VFX Mist / Foam / Embers (capped at 10 per body per tick)
                            if vfx_liquid_splashes < 10 {
                                if let Some(ref mut pool) = particle_pool {
                                    let vfx_gx = wx;
                                    let vfx_gy = -(wy + up_step) - 1;
                                    let vfx_pos = [vfx_gx as f32 + 0.5, vfx_gy as f32 + 0.5];

                                    let is_emissive = world_atom.color[3] >= 1 && world_atom.color[3] <= 127;
                                    let (vfx_color, lifetime, bounces) = if is_emissive {
                                        (world_atom.color, rng.u8_range(14, 26), true)
                                    } else {
                                        let r = ((world_atom.color[0] as f32) * 0.4 + 240.0 * 0.6) as u8;
                                        let g = ((world_atom.color[1] as f32) * 0.4 + 248.0 * 0.6) as u8;
                                        let b = ((world_atom.color[2] as f32) * 0.4 + 255.0 * 0.6) as u8;
                                        ([r, g, b, 190], rng.u8_range(10, 18), false)
                                    };

                                    pool.spawn_fx(vfx_pos, splash_sim_vel, vfx_color, lifetime, bounces);
                                    vfx_liquid_splashes += 1;
                                }
                            }
                        }
                    } else if world_atom.is_powder() {
                        // Case B: Powder (Velocity Displacement vs. Impediment Stopping)
                        let min_bottom_pixels = (pixel_body.width as f32 * 0.8).max(4.0);
                        let per_pixel_impediment = (body_mass / min_bottom_pixels) * (world_weight / rb_weight.max(0.1));

                        if speed > SAND_DISPLACE_SPEED {
                            // High Speed: Plow & Throw Sand outside the rigidbody hull
                            if particles_thrown_this_frame < 4 {
                                particles_thrown_this_frame += 1;
                                let mut rng = SimpleRng::new(
                                    tick,
                                    (sample_idx * 50 + (wx.abs() as usize * 13 + wy.abs() as usize)) as u32,
                                );
                                let dx = (wx as f32 + 0.5) - body_pos.x;
                                let side_sign = if dx.abs() > 0.5 {
                                    dx.signum()
                                } else if rng.bool() {
                                    1.0
                                } else {
                                    -1.0
                                };
                                let speed_scale = speed.clamp(2.0, 10.0);
                                let vx = side_sign * rng.f32_range(0.35, 0.85) * speed_scale + point_vel.x * 0.25;
                                let vy = up_sign * rng.f32_range(0.45, 0.95) * speed_scale;
                                let throw_sim_vel = [vx, -vy];

                                let (exit_wx, exit_wy) = find_hull_exit_pos(
                                    pixel_body,
                                    body_pos,
                                    sin_a,
                                    cos_a,
                                    &chunk_world,
                                    wx,
                                    wy,
                                    side_sign,
                                    up_sign,
                                );

                                let orig_gx = wx;
                                let orig_gy = -wy - 1;
                                let spawn_gx = exit_wx;
                                let spawn_gy = -exit_wy - 1;

                                if let Some(ref mut pool) = particle_pool {
                                    pool.eject_atom_at_pos_dirty(
                                        &mut chunk_world,
                                        dirty_rects.as_deref_mut(),
                                        orig_gx,
                                        orig_gy,
                                        spawn_gx,
                                        spawn_gy,
                                        throw_sim_vel,
                                    );
                                    pool.spawn_fx(
                                        [spawn_gx as f32 + 0.5, spawn_gy as f32 + 0.5],
                                        [throw_sim_vel[0] * 1.1, throw_sim_vel[1] * 1.1],
                                        crate::actor::atom_color(&world_atom),
                                        18,
                                        true,
                                    );
                                }

                                // Wake adjacent powder neighbors in ChunkWorld
                                for n_wy in [wy - 1, wy + 1] {
                                    let n_cp = global_to_chunk(world_cell_to_global_atom(wx, n_wy));
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
                            }

                            // Strong impact drag slows down plowing through sand
                            let k_impact = (body_mass / (pixel_body.width as f32 * 2.0)).min(body_mass * 0.5);
                            let drag_impulse = -point_vel * (world_weight * k_impact * dt * 10.0);
                            rb.apply_impulse_at_point(to_rapier_vec(drag_impulse), to_rapier_vec(cell_world), true);
                        } else {
                            // Low Speed: Rest on Sand via Impediment
                            powder_impediment += per_pixel_impediment;

                            if point_vel.y < 0.0 && !world_atom.is_free_falling() {
                                let support_y = (-point_vel.y * (body_mass / min_bottom_pixels) * 1.2).min(body_mass * 15.0 * dt);
                                let support_impulse = Vec2::new(0.0, support_y);
                                rb.apply_impulse_at_point(to_rapier_vec(support_impulse), to_rapier_vec(cell_world), true);
                            }
                        }
                    }
                }
            }

            pixel_body.prev_submerged_liquid_count = submerged_liquid_count;

            // 3. Apply fluid damping and powder impediment
            let submerged_fraction = (submerged_liquid_count as f32 / total_pixels as f32).clamp(0.0, 1.0);
            if submerged_fraction > 0.0 {
                let cur_linvel = rb.linvel();
                let cur_angvel = rb.angvel();
                let lin_damp = 1.0 - (0.08 * submerged_fraction);
                let ang_damp = 1.0 - (0.12 * submerged_fraction);
                rb.set_linvel(
                    rapier2d::math::Vec2::new(cur_linvel.x * lin_damp, cur_linvel.y * lin_damp),
                    false,
                );
                rb.set_angvel(cur_angvel * ang_damp, false);
            }

            let norm_impediment = (powder_impediment / body_mass).clamp(0.0, 1.0);
            let current_gravity_scale = rb.gravity_scale();

            if submerged_liquid_count > 0 && powder_impediment < 0.01 {
                if current_gravity_scale < 0.99 {
                    rb.set_gravity_scale(1.0, true);
                    rb.wake_up(true);
                }
            } else if norm_impediment > 0.01 {
                rb.set_gravity_scale(1.0 - norm_impediment, false);
            } else if current_gravity_scale < 0.99 {
                // If the sand underneath a resting rigidbody is later dug away,
                // restore rb.set_gravity_scale(1.0, true) and wake the rigidbody so it falls!
                rb.set_gravity_scale(1.0, true);
                rb.wake_up(true);
            }

            // Powder damping
            if norm_impediment > 0.01 {
                let mut powder_damp = 1.0;
                if norm_impediment > 0.85 {
                    powder_damp = 0.6;
                } else if norm_impediment > 0.2 {
                    powder_damp = 1.0 - (norm_impediment - 0.2) * 0.45;
                }
                let cur_linvel = rb.linvel();
                let cur_angvel = rb.angvel();
                rb.set_linvel(
                    rapier2d::math::Vec2::new(cur_linvel.x * powder_damp, cur_linvel.y * powder_damp),
                    false,
                );
                rb.set_angvel(cur_angvel * powder_damp, false);

                if norm_impediment > 0.85 {
                    let cur_linvel = rb.linvel();
                    if cur_linvel.length_squared() < 4.0 {
                        rb.set_linvel(rapier2d::math::Vec2::ZERO, false);
                    }
                    if rb.angvel().abs() < 0.05 {
                        rb.set_angvel(0.0, false);
                    }
                }
            }
        }
    }

    // 3. Step Rapier Physics Pipeline using disjoint field destructuring
    {
        let RapierPhysics {
            ref mut pipeline,
            gravity,
            ref integration_parameters,
            ref mut island_manager,
            ref mut broad_phase,
            ref mut narrow_phase,
            ref mut bodies,
            ref mut colliders,
            ref mut impulse_joints,
            ref mut multibody_joints,
            ref mut soft_bodies,
            ref mut ccd_solver,
            ..
        } = *physics;

        pipeline.step(
            gravity,
            integration_parameters,
            island_manager,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            soft_bodies,
            ccd_solver,
            &(),
            &(),
        );
    }

    // Step B (Two-Way Coupling Resolution):
    // Evaluate contact manifolds between GROUP_ACTOR and GROUP_DYNAMIC_BODY in deterministic handle order.
    struct ActorRbContact {
        actor_ent: Entity,
        actor_body_h: RigidBodyHandle,
        rb_h: RigidBodyHandle,
        normal: Vec2,
        contact_pt: Vec2,
        penetration: f32,
        contact_count: usize,
    }

    let mut actor_contacts = Vec::new();

    for pair in physics.narrow_phase.contact_pairs() {
        let Some(col1) = physics.colliders.get(pair.collider1) else {
            continue;
        };
        let Some(col2) = physics.colliders.get(pair.collider2) else {
            continue;
        };

        let is_actor_1 = col1.collision_groups().memberships.contains(GROUP_ACTOR);
        let is_rb_2 = col2.collision_groups().memberships.contains(GROUP_DYNAMIC_BODY);
        let is_actor_2 = col2.collision_groups().memberships.contains(GROUP_ACTOR);
        let is_rb_1 = col1.collision_groups().memberships.contains(GROUP_DYNAMIC_BODY);

        let (actor_col, rb_col, normal_sign) = if is_actor_1 && is_rb_2 {
            (col1, col2, 1.0)
        } else if is_actor_2 && is_rb_1 {
            (col2, col1, -1.0)
        } else {
            continue;
        };

        let Some(actor_body_h) = actor_col.parent() else {
            continue;
        };
        let Some(rb_h) = rb_col.parent() else {
            continue;
        };
        let actor_ent = user_data_to_entity(actor_col.user_data);

        let Some(rb_body) = physics.bodies.get(rb_h) else {
            continue;
        };
        if !rb_body.is_dynamic() {
            continue;
        }

        for manifold in pair.manifolds() {
            let normal = from_rapier_vec(manifold.data.normal) * normal_sign;
            let contact_count = manifold.data.solver_contacts.len().max(1);
            for contact in &manifold.data.solver_contacts {
                let penetration = if contact.dist < 0.0 { -contact.dist } else { 0.0 };
                let rb_anchor = if normal_sign > 0.0 {
                    contact.anchor2
                } else {
                    contact.anchor1
                };
                let contact_pt_rapier = rb_body.position() * rb_anchor;
                let contact_pt = from_rapier_vec(contact_pt_rapier);

                actor_contacts.push(ActorRbContact {
                    actor_ent,
                    actor_body_h,
                    rb_h,
                    normal,
                    contact_pt,
                    penetration,
                    contact_count,
                });
            }
        }
    }

    actor_contacts.sort_by_key(|c| {
        (
            c.actor_ent.index(),
            c.actor_ent.generation(),
            c.rb_h.into_raw_parts(),
        )
    });

    for contact in actor_contacts {
        let Some(actor) = actors.get_mut(contact.actor_ent) else {
            continue;
        };
        let Some(v_actor) = physics
            .bodies
            .get(contact.actor_body_h)
            .map(|b| from_rapier_vec(b.linvel()))
        else {
            continue;
        };
        let Some(rb) = physics.bodies.get_mut(contact.rb_h) else {
            continue;
        };

        let mass_a = actor.weight.max(1.0);
        let mass_b = rb.mass().max(0.1);
        let total_mass = mass_a + mass_b;
        let actor_share = mass_b / total_mass;
        let rb_share = mass_a / total_mass;
        let m_eff = (mass_a * mass_b) / total_mass;
        let scale = 1.0 / (contact.contact_count as f32);

        let v_rb = from_rapier_vec(rb.velocity_at_point(to_rapier_vec(contact.contact_pt)));
        let v_rel = v_actor - v_rb;
        let v_closing = v_rel.dot(contact.normal);

        let j_vel = v_closing.max(0.0) * m_eff * scale;
        let j_pen = (contact.penetration * inv_dt * 0.5 * scale) * m_eff;
        let j = contact.normal * (j_vel + j_pen);

        // 1. Actor Pushes & Tilts Rigidbody at exact contact point
        let weight_impulse = if contact.normal.y < -0.5 {
            contact.normal * (mass_a * 10.0 * dt * scale)
        } else {
            Vec2::ZERO
        };
        let rb_impulse = j + weight_impulse;
        rb.apply_impulse_at_point(
            to_rapier_vec(rb_impulse),
            to_rapier_vec(contact.contact_pt),
            true,
        );
        rb.wake_up(true);

        // Displace the dynamic body forward by its share of penetration
        if contact.penetration > 0.0 {
            let rb_disp = contact.normal * (contact.penetration * rb_share * scale);
            let new_rb_trans = rb.translation() + to_rapier_vec(rb_disp);
            rb.set_translation(new_rb_trans, true);
        }

        // 2. Rigidbody Pushes & Carries Actor
        let vel_correction = -contact.normal * (v_closing.max(0.0) * dt * actor_share * scale);
        actor.vel += vel_correction;

        let disp_correction = -contact.normal * (contact.penetration * actor_share * scale);
        crate::actor::sweep_actor_displacement(&chunk_world, actor, disp_correction);
    }

    // 4. Post-step sync back to components & spawn impact particles
    for ent in &ordered_entities {
        if let Some(handle) = handles.get_mut(*ent)
            && let Some(pixel_body) = pixel_bodies.get_mut(*ent)
            && let Some(rb) = physics.bodies.get(handle.handle())
        {
            let new_pos = from_rapier_vec(rb.translation());
            let new_angle = rb.rotation().angle();
            let new_linvel = from_rapier_vec(rb.linvel());

            handle.prev_pos = handle.pos;
            handle.pos = new_pos;
            handle.prev_angle = handle.angle;
            handle.angle = new_angle;

            pixel_body.prev_pos = handle.prev_pos;
            pixel_body.pos = handle.pos;
            pixel_body.prev_angle = handle.prev_angle;
            pixel_body.angle = handle.angle;

            // Check impact velocity drop for rock chips / dust puffs
            let vel_drop = pixel_body.last_linvel.length() - new_linvel.length();
            if vel_drop > 40.0 {
                if let Some(ref mut pool) = particle_pool {
                    let chip_count = ((vel_drop / 30.0).round() as usize).clamp(2, 8);
                    let color = [160, 160, 160, 255];
                    for i in 0..chip_count {
                        let angle = (i as f32 * std::f32::consts::TAU / chip_count as f32)
                            + (deterministic_noise(tick, i as u32) - 0.5) * 0.4;
                        let speed = (vel_drop * 0.05).clamp(1.5, 6.0) + 1.0;
                        let vel = [angle.cos() * speed, -angle.sin().abs() * speed];
                        pool.spawn_fx([new_pos.x, -new_pos.y], vel, color, 18, true);
                    }
                }
            }
            pixel_body.last_linvel = new_linvel;
        }
    }

    // 5. Populate ParticleRigidBodyObstacles so simulate_particles_system has safety net
    if let Some(ref mut obstacles_res) = rb_obstacles {
        obstacles_res.obstacles.clear();
        for ent in &ordered_entities {
            if let Some(handle) = handles.get(*ent)
                && let Some(pixel_body) = pixel_bodies.get(*ent)
                && let Some(rb) = physics.bodies.get(handle.handle())
            {
                if !rb.is_dynamic() {
                    continue;
                }
                let body_pos = from_rapier_vec(rb.translation());
                let body_angle = rb.rotation().angle();
                let (sin_a, cos_a) = body_angle.sin_cos();
                let half_w = pixel_body.width as f32 * 0.5;
                let half_h = pixel_body.height as f32 * 0.5;

                let corners = [
                    Vec2::new(-half_w, -half_h),
                    Vec2::new(half_w, -half_h),
                    Vec2::new(half_w, half_h),
                    Vec2::new(-half_w, half_h),
                ];
                let mut min_x = f32::MAX;
                let mut max_x = f32::MIN;
                let mut min_y = f32::MAX;
                let mut max_y = f32::MIN;
                for c in corners {
                    let wx = body_pos.x + c.x * cos_a - c.y * sin_a;
                    let wy = body_pos.y + c.x * sin_a + c.y * cos_a;
                    min_x = min_x.min(wx);
                    max_x = max_x.max(wx);
                    min_y = min_y.min(wy);
                    max_y = max_y.max(wy);
                }

                obstacles_res.obstacles.push(ParticleRigidBodyObstacle {
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    pos: [body_pos.x, body_pos.y],
                    sin_a,
                    cos_a,
                    width: pixel_body.width,
                    height: pixel_body.height,
                    atoms: pixel_body.atoms.clone(),
                });
            }
        }
    }
}

/// Actor sweep obstacle information representing an active dynamic body's bounding box.
#[derive(Clone, Copy, Debug)]
pub struct DynamicObstacle {
    pub entity: Entity,
    pub min: Vec2,
    pub max: Vec2,
    pub linvel: Vec2,
    pub mass: f32,
    pub handle: RigidBodyHandle,
}

/// Collects all active dynamic body AABBs from [`RapierPhysics`].
pub fn collect_dynamic_obstacles(physics: &RapierPhysics) -> Vec<DynamicObstacle> {
    let mut obstacles = Vec::new();
    for (handle, body) in physics.bodies.iter() {
        if !body.is_dynamic() {
            continue;
        }
        for col_handle in body.colliders() {
            if let Some(col) = physics.colliders.get(*col_handle) {
                let aabb = col.compute_aabb();
                let ent = user_data_to_entity(body.user_data);
                obstacles.push(DynamicObstacle {
                    entity: ent,
                    min: vec2(aabb.mins.x, aabb.mins.y),
                    max: vec2(aabb.maxs.x, aabb.maxs.y),
                    linvel: from_rapier_vec(body.linvel()),
                    mass: body.mass(),
                    handle,
                });
            }
        }
    }
    // Sort deterministically by entity
    obstacles.sort_by_key(|o| (o.entity.index(), o.entity.generation()));
    obstacles
}


/// Particle-rigidbody interaction system: bounces particles or applies collision impulses to dynamic bodies.
pub fn particle_rigidbody_interaction_system(
    mut physics: ResMut<RapierPhysics>,
    mut particle_pool: ResMut<ParticlePool>,
) {
    let pool_len = particle_pool.len();
    if pool_len == 0 {
        return;
    }

    let obstacles = collect_dynamic_obstacles(&physics);
    if obstacles.is_empty() {
        return;
    }

    let particles = particle_pool.particles_mut();
    for p in particles.iter_mut() {
        if (p.flags & astratomic_sim::particle::PARTICLE_IGNORE_WORLD_COLLISION) != 0 {
            continue;
        }

        let world_x = p.pos[0];
        let world_y = -p.pos[1];

        for obs in &obstacles {
            if world_x >= obs.min.x
                && world_x <= obs.max.x
                && world_y >= obs.min.y
                && world_y <= obs.max.y
            {
                // Apply small impulse to dynamic body
                if let Some(rb) = physics.bodies.get_mut(obs.handle) {
                    let imp_x = p.vel[0] * 0.02;
                    let imp_y = -p.vel[1] * 0.02;
                    rb.apply_impulse(rapier2d::math::Vec2::new(imp_x, imp_y), true);
                }

                // Bounce or deflect particle
                if (p.atom.flags & FLAG_PARTICLE_BOUNCE) != 0 {
                    p.vel[1] = -p.vel[1] * 0.5;
                    p.vel[0] = p.vel[0] * 0.7;
                } else {
                    let center_x = (obs.min.x + obs.max.x) * 0.5;
                    let center_y = (obs.min.y + obs.max.y) * 0.5;
                    let dx = world_x - center_x;
                    let dy = world_y - center_y;
                    let sx = if dx.abs() > 0.1 { dx.signum() } else { 1.0 };
                    let sy = if dy.abs() > 0.1 { dy.signum() } else { 1.0 };
                    p.vel[0] = sx * 1.5;
                    p.vel[1] = -sy * 1.5;
                }
            }
        }
    }
}

/// Stamps temporary [`AtomState::Object`] atoms across the footprints of all active
/// [`PixelRigidBody`] entities before cellular automata simulation runs, preventing
/// falling powder (sand, gravel, dirt, etc.) from passing through rigidbodies,
/// while allowing fluids to flow through.
pub fn stamp_rigidbody_objects_system(
    sim_settings: Option<Res<SimulationSettings>>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut stamp: ResMut<RigidbodySimulationStamp>,
    mut dirty_rects: ResMut<DirtyRects>,
    entities: Res<Entities>,
    pixel_bodies: Comp<PixelRigidBody>,
    handles: Comp<PhysicsBodyHandle>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    stamp.stamped_atoms.clear();

    let mut ordered_entities: Vec<Entity> =
        entities.iter_with(&pixel_bodies).map(|(e, _)| e).collect();
    ordered_entities.sort_unstable();

    for ent in ordered_entities {
        let Some(body) = pixel_bodies.get(ent) else {
            continue;
        };
        let (pos, angle) = match handles.get(ent) {
            Some(h) => (h.pos, h.angle),
            None => (body.pos, body.angle),
        };

        let half_w = body.width as f32 * 0.5;
        let half_h = body.height as f32 * 0.5;
        let (sin_a, cos_a) = angle.sin_cos();
        let x_extent = half_w * cos_a.abs() + half_h * sin_a.abs();
        let y_extent = half_w * sin_a.abs() + half_h * cos_a.abs();

        let min_wx = (pos.x - x_extent).floor() as i32;
        let max_wx = (pos.x + x_extent).ceil() as i32;
        let min_wy = (pos.y - y_extent).floor() as i32;
        let max_wy = (pos.y + y_extent).ceil() as i32;

        for wy in min_wy..=max_wy {
            for wx in min_wx..=max_wx {
                if let Some(atom) = body.sample_rigidbody_at_world_cell(wx, wy, pos, sin_a, cos_a) {
                    if !atom.is_air() && !atom.is_void() {
                        let global = world_cell_to_global_atom(wx, wy);
                        let chunk_pos = global_to_chunk(global);

                        if let Some(chunk_atom) = chunk_world.get_atom(&chunk_pos) {
                            if chunk_atom.is_void() {
                                let original = *chunk_atom;
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
    }
}

/// Cleans up temporary [`AtomState::Object`] atoms after cellular automata simulation runs,
/// restoring original atoms in [`ChunkWorld`].
pub fn clear_rigidbody_objects_system(
    mut chunk_world: ResMut<ChunkWorld>,
    mut stamp: ResMut<RigidbodySimulationStamp>,
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

pub use rapier2d::math::Vector as RapierVector;
pub use rapier2d::pipeline::{
    DebugColor, DebugRenderBackend, DebugRenderMode, DebugRenderObject, DebugRenderPipeline,
    DebugRenderStyle,
};

/// Runs Rapier's built-in [`DebugRenderPipeline`] across active bodies, colliders, and contacts.
pub fn debug_render_rapier(
    physics: &RapierPhysics,
    backend: &mut impl DebugRenderBackend,
    mode: DebugRenderMode,
) {
    let mut pipeline = DebugRenderPipeline::new(DebugRenderStyle::default(), mode);
    pipeline.render(
        backend,
        &physics.bodies,
        &physics.colliders,
        &physics.impulse_joints,
        &physics.multibody_joints,
        &physics.narrow_phase,
        &physics.soft_bodies,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::KinematicActor;

    #[test]
    fn test_pixel_rigidbody_inverse_rasterization_zero_holes() {
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 10.0));
        assert_eq!(crate_body.width, 16);
        assert_eq!(crate_body.height, 16);

        // 1. Unrotated body at (10.0, 10.0):
        // Spans local x: 0..16, y: 0..16 => world cells wx in 2..18, wy in 2..18
        for wy in 2..18 {
            for wx in 2..18 {
                let atom = crate_body.sample_rigidbody_at_world_cell(wx, wy, crate_body.pos, 0.0, 1.0);
                assert!(atom.is_some(), "Cell ({wx}, {wy}) must contain an atom");
                assert!(!atom.unwrap().is_air(), "Crate atom should not be air");
            }
        }
        // Outside bounds returns None
        assert!(crate_body.sample_rigidbody_at_world_cell(1, 10, crate_body.pos, 0.0, 1.0).is_none());
        assert!(crate_body.sample_rigidbody_at_world_cell(18, 10, crate_body.pos, 0.0, 1.0).is_none());

        // 2. Rotated 45 degrees:
        let angle = std::f32::consts::FRAC_PI_4;
        let (sin_a, cos_a) = angle.sin_cos();
        let half_w = 8.0_f32;
        let half_h = 8.0_f32;
        let x_extent = half_w * cos_a.abs() + half_h * sin_a.abs();
        let y_extent = half_w * sin_a.abs() + half_h * cos_a.abs();
        let min_wx = (crate_body.pos.x - x_extent).floor() as i32;
        let max_wx = (crate_body.pos.x + x_extent).ceil() as i32;
        let min_wy = (crate_body.pos.y - y_extent).floor() as i32;
        let max_wy = (crate_body.pos.y + y_extent).ceil() as i32;

        let mut sampled_solid_cells = 0;
        for wy in min_wy..=max_wy {
            for wx in min_wx..=max_wx {
                if let Some(atom) = crate_body.sample_rigidbody_at_world_cell(wx, wy, crate_body.pos, sin_a, cos_a) {
                    if !atom.is_air() {
                        sampled_solid_cells += 1;
                    }
                }
            }
        }
        // A 16x16 crate has 256 pixels. Rotated onto an integer grid, inverse rasterization
        // guarantees area coverage without gaps (sampled cells >= 240).
        assert!(
            sampled_solid_cells >= 240,
            "Rotated crate must rasterize without internal holes! Got {} cells",
            sampled_solid_cells
        );

        // Inscribed square check: within radius 4 of center, 100% of cells must be solid
        for dy in -4..=4 {
            for dx in -4..=4 {
                let wx = (crate_body.pos.x as i32) + dx;
                let wy = (crate_body.pos.y as i32) + dy;
                let atom = crate_body.sample_rigidbody_at_world_cell(wx, wy, crate_body.pos, sin_a, cos_a);
                assert!(
                    atom.is_some() && !atom.unwrap().is_air(),
                    "Rotated core cell ({wx}, {wy}) must not have internal holes"
                );
            }
        }
    }

    #[test]
    fn test_lazy_awake_aabb_terrain_collider_sync() {
        let mut world = ChunkWorld::default();
        let mut chunk = Chunk::default();

        // Fill row ly = 63 (bottom of chunk 0, 0, wy = -64) with solid Stone
        for x in 0..CHUNK_SIZE {
            chunk.grid_mut().atoms[63 * CHUNK_SIZE + x] =
                Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
        }

        // Set powder (free-falling) at ly = 50: should NOT be included in static collider
        let mut falling_sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        falling_sand.set_free_falling(true);
        chunk.grid_mut().atoms[50 * CHUNK_SIZE + 10] = falling_sand;

        // Set liquid at ly = 55: should NOT be included
        chunk.grid_mut().atoms[55 * CHUNK_SIZE + 10] =
            Atom::with_state(3, AtomState::Liquid, [0, 0, 255, 255]);

        chunk.collider_dirty = true;
        world.add_chunk(ivec2(0, 0), chunk);

        let mut physics = RapierPhysics::default();
        // Dynamic awake body near chunk (0, 0)
        let rb = RigidBodyBuilder::dynamic()
            .translation(rapier2d::math::Vec2::new(10.0, -50.0))
            .build();
        let rb_h = physics.bodies.insert(rb);
        let col = ColliderBuilder::cuboid(4.0, 4.0).build();
        physics
            .colliders
            .insert_with_parent(col, rb_h, &mut physics.bodies);

        assert!(world.chunks.get(&ivec2(0, 0)).unwrap().collider_dirty);
        assert!(!physics.chunk_colliders.contains_key(&(0, 0)));

        sync_awake_terrain_colliders(&mut physics, &mut world);

        // Now chunk collider is built
        assert!(physics.chunk_colliders.contains_key(&(0, 0)));
        assert!(!world.chunks.get(&ivec2(0, 0)).unwrap().collider_dirty);

        let handle = physics.chunk_colliders[&(0, 0)];
        let col = physics.colliders.get(handle).unwrap();
        let polyline = col.shape().as_polyline().unwrap();
        assert!(!polyline.vertices().is_empty());

        // Second sync without dirtying chunk preserves existing handle
        sync_awake_terrain_colliders(&mut physics, &mut world);
        assert_eq!(physics.chunk_colliders[&(0, 0)], handle);
    }

    #[test]
    fn test_dynamic_crate_falling_and_landing_on_chunk_terrain() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<RapierPhysics>();

        // Create solid floor in chunk (0, 0) at ly = 0 (world y = -1 to 0)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for x in 0..CHUNK_SIZE {
                chunk.grid_mut().atoms[0 * CHUNK_SIZE + x] =
                    Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Spawn dynamic crate at y = 20.0
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 20.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Step simulation across 40 ticks
        for _ in 0..40 {
            world.run_system(step_rapier_physics_system, ());
        }

        // Crate has fallen and landed on the floor
        let handles = world.component::<PhysicsBodyHandle>();
        let handle = handles.get(crate_ent).unwrap();
        assert!(
            handle.pos.y >= 7.0 && handle.pos.y <= 9.5,
            "Crate should rest on floor at y~8, got {}",
            handle.pos.y
        );
    }

    #[test]
    fn test_floating_crate_buoyancy_in_water() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<RapierPhysics>();
        world.insert_resource(Materials::default_builtin());

        // Fill water in chunk (0, 0) for ly in 0..30 (world wy in -30..=-1)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for ly in 0..30 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(3, AtomState::Liquid, [0, 0, 255, 255]);
                }
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Submerged wooden crate in water (weight 0.5 < water weight 1.0)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(16.0, -18.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Step 15 ticks
        for _ in 0..15 {
            world.run_system(step_rapier_physics_system, ());
        }

        let handles = world.component::<PhysicsBodyHandle>();
        let handle = handles.get(crate_ent).unwrap();
        let physics = world.resource::<RapierPhysics>();
        let rb = physics.bodies.get(handle.handle()).unwrap();

        // Archimedes buoyancy accelerates the submerged wooden crate upward!
        assert!(
            handle.pos.y > -18.0,
            "Submerged wooden crate must rise upward, got initial -18.0 -> pos.y = {}",
            handle.pos.y
        );
        assert!(
            rb.linvel().y > 0.0,
            "Wooden crate in water must have positive upward velocity, got vy = {}",
            rb.linvel().y
        );

        // Water atoms beneath the crate must remain intact in ChunkWorld (not deleted or ejected)
        let cw = world.resource::<ChunkWorld>();
        let sample_water_cp = global_to_chunk(world_cell_to_global_atom(16, -25));
        let sample_water = cw.get_atom(&sample_water_cp);
        assert!(
            sample_water.is_some() && sample_water.unwrap().is_liquid(),
            "Water pool must stay intact under floating crate"
        );
    }

    #[test]
    fn test_heavy_stone_boulder_sinks_in_water_but_floats_in_dense_lava() {
        // --- 1. Stone Boulder in Water (Stone weight 2.5 > Water weight 1.0 => Sinks) ---
        {
            let mut world = World::default();
            world.init_resource::<Entities>();
            world.init_resource::<ChunkWorld>();
            world.init_resource::<DirtyRects>();
            world.init_resource::<RapierPhysics>();
            world.insert_resource(Materials::default_builtin());

            {
                let mut cw = world.resource_mut::<ChunkWorld>();
                let mut chunk = Chunk::default();
                for ly in 0..30 {
                    for lx in 0..CHUNK_SIZE {
                        chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                            Atom::with_state(3, AtomState::Liquid, [0, 0, 255, 255]); // Water
                    }
                }
                cw.add_chunk(ivec2(0, 0), chunk);
            }

            let stone_atom = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]); // Material 8: Rock (weight 2.5)
            let stone_body = PixelRigidBody::new_box(16, 16, stone_atom, vec2(16.0, -18.0));
            let stone_ent = spawn_pixel_rigidbody(&mut world, None, stone_body);

            for _ in 0..10 {
                world.run_system(step_rapier_physics_system, ());
            }

            let handles = world.component::<PhysicsBodyHandle>();
            let handle = handles.get(stone_ent).unwrap();
            let physics = world.resource::<RapierPhysics>();
            let rb = physics.bodies.get(handle.handle()).unwrap();

            assert!(
                handle.pos.y < -18.0,
                "Dense rock boulder (weight 2.5) must sink in water (weight 1.0), got y = {}",
                handle.pos.y
            );
            assert!(
                rb.linvel().y < 0.0,
                "Rock boulder in water must sink downward (vy < 0.0), got vy = {}",
                rb.linvel().y
            );
        }

        // --- 2. Stone Boulder in Lava (Stone weight 2.5 vs Lava weight 2.2 * 1.6 BUOYANCY => Floats) ---
        {
            let mut world = World::default();
            world.init_resource::<Entities>();
            world.init_resource::<ChunkWorld>();
            world.init_resource::<DirtyRects>();
            world.init_resource::<RapierPhysics>();
            world.insert_resource(Materials::default_builtin());

            {
                let mut cw = world.resource_mut::<ChunkWorld>();
                let mut chunk = Chunk::default();
                for ly in 0..30 {
                    for lx in 0..CHUNK_SIZE {
                        chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                            Atom::with_state(5, AtomState::Liquid, [255, 100, 0, 255]); // Lava (weight 2.2)
                    }
                }
                cw.add_chunk(ivec2(0, 0), chunk);
            }

            let stone_atom = Atom::with_state(8, AtomState::Solid, [128, 128, 128, 255]); // Material 8: Rock (weight 2.5)
            let stone_body = PixelRigidBody::new_box(16, 16, stone_atom, vec2(16.0, -18.0));
            let stone_ent = spawn_pixel_rigidbody(&mut world, None, stone_body);

            for _ in 0..10 {
                world.run_system(step_rapier_physics_system, ());
            }

            let handles = world.component::<PhysicsBodyHandle>();
            let handle = handles.get(stone_ent).unwrap();
            let physics = world.resource::<RapierPhysics>();
            let rb = physics.bodies.get(handle.handle()).unwrap();

            assert!(
                handle.pos.y > -18.0,
                "Dense rock boulder (weight 2.5) must float upward in heavy lava (weight 2.2 * 1.6 = 3.52), got y = {}",
                handle.pos.y
            );
            assert!(
                rb.linvel().y > 0.0,
                "Rock boulder in lava must rise upward (vy > 0.0), got vy = {}",
                rb.linvel().y
            );
        }
    }

    fn test_chunk_world() -> ChunkWorld {
        let mut cw = ChunkWorld::default();
        for cy in -2..=2 {
            for cx in -2..=2 {
                cw.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }
        cw
    }

    #[test]
    fn test_actor_standing_on_crate_and_auto_step_up() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<RapierPhysics>();

        // 1. Landing on crate: 16x16 crate at pos (10.0, 8.0) => top is at y = 16.0
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 8.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Spawn actor falling onto crate from above
        let mut actor = KinematicActor::new(vec2(10.0, 24.0));
        actor.vel = vec2(0.0, -4.0);
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        // Run systems: Rapier Step & KinematicActor Movement System
        world.run_system(step_rapier_physics_system, ());
        world.run_system(crate::actor::kinematic_actor_movement_system, ());

        {
            let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();
            let actors = world.component::<KinematicActor>();
            let a = actors.get(actor_ent).unwrap();
            assert!(a.grounded, "Actor must be grounded on top of crate");
            assert_eq!(a.vel.y, 0.0, "Vertical velocity must zero upon landing");
            let expected_y = handle.pos.y + 8.0 + a.half_size.y + 0.001;
            assert!(
                (a.pos.y - expected_y).abs() < 0.1,
                "Actor should rest on crate top ({}), got {}",
                expected_y,
                a.pos.y
            );
        }

        // 2. Auto-step-up onto a low step crate (height 6.0, step_height 18)
        let low_crate = PixelRigidBody::new_box(
            16,
            6,
            Atom::with_state(1, AtomState::Solid, [100, 100, 100, 255]),
            vec2(30.0, 3.0),
        );
        spawn_pixel_rigidbody(&mut world, None, low_crate);

        let mut walking_actor = KinematicActor::new(vec2(18.0, 6.05));
        walking_actor.vel = vec2(3.0, 0.0);
        walking_actor.step_height = 18;
        let walking_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(walking_ent, walking_actor);

        world.run_system(step_rapier_physics_system, ());
        world.run_system(crate::actor::kinematic_actor_movement_system, ());

        let actors = world.component::<KinematicActor>();
        let wa = actors.get(walking_ent).unwrap();
        assert!(
            wa.pos.y > 6.1,
            "Actor must auto-step-up onto low crate, pos.y = {}",
            wa.pos.y
        );
    }

    #[test]
    fn test_actor_pushes_dynamic_crate() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<RapierPhysics>();

        // Crate of 16x16 at (20.0, 8.0)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(20.0, 8.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Actor walking right into the left side of the crate
        let mut actor = KinematicActor::new(vec2(9.0, 8.0));
        actor.vel = vec2(4.0, 0.0);
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        // Run step_rapier_physics_system:
        // Step A syncs actor body to Rapier with linvel.
        // Step B evaluates contact manifold between actor and dynamic crate, applies impulse.
        world.run_system(step_rapier_physics_system, ());

        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();
        let physics = world.resource::<RapierPhysics>();
        let rb = physics.bodies.get(handle.handle()).unwrap();

        assert!(
            rb.linvel().x > 0.0,
            "Dynamic crate must be pushed to the right by actor, got linvel.x = {}",
            rb.linvel().x
        );
    }

    #[test]
    fn test_actor_continuously_pushes_dynamic_crate_over_multiple_ticks() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<RapierPhysics>();

        // Crate of 16x16 at (20.0, 8.0) -> left edge at 12.0
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(20.0, 8.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Actor at x = 4.0, half_w = 4.0 -> right edge at 8.0 (separated by 4 pixels)
        let mut actor = KinematicActor::new(vec2(4.0, 8.0));
        actor.vel = vec2(2.5, 0.0);
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        let initial_crate_x = 20.0;

        // Run full movement & physics pipeline for 20 ticks:
        // Actor walks right at 2.5 px/tick, contacts crate, pushes it forward continuously
        for _ in 0..20 {
            // Simulated player controller: keeps desired running speed vel.x = 2.5
            if let Some(a) = world.component_mut::<KinematicActor>().get_mut(actor_ent) {
                a.vel.x = 2.5;
            }
            world.run_system(step_rapier_physics_system, ());
            world.run_system(crate::actor::kinematic_actor_movement_system, ());
        }

        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();
        let actors = world.component::<KinematicActor>();
        let a = actors.get(actor_ent).unwrap();

        assert!(
            handle.pos.x > initial_crate_x + 5.0,
            "Crate must be pushed forward significantly across ticks, start={}, now={}",
            initial_crate_x,
            handle.pos.x
        );
        assert!(
            a.pos.x > 10.0,
            "Actor must follow right behind the pushed crate, pos.x = {}",
            a.pos.x
        );
        // Actor must stay behind the crate and not tunnel through
        assert!(
            a.pos.x + a.half_size.x <= handle.pos.x - 8.0 + 1.0,
            "Actor must not tunnel into crate: actor_right={}, crate_left={}",
            a.pos.x + a.half_size.x,
            handle.pos.x - 8.0
        );
    }

    #[test]
    fn test_actor_tilts_plank_off_center() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        let mut physics = RapierPhysics::default();
        physics.gravity = rapier2d::math::Vec2::new(0.0, 0.0); // zero ambient gravity for pure torque test
        world.insert_resource(physics);

        // Wide plank: width 40, height 4 at pos (50.0, 20.0), top at y = 22.0
        let plank = PixelRigidBody::new_box(
            40,
            4,
            Atom::with_state(1, AtomState::Solid, [150, 100, 50, 255]),
            vec2(50.0, 20.0),
        );
        let plank_ent = spawn_pixel_rigidbody(&mut world, None, plank);

        // Actor standing off-center on the far right end of the plank (x = 65.0, center is 50.0)
        // Actor positioned with overlap and downward velocity
        let mut actor = KinematicActor::new(vec2(65.0, 27.0));
        actor.vel = vec2(0.0, -3.0);
        actor.weight = 70.0;
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        world.run_system(step_rapier_physics_system, ());

        let handle = *world.component::<PhysicsBodyHandle>().get(plank_ent).unwrap();
        let physics = world.resource::<RapierPhysics>();
        let rb = physics.bodies.get(handle.handle()).unwrap();

        // Right side pushed down => clockwise rotation in 2D (+X right, +Y up => clockwise torque is negative angvel)
        assert!(
            rb.angvel().abs() > 0.01,
            "Plank must tilt when actor steps on edge, got angvel = {}",
            rb.angvel()
        );
        assert!(
            rb.angvel() < 0.0,
            "Right-side load must tilt plank clockwise (negative angvel), got angvel = {}",
            rb.angvel()
        );
    }

    #[test]
    fn test_actor_rides_moving_boulder() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<RapierPhysics>();

        // Boulder at (20.0, 8.0) of size 16x16, top at y = 16.0
        let boulder = PixelRigidBody::new_crate(16, 16, vec2(20.0, 8.0));
        let boulder_ent = spawn_pixel_rigidbody(&mut world, None, boulder);

        // Set horizontal velocity on boulder
        let handle = *world.component::<PhysicsBodyHandle>().get(boulder_ent).unwrap();
        {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let rb = physics.bodies.get_mut(handle.handle()).unwrap();
            rb.set_linvel(rapier2d::math::Vec2::new(120.0, 0.0), true);
        }

        // Actor resting on top of the boulder at y = 22.001 (feet at y = 16.001)
        let mut actor = KinematicActor::new(vec2(20.0, 22.001));
        actor.grounded = true;
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        let initial_x = 20.0;
        // Step movement system (Step C inherits floor surface velocity)
        world.run_system(crate::actor::kinematic_actor_movement_system, ());

        let actors = world.component::<KinematicActor>();
        let a = actors.get(actor_ent).unwrap();
        assert!(
            a.pos.x > initial_x + 1.0,
            "Actor riding on moving boulder must move with it, pos.x = {}, initial = {}",
            a.pos.x,
            initial_x
        );
    }

    #[test]
    fn test_heavy_falling_rigidbody_pushes_actor_against_terrain() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<RapierPhysics>();

        // Solid floor in chunk (0, 0) from wy = -10 to wy = 0 (top of ground at y = 1.0)
        {
            let mut chunk_world = ChunkWorld::default();
            let mut chunk = Chunk::default();
            for ly in 0..10 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
                }
            }
            chunk_world.add_chunk(ivec2(0, 0), chunk);
            world.insert_resource(chunk_world);
        }

        // Heavy crate (mass 500) falling from above
        let heavy_crate = PixelRigidBody::new_crate(16, 16, vec2(10.0, 20.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, heavy_crate);
        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();
        {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let rb = physics.bodies.get_mut(handle.handle()).unwrap();
            rb.set_linvel(rapier2d::math::Vec2::new(0.0, -100.0), true);
        }

        // Actor standing on the ground
        let mut actor = KinematicActor::new(vec2(10.0, 7.0));
        actor.grounded = true;
        let actor_ent = world.resource_mut::<Entities>().create();
        world.component_mut::<KinematicActor>().insert(actor_ent, actor);

        // Run physics and movement
        world.run_system(step_rapier_physics_system, ());
        world.run_system(crate::actor::kinematic_actor_movement_system, ());

        let actors = world.component::<KinematicActor>();
        let a = actors.get(actor_ent).unwrap();
        // Ground is solid below y = 1.0; actor half_size.y = 6.0 => actor pos.y cannot penetrate solid ground!
        assert!(
            a.pos.y >= 6.0,
            "Actor must never be pushed inside solid terrain! pos.y = {}",
            a.pos.y
        );
    }

    #[test]
    fn test_particle_colliding_with_dynamic_crate() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<RapierPhysics>();

        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(20.0, 20.0));
        let _crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        let mut pool = ParticlePool::new();
        let mut sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        sand.flags = FLAG_PARTICLE_BOUNCE;

        // In global particle coords (+Y down), world (20.0, 20.0) is pos [20.0, -20.0]
        pool.push_particle(astratomic_sim::particle::Particle::new(
            [20.0, -20.0],
            [4.0, 6.0],
            sand,
        ));

        // Interaction system check
        let physics = world.resource::<RapierPhysics>();
        let obstacles = collect_dynamic_obstacles(&physics);
        assert_eq!(obstacles.len(), 1);

        for p in pool.particles_mut().iter_mut() {
            let world_x = p.pos[0];
            let world_y = -p.pos[1];
            for obs in &obstacles {
                if world_x >= obs.min.x
                    && world_x <= obs.max.x
                    && world_y >= obs.min.y
                    && world_y <= obs.max.y
                {
                    if (p.atom.flags & FLAG_PARTICLE_BOUNCE) != 0 {
                        p.vel[1] = -p.vel[1] * 0.5;
                        p.vel[0] = p.vel[0] * 0.7;
                    }
                }
            }
        }

        assert!(
            pool.particles()[0].vel[1] < 0.0,
            "Bouncy particle should reverse vertical velocity"
        );
    }

    #[test]
    fn test_marching_squares_rdp_slope_and_despeckle() {
        let mut chunk = Chunk::default();

        // 1. Despeckling: single isolated atom has 0 4-neighbors -> should be completely ignored
        chunk.grid_mut().atoms[10 * CHUNK_SIZE + 10] =
            Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
        let loops = build_chunk_terrain_loops(&chunk);
        assert!(
            loops.is_empty(),
            "Single isolated crumb atom should be despeckled and have no colliders"
        );

        // 2. 45-degree slope staircase: build a 16-pixel diagonal ramp
        for i in 0..16 {
            for y in (30 + i)..CHUNK_SIZE {
                chunk.grid_mut().atoms[y * CHUNK_SIZE + (10 + i)] =
                    Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
            }
        }

        let slope_loops = build_chunk_terrain_loops(&chunk);
        assert_eq!(
            slope_loops.len(),
            1,
            "Slope should form a single closed loop"
        );

        let poly = &slope_loops[0];
        // Jagged raw staircase would have over 40 vertices. With RDP (epsilon = 1.4), the diagonal
        // surface collapses into just a few straight segments.
        assert!(
            poly.len() < 15,
            "RDP should simplify jagged staircase down to < 15 vertices, got {}",
            poly.len()
        );

        // 3. Polyline creation produces valid SharedShape::polyline
        let shape = build_chunk_terrain_polyline(&chunk, 0, 0).expect("Polyline shape must build");
        let pl = shape.as_polyline().expect("Shape must be a polyline");
        assert!(!pl.vertices().is_empty());
        assert!(!pl.indices().is_empty());
    }

    #[test]
    fn test_terrain_colliders_radius_and_cleanup_when_no_rigidbodies() {
        let mut world = ChunkWorld::default();
        let mut chunk = Chunk::default();

        // Solid floor in chunk (0, 0)
        for x in 0..CHUNK_SIZE {
            chunk.grid_mut().atoms[63 * CHUNK_SIZE + x] =
                Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
        }
        chunk.collider_dirty = true;
        world.add_chunk(ivec2(0, 0), chunk);

        let mut physics = RapierPhysics::default();

        // 1. If there are NO rigidbodies close, there should not be any terrain colliders
        sync_awake_terrain_colliders(&mut physics, &mut world);
        assert!(
            physics.chunk_colliders.is_empty(),
            "No terrain colliders should exist when there are no dynamic rigidbodies"
        );

        // 2. Add an awake dynamic rigidbody inside chunk (0, 0)
        let rb = RigidBodyBuilder::dynamic()
            .translation(rapier2d::math::Vec2::new(10.0, -30.0))
            .build();
        let rb_h = physics.bodies.insert(rb);
        let col = ColliderBuilder::cuboid(4.0, 4.0).build();
        physics
            .colliders
            .insert_with_parent(col, rb_h, &mut physics.bodies);

        sync_awake_terrain_colliders(&mut physics, &mut world);
        assert!(
            physics.chunk_colliders.contains_key(&(0, 0)),
            "Chunk (0, 0) collider must be created when awake dynamic body is nearby"
        );

        // 3. Teleport the dynamic body far away (e.g. 500 px away)
        if let Some(rb_mut) = physics.bodies.get_mut(rb_h) {
            rb_mut.set_translation(rapier2d::math::Vec2::new(500.0, -500.0), true);
        }

        sync_awake_terrain_colliders(&mut physics, &mut world);
        assert!(
            !physics.chunk_colliders.contains_key(&(0, 0)),
            "Chunk (0, 0) collider must be removed when dynamic body moves outside TERRAIN_COLLIDER_RADIUS"
        );

        // 4. Remove the dynamic body completely -> all terrain colliders cleaned up
        physics.remove_body(rb_h);
        sync_awake_terrain_colliders(&mut physics, &mut world);
        assert!(
            physics.chunk_colliders.is_empty(),
            "All terrain colliders must be cleaned up when no dynamic bodies exist"
        );
    }

    #[test]
    fn test_convex_decomp() {
        // Concave L-shape
        let vertices = vec![
            rapier2d::math::Vec2::new(-8.0, -8.0),
            rapier2d::math::Vec2::new(8.0, -8.0),
            rapier2d::math::Vec2::new(8.0, 0.0),
            rapier2d::math::Vec2::new(0.0, 0.0),
            rapier2d::math::Vec2::new(0.0, 8.0),
            rapier2d::math::Vec2::new(-8.0, 8.0),
        ];
        let indices = vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]];
        let shape = SharedShape::convex_decomposition(&vertices, &indices);
        let compound = shape.as_compound().expect("Should be compound");
        assert!(compound.shapes().len() >= 2, "L-shape decomposes into 2 convex parts");

        let pt = rapier2d::math::Vec2::new(-4.0, -4.0);
        let pose = rapier2d::math::Pose2::identity();
        assert!(shape.contains_point(&pose, pt));

        // Two disconnected squares
        let vertices2 = vec![
            // Square 1: [-12, -4] to [-4, 4]
            rapier2d::math::Vec2::new(-12.0, -4.0),
            rapier2d::math::Vec2::new(-4.0, -4.0),
            rapier2d::math::Vec2::new(-4.0, 4.0),
            rapier2d::math::Vec2::new(-12.0, 4.0),
            // Square 2: [4, -4] to [12, 4]
            rapier2d::math::Vec2::new(4.0, -4.0),
            rapier2d::math::Vec2::new(12.0, -4.0),
            rapier2d::math::Vec2::new(12.0, 4.0),
            rapier2d::math::Vec2::new(4.0, 4.0),
        ];
        let indices2 = vec![
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
        ];
        let shape2 = SharedShape::convex_decomposition(&vertices2, &indices2);
        let compound2 = shape2.as_compound().expect("Should be compound");
        assert_eq!(compound2.shapes().len(), 2, "Two squares decompose into 2 shapes");
    }

    #[test]
    fn test_falling_powder_blocked_by_stamped_rigidbody() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<DirtyRects>();
        world.init_resource::<RigidbodySimulationStamp>();
        world.init_resource::<RapierPhysics>();

        // 16x16 crate at pos (10.0, 10.0) -> spans y from 2.0 to 18.0 (top surface at y = 18.0)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 10.0));
        let _crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Put falling sand in ChunkWorld at wx = 10, wy = 19 (1 pixel directly above the crate)
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        {
            let mut chunk_world = world.resource_mut::<ChunkWorld>();
            let cp = global_to_chunk(world_cell_to_global_atom(10, 19));
            chunk_world.set_atom(cp, sand);
        }

        // Run stamping system
        world.run_system(stamp_rigidbody_objects_system, ());

        {
            let chunk_world = world.resource::<ChunkWorld>();
            // The crate's interior cells must be stamped as Object
            let crate_top_cp = global_to_chunk(world_cell_to_global_atom(10, 17));
            let crate_top_atom = chunk_world.get_atom(&crate_top_cp).unwrap();
            assert!(crate_top_atom.is_object(), "Crate top must be stamped as AtomState::Object");

            // The cell with sand should NOT be overwritten
            let sand_cp = global_to_chunk(world_cell_to_global_atom(10, 19));
            let sand_atom = chunk_world.get_atom(&sand_cp).unwrap();
            assert!(sand_atom.is_powder(), "Sand must remain intact above crate");
        }

        // Run clear system -> crate cells restored to void
        world.run_system(clear_rigidbody_objects_system, ());

        {
            let chunk_world = world.resource::<ChunkWorld>();
            let crate_top_cp = global_to_chunk(world_cell_to_global_atom(10, 17));
            let crate_top_atom = chunk_world.get_atom(&crate_top_cp).unwrap();
            assert!(crate_top_atom.is_void(), "Crate top must be restored back to void");
        }
    }

    #[test]
    fn test_rigidbody_rdp_more_detailed() {
        // Create an irregular pixel body with small 1-pixel notches along its edges
        let mut atoms = vec![Atom::AIR; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                // Cut out 1-pixel notches:
                if (x == 4 && y == 15) || (x == 8 && y == 15) || (x == 12 && y == 15) {
                    continue;
                }
                atoms[y * 16 + x] = Atom::with_state(1, AtomState::Solid, [128, 128, 128, 255]);
            }
        }
        let body = PixelRigidBody {
            width: 16,
            height: 16,
            atoms: std::sync::Arc::new(atoms),
            pos: Vec2::ZERO,
            prev_pos: Vec2::ZERO,
            angle: 0.0,
            prev_angle: 0.0,
            last_linvel: Vec2::ZERO,
            image_handle: None,
            render_cid: None,
            prev_submerged_liquid_count: 0,
        };
        let loops_detailed = build_pixel_body_loops(&body);
        assert!(!loops_detailed.is_empty());
        // At epsilon = 0.7, 1-pixel notches are preserved so the loop has more than 8 vertices
        assert!(
            loops_detailed[0].len() > 8,
            "Detailed RDP with epsilon = 0.7 must preserve 1-pixel notches, got {} vertices",
            loops_detailed[0].len()
        );
    }

    #[test]
    fn test_chunk_colliders_never_rebuild_when_sand_or_water_moves() {
        let world = World::default();
        world.init_resource::<Entities>();
        world.insert_resource(test_chunk_world());
        world.init_resource::<DirtyRects>();
        world.init_resource::<RapierPhysics>();

        // Initially clear collider_dirty on chunk (0, 0)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            if let Some(c) = cw.chunks.get_mut(&ivec2(0, 0)) {
                c.collider_dirty = false;
            }
        }

        // Add sand and water into ChunkWorld
        let sand = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        let water = Atom::with_state(3, AtomState::Liquid, [0, 0, 255, 255]);
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let cp_sand = global_to_chunk(world_cell_to_global_atom(10, 15));
            let cp_water = global_to_chunk(world_cell_to_global_atom(11, 15));
            cw.set_atom(cp_sand, sand);
            cw.set_atom(cp_water, water);
        }

        // Verify that setting sand and water NEVER marked collider_dirty = true!
        {
            let cw = world.resource::<ChunkWorld>();
            let c = cw.chunks.get(&ivec2(0, 0)).unwrap();
            assert!(!c.collider_dirty, "Moving or settling powder/water must NEVER mark collider_dirty = true");
        }
    }

    #[test]
    fn test_dynamic_crate_falling_into_sand_displaces_and_rests_on_sand() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<ParticlePool>();
        world.init_resource::<Materials>();
        world.init_resource::<RapierPhysics>();

        // Create a sand pile in chunk (0, 0): sand fills from wy = -10 to wy = 5
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for ly in 0..16 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
                }
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Spawn a 16x16 crate at y = 25.0 (falling into sand surface at y ~ 0)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 25.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);
        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();

        // Give downward velocity so it slams into sand
        {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let rb = physics.bodies.get_mut(handle.handle()).unwrap();
            rb.set_linvel(rapier2d::math::Vec2::new(0.0, -120.0), true);
        }

        // Step simulation for 60 ticks
        for _ in 0..60 {
            world.run_system(step_rapier_physics_system, ());
        }

        // Verify:
        // 1. Real sand particles were thrown into ParticlePool on impact
        let pool = world.resource::<ParticlePool>();
        assert!(pool.len() > 0, "Impact with sand must throw sand particles into ParticlePool");

        // 2. Crate rests on top of sand pile and does not fall to the bottom of the chunk!
        let handles = world.component::<PhysicsBodyHandle>();
        let h = handles.get(crate_ent).unwrap();
        assert!(
            h.pos.y > -8.0,
            "Crate must rest on top of sand pile, got y = {}",
            h.pos.y
        );

        // 3. Gravity scale was reduced by sand impediment
        let physics = world.resource::<RapierPhysics>();
        let rb = physics.bodies.get(handle.handle()).unwrap();
        assert!(
            rb.gravity_scale() < 0.5,
            "Resting on sand must decrease gravity scale via impediment, got {}",
            rb.gravity_scale()
        );
    }

    #[test]
    fn test_digging_sand_under_resting_crate_wakes_and_falls() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<ParticlePool>();
        world.init_resource::<Materials>();
        world.init_resource::<RapierPhysics>();

        // Create a sand pile in chunk (0, 0)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for ly in 0..16 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
                }
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Spawn crate resting on sand
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 10.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);
        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();

        // Step 20 ticks so crate settles on sand
        for _ in 0..20 {
            world.run_system(step_rapier_physics_system, ());
        }

        let resting_y = {
            let handles = world.component::<PhysicsBodyHandle>();
            handles.get(crate_ent).unwrap().pos.y
        };

        // Dig out all sand beneath the crate (turn chunk into air)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            if let Some(chunk) = cw.chunks.get_mut(&ivec2(0, 0)) {
                chunk.atoms_mut().fill(Atom::AIR);
            }
        }

        // Step 1 tick -> crate wakes up and restores gravity scale to 1.0!
        world.run_system(step_rapier_physics_system, ());
        {
            let physics = world.resource::<RapierPhysics>();
            let rb = physics.bodies.get(handle.handle()).unwrap();
            assert!(
                (rb.gravity_scale() - 1.0).abs() < 1e-3,
                "Digging sand must restore gravity scale to 1.0, got {}",
                rb.gravity_scale()
            );
        }

        // Step 30 more ticks -> crate falls into the dug hole!
        for _ in 0..30 {
            world.run_system(step_rapier_physics_system, ());
        }

        let fallen_y = {
            let handles = world.component::<PhysicsBodyHandle>();
            handles.get(crate_ent).unwrap().pos.y
        };
        assert!(
            fallen_y < resting_y - 10.0,
            "Crate must fall into dug hole! resting_y = {}, fallen_y = {}",
            resting_y,
            fallen_y
        );
    }

    #[test]
    fn test_crate_slamming_into_sand_ejects_particles_outside_hull_in_v_plume() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<ParticlePool>();
        world.init_resource::<ParticleRigidBodyObstacles>();
        world.init_resource::<Materials>();
        world.init_resource::<RapierPhysics>();

        // Create a sand pile in chunk (0, 0) for ly in 0..16 (wy in -16..=-1)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for ly in 0..16 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
                }
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Spawn a 16x16 crate at y = 8.0 (bottom is at y = 0.0, top is at 16.0)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(16.0, 8.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body.clone());
        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();

        // Give high downward velocity to impact the sand pile
        {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let rb = physics.bodies.get_mut(handle.handle()).unwrap();
            rb.set_linvel(rapier2d::math::Vec2::new(0.0, -140.0), true);
        }

        // Step physics ticks (impact & ejection)
        for _ in 0..2 {
            world.run_system(step_rapier_physics_system, ());
        }

        let pool = world.resource::<ParticlePool>();
        assert!(
            pool.len() > 0,
            "Slamming into sand must eject particles"
        );

        let physics = world.resource::<RapierPhysics>();
        let rb = physics.bodies.get(handle.handle()).unwrap();
        let cur_pos = from_rapier_vec(rb.translation());
        let (sin_a, cos_a) = rb.rotation().angle().sin_cos();

        for p in pool.particles() {
            // Convert particle simulation pos to world cell
            let wx = p.pos[0].floor() as i32;
            let wy = -(p.pos[1].floor() as i32) - 1;

            // 1. Particle MUST NOT be inside the crate's interior!
            let rb_sample = crate_body.sample_rigidbody_at_world_cell(wx, wy, cur_pos, sin_a, cos_a);
            assert!(
                rb_sample.is_none() || rb_sample.unwrap().is_air(),
                "Ejected particle at ({}, {}) was spawned INSIDE the rigidbody hull!",
                wx,
                wy
            );

            // 2. Upward V-plume: in simulation coordinates, upward Y velocity is negative!
            assert!(
                p.vel[1] < 0.0,
                "Ejected particle must have upward velocity in simulation space (vy < 0), got vy = {}",
                p.vel[1]
            );
        }
    }

    #[test]
    fn test_crate_moving_in_water_spawns_real_and_vfx_surface_splashes_while_floating() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<ChunkWorld>();
        world.init_resource::<DirtyRects>();
        world.init_resource::<ParticlePool>();
        world.init_resource::<ParticleRigidBodyObstacles>();
        world.insert_resource(Materials::default_builtin());
        world.init_resource::<RapierPhysics>();

        // Fill water in chunk (0, 0) for ly in 0..30 (wy in -30..=-1)
        {
            let mut cw = world.resource_mut::<ChunkWorld>();
            let mut chunk = Chunk::default();
            for ly in 0..30 {
                for lx in 0..CHUNK_SIZE {
                    chunk.grid_mut().atoms[ly * CHUNK_SIZE + lx] =
                        Atom::with_state(3, AtomState::Liquid, [0, 0, 255, 255]);
                }
            }
            cw.add_chunk(ivec2(0, 0), chunk);
        }

        // Spawn a 16x16 wooden crate crossing the water surface at y = 4.0
        // (bottom is at y = -4.0, dipping into water)
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(16.0, 4.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body.clone());
        let handle = *world.component::<PhysicsBodyHandle>().get(crate_ent).unwrap();

        // Give downward entry velocity to trigger waterline splashes
        {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let rb = physics.bodies.get_mut(handle.handle()).unwrap();
            rb.set_linvel(rapier2d::math::Vec2::new(10.0, -100.0), true);
        }

        // Step 3 ticks
        for _ in 0..3 {
            world.run_system(step_rapier_physics_system, ());
        }

        let pool = world.resource::<ParticlePool>();
        assert!(
            pool.len() > 0,
            "Entering water surface must spawn particles"
        );

        let mut has_real_liquid = false;
        let mut has_vfx_foam = false;

        for p in pool.particles() {
            if p.atom.life_5bit() == 0 {
                has_real_liquid = true;
                assert_eq!(p.atom.material_id, 3, "Real matter droplet must have water material ID 3");
            } else {
                has_vfx_foam = true;
                assert_eq!(p.atom.material_id, 0, "VFX foam must have ephemeral material ID 0");
            }
        }

        assert!(has_vfx_foam, "Waterline impact must produce VFX mist/foam particles");
        assert!(has_real_liquid, "High speed water entry must produce real matter liquid droplets");

        // The underwater pool remains intact
        let cw = world.resource::<ChunkWorld>();
        let deep_water_cp = global_to_chunk(world_cell_to_global_atom(16, -20));
        let deep_water = cw.get_atom(&deep_water_cp);
        assert!(
            deep_water.is_some() && deep_water.unwrap().is_liquid(),
            "Deep underwater pool must remain intact"
        );
    }
}
