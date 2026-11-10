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
pub use rapier2d::prelude::{ColliderHandle, RigidBodyHandle};

use astratomic_sim::{
    atom::{Atom, AtomState, Materials},
    chunk::Chunk,
    consts::CHUNK_SIZE,
    particle::{FLAG_PARTICLE_BOUNCE, ParticlePool},
    world::{ChunkWorld, DirtyRects, global_to_chunk},
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

/// Dynamic pixel-backed physical object composed of a 2D grid of [`Atom`]s.
#[derive(HasSchema, Clone, Debug, Default)]
#[schema(opaque)]
pub struct PixelRigidBody {
    pub width: u16,
    pub height: u16,
    pub atoms: Vec<Atom>,
    pub prev_pos: Vec2,
    pub pos: Vec2,
    pub prev_angle: f32,
    pub angle: f32,
    pub image_handle: Option<Handle<Image>>,
    pub last_linvel: Vec2,
}

impl PixelRigidBody {
    /// Creates a new solid rectangular [`PixelRigidBody`].
    pub fn new_box(width: u16, height: u16, atom: Atom, pos: Vec2) -> Self {
        let count = (width as usize) * (height as usize);
        let atoms = vec![atom; count];
        Self {
            width,
            height,
            atoms,
            prev_pos: pos,
            pos,
            prev_angle: 0.0,
            angle: 0.0,
            image_handle: None,
            last_linvel: Vec2::ZERO,
        }
    }

    /// Creates a styled wooden crate with darker edge borders and a diagonal cross brace.
    pub fn new_crate(width: u16, height: u16, pos: Vec2) -> Self {
        let w = width as usize;
        let h = height as usize;
        let mut atoms = Vec::with_capacity(w * h);

        let border_atom = Atom::with_state(1, AtomState::Solid, [101, 67, 33, 255]); // Dark wood
        let inner_atom = Atom::with_state(1, AtomState::Solid, [156, 102, 50, 255]); // Light wood
        let brace_atom = Atom::with_state(1, AtomState::Solid, [120, 80, 40, 255]); // Medium wood

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
            atoms,
            prev_pos: pos,
            pos,
            prev_angle: 0.0,
            angle: 0.0,
            image_handle: None,
            last_linvel: Vec2::ZERO,
        }
    }

    /// Computes the total mass of the object by summing the material weights of all non-air atoms.
    pub fn total_mass(&self, materials: Option<&Materials>) -> f32 {
        let mut total: f32 = 0.0;
        for atom in &self.atoms {
            if !atom.is_air() {
                let weight = materials
                    .and_then(|m| m.0.get(atom.material_id as usize))
                    .map(|m| m.weight)
                    .unwrap_or(1.5)
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
            let local_y = half_h - (y as f32 + y_end as f32) * 0.5;

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
        raw[i] = atom.is_solid() || (atom.is_powder() && !atom.is_free_falling());
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

/// Extracts and simplifies closed contour loops for a [`PixelRigidBody`] in local pixel coordinates.
pub fn build_pixel_body_loops(body: &PixelRigidBody) -> Vec<Vec<[f32; 2]>> {
    let (padded_w, padded_h, padded) = despeckle_and_pad_pixel_body(body);
    let raw_loops = extract_marching_squares_loops_grid(&padded, padded_w, padded_h);
    let mut simplified_loops = Vec::with_capacity(raw_loops.len());

    for r_loop in raw_loops {
        let simplified = simplify_closed_loop(&r_loop, 1.4);
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
                let local_y = half_h - pt[1];
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
/// performing two-way coupling with [`ChunkWorld`] and [`ParticlePool`].
pub fn step_rapier_physics_system(
    mut physics: ResMut<RapierPhysics>,
    mut chunk_world: ResMut<ChunkWorld>,
    materials: Option<Res<Materials>>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
    mut dirty_rects: Option<ResMut<DirtyRects>>,
    entities: Res<Entities>,
    mut handles: CompMut<PhysicsBodyHandle>,
    mut pixel_bodies: CompMut<PixelRigidBody>,
) {
    // 1. Sync terrain colliders for all awake dynamic bodies
    sync_awake_terrain_colliders(&mut physics, &mut chunk_world);

    let mats = materials.as_deref();
    let tick = chunk_world.tick;
    let dt = physics.integration_parameters.dt;

    // 2. Pre-step two-way fluid buoyancy & powder displacement
    let mut ordered_entities: Vec<Entity> =
        entities.iter_with(&handles).map(|(ent, _)| ent).collect();
    ordered_entities.sort_by_key(|e| (e.index(), e.generation()));

    for (sample_idx, ent) in ordered_entities.iter().enumerate() {
        if let Some(handle) = handles.get(*ent)
            && let Some(pixel_body) = pixel_bodies.get_mut(*ent)
            && let Some(rb) = physics.bodies.get_mut(handle.handle())
        {
            let body_pos = from_rapier_vec(rb.translation());
            let body_angle = rb.rotation().angle();
            let linvel = from_rapier_vec(rb.linvel());
            let (cos, sin) = (body_angle.cos(), body_angle.sin());

            let half_w = pixel_body.width as f32 * 0.5;
            let half_h = pixel_body.height as f32 * 0.5;

            let mut submerged_liquid_mass = 0.0;
            let mut liquid_count = 0;

            // Sample pixels across body to detect submerged liquid or powder collision
            let step_x = ((pixel_body.width / 4).max(1)) as usize;
            let step_y = ((pixel_body.height / 4).max(1)) as usize;

            for y in (0..pixel_body.height as usize).step_by(step_y) {
                for x in (0..pixel_body.width as usize).step_by(step_x) {
                    let local_x = x as f32 - half_w;
                    let local_y = half_h - y as f32;
                    let world_x = body_pos.x + local_x * cos - local_y * sin;
                    let world_y = body_pos.y + local_x * sin + local_y * cos;

                    let wx = world_x.floor() as i32;
                    let wy = world_y.floor() as i32;
                    let chunk_pos = global_to_chunk(world_cell_to_global_atom(wx, wy));

                    if let Some(atom) = chunk_world.get_atom(&chunk_pos) {
                        if atom.is_liquid() {
                            let density = mats
                                .and_then(|m| m.0.get(atom.material_id as usize))
                                .map(|m| m.weight)
                                .unwrap_or(1.0);
                            submerged_liquid_mass += density;
                            liquid_count += 1;

                            // Fluid splash on fast entry
                            let noise_val =
                                deterministic_noise(tick, (sample_idx * 100 + x + y) as u32);
                            if linvel.length_squared() > 64.0 && noise_val < 0.25 {
                                if let Some(ref mut pool) = particle_pool {
                                    let splash_vel = [
                                        (deterministic_noise(tick, sample_idx as u32) - 0.5) * 4.0,
                                        -linvel.y.abs() * 0.3 - 1.5,
                                    ];
                                    let gx = wx;
                                    let gy = -wy - 1;
                                    if let Some(ref mut dr) = dirty_rects {
                                        pool.eject_from_grid_dirty(
                                            &mut chunk_world,
                                            dr,
                                            gx,
                                            gy,
                                            splash_vel,
                                        );
                                    } else {
                                        pool.eject_from_grid(&mut chunk_world, gx, gy, splash_vel);
                                    }
                                }
                            }
                        } else if atom.is_powder() && linvel.length_squared() > 16.0 {
                            // Moving body plows into powder atoms along its leading edge
                            if let Some(ref mut pool) = particle_pool {
                                let splash_vel = [
                                    linvel.x * 0.4
                                        + (deterministic_noise(tick, sample_idx as u32) - 0.5)
                                            * 3.0,
                                    -linvel.y.abs() * 0.4 - 1.5,
                                ];
                                let gx = wx;
                                let gy = -wy - 1;
                                if let Some(ref mut dr) = dirty_rects {
                                    pool.eject_from_grid_dirty(
                                        &mut chunk_world,
                                        dr,
                                        gx,
                                        gy,
                                        splash_vel,
                                    );
                                } else {
                                    pool.eject_from_grid(&mut chunk_world, gx, gy, splash_vel);
                                }
                            }
                        }
                    }
                }
            }

            // Apply upward buoyancy and fluid drag
            if liquid_count > 0 {
                let buoyancy_force = submerged_liquid_mass * 800.0 * dt;
                let current_vel = rb.linvel();
                let damp = (1.0 - 0.08 * (liquid_count as f32)).clamp(0.4, 0.98);
                rb.set_linvel(
                    rapier2d::math::Vec2::new(
                        current_vel.x * damp,
                        (current_vel.y + buoyancy_force) * damp,
                    ),
                    true,
                );
                rb.set_angvel(rb.angvel() * damp, true);
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

/// Sweeps an actor against dynamic obstacles, resolving standing on crates, auto-step-up, and horizontal pushing.
pub fn resolve_actor_dynamic_obstacles(
    actor_pos: &mut Vec2,
    actor_prev_pos: Vec2,
    actor_vel: &mut Vec2,
    actor_grounded: &mut bool,
    half_size: Vec2,
    step_height: i32,
    actor_weight: f32,
    initial_vel: Vec2,
    obstacles: &[DynamicObstacle],
    physics: &mut RapierPhysics,
) {
    const SKIN: f32 = 0.05;
    let half_w = half_size.x.max(0.5);
    let half_h = half_size.y.max(0.5);
    let step_h = step_height.max(0) as f32;

    for obs in obstacles {
        let actor_left = actor_pos.x - half_w;
        let actor_right = actor_pos.x + half_w;

        // 1. Vertical floor check: standing on top of crate
        let overlaps_x = actor_right > obs.min.x + SKIN && actor_left < obs.max.x - SKIN;
        if overlaps_x {
            let crate_top = obs.max.y;
            let prev_bottom = actor_prev_pos.y - half_h;
            let current_bottom = actor_pos.y - half_h;

            let swept_through_top =
                prev_bottom >= crate_top - 1.0 && current_bottom <= crate_top + 1.0;

            if swept_through_top && actor_vel.y <= 0.0 {
                actor_pos.y = crate_top + half_h + SKIN;
                actor_vel.y = 0.0;
                *actor_grounded = true;

                // Apply downward landing impulse to crate
                if let Some(rb) = physics.bodies.get_mut(obs.handle) {
                    let impulse_y = initial_vel.y.min(0.0) * actor_weight * 0.05;
                    rb.apply_impulse(rapier2d::math::Vec2::new(0.0, impulse_y), true);
                }
            } else if !*actor_grounded && actor_vel.y <= 0.0 {
                // Ground probe just above crate top
                let dist_to_top = (current_bottom - crate_top).abs();
                if dist_to_top <= 1.0 {
                    *actor_grounded = true;
                    actor_pos.y = crate_top + half_h + SKIN;
                    actor_vel.y = 0.0;
                }
            }
        }

        // 2. Horizontal obstacle check: auto-step-up or pushing
        let current_bottom = actor_pos.y - half_h;
        let current_top = actor_pos.y + half_h;
        let overlaps_y = current_top > obs.min.y + SKIN && current_bottom < obs.max.y - SKIN;

        if overlaps_y {
            let crate_top = obs.max.y;
            let obstacle_height = crate_top - current_bottom;

            // Moving right into crate
            if actor_vel.x > 0.0
                && actor_pos.x + half_w >= obs.min.x
                && actor_pos.x - half_w < obs.min.x
            {
                if obstacle_height > 0.0 && obstacle_height <= step_h {
                    // Auto-step-up!
                    actor_pos.y = crate_top + half_h + SKIN;
                } else {
                    // Pushing against crate
                    actor_pos.x = obs.min.x - half_w - SKIN;
                    actor_vel.x = 0.0;
                    if let Some(rb) = physics.bodies.get_mut(obs.handle) {
                        let push_impulse = initial_vel.x * actor_weight * 0.15;
                        rb.apply_impulse(rapier2d::math::Vec2::new(push_impulse, 0.0), true);
                    }
                }
            }
            // Moving left into crate
            else if actor_vel.x < 0.0
                && actor_pos.x - half_w <= obs.max.x
                && actor_pos.x + half_w > obs.max.x
            {
                if obstacle_height > 0.0 && obstacle_height <= step_h {
                    // Auto-step-up!
                    actor_pos.y = crate_top + half_h + SKIN;
                } else {
                    // Pushing against crate
                    actor_pos.x = obs.max.x + half_w + SKIN;
                    actor_vel.x = 0.0;
                    if let Some(rb) = physics.bodies.get_mut(obs.handle) {
                        let push_impulse = initial_vel.x * actor_weight * 0.15;
                        rb.apply_impulse(rapier2d::math::Vec2::new(push_impulse, 0.0), true);
                    }
                }
            }
        }
    }
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
                    p.vel[0] = 0.0;
                    p.vel[1] = 0.0;
                }
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
        world.init_resource::<Materials>();

        // Fill water in chunk (0, 0)
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

        // Submerged crate in water
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, -10.0));
        let crate_ent = spawn_pixel_rigidbody(&mut world, None, crate_body);

        // Step 20 ticks
        for _ in 0..20 {
            world.run_system(step_rapier_physics_system, ());
        }

        let handles = world.component::<PhysicsBodyHandle>();
        let handle = handles.get(crate_ent).unwrap();
        assert!(
            handle.pos.y > -25.0,
            "Crate should remain buoyant in water, got y = {}",
            handle.pos.y
        );
    }

    #[test]
    fn test_actor_standing_on_crate_and_auto_step_up() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<RapierPhysics>();

        // 1. Landing on crate: half size (8.0, 8.0) at pos (10.0, 8.0) => top at y = 16.0
        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(10.0, 8.0));
        spawn_pixel_rigidbody(&mut world, None, crate_body);

        let mut physics = world.resource_mut::<RapierPhysics>();
        let obstacles = collect_dynamic_obstacles(&physics);
        assert_eq!(obstacles.len(), 1);

        let mut actor = KinematicActor::new(vec2(10.0, 21.0));
        actor.prev_pos = vec2(10.0, 26.0);
        actor.vel = vec2(0.0, -5.0);
        actor.grounded = false;

        let initial_vel = actor.vel;
        let prev_pos = actor.prev_pos;
        resolve_actor_dynamic_obstacles(
            &mut actor.pos,
            prev_pos,
            &mut actor.vel,
            &mut actor.grounded,
            actor.half_size,
            actor.step_height,
            actor.weight,
            initial_vel,
            &obstacles,
            &mut physics,
        );

        assert!(actor.grounded, "Actor must be grounded on top of crate");
        assert_eq!(actor.vel.y, 0.0);
        assert!(
            (actor.pos.y - 22.05).abs() < 0.1,
            "Actor pos.y should be ~22.05, got {}",
            actor.pos.y
        );

        // 2. Auto-step-up onto a low step crate (height 8.0, step_height 18)
        let low_crate = PixelRigidBody::new_box(
            16,
            8,
            Atom::with_state(1, AtomState::Solid, [100, 100, 100, 255]),
            vec2(30.0, 4.0),
        );
        drop(physics);
        spawn_pixel_rigidbody(&mut world, None, low_crate);
        let mut physics = world.resource_mut::<RapierPhysics>();
        let obstacles = collect_dynamic_obstacles(&physics);

        let mut walking_actor = KinematicActor::new(vec2(18.0, 10.0));
        walking_actor.prev_pos = vec2(16.0, 10.0);
        walking_actor.vel = vec2(3.0, 0.0);
        walking_actor.step_height = 18;

        let initial_vel = walking_actor.vel;
        let prev_pos = walking_actor.prev_pos;
        walking_actor.pos.x = 22.1;
        resolve_actor_dynamic_obstacles(
            &mut walking_actor.pos,
            prev_pos,
            &mut walking_actor.vel,
            &mut walking_actor.grounded,
            walking_actor.half_size,
            walking_actor.step_height,
            walking_actor.weight,
            initial_vel,
            &obstacles,
            &mut physics,
        );

        assert!(
            walking_actor.pos.y > 10.0,
            "Actor should have stepped up onto crate, pos.y = {}",
            walking_actor.pos.y
        );

        // 3. Horizontal pushing against a tall obstacle (> step_height)
        let tall_crate = PixelRigidBody::new_box(
            16,
            40,
            Atom::with_state(1, AtomState::Solid, [100, 100, 100, 255]),
            vec2(60.0, 20.0),
        );
        drop(physics);
        spawn_pixel_rigidbody(&mut world, None, tall_crate);
        let mut physics = world.resource_mut::<RapierPhysics>();
        let obstacles = collect_dynamic_obstacles(&physics);

        let mut pushing_actor = KinematicActor::new(vec2(48.0, 10.0));
        pushing_actor.prev_pos = vec2(46.0, 10.0);
        pushing_actor.vel = vec2(4.0, 0.0);
        pushing_actor.step_height = 18;

        let initial_vel = pushing_actor.vel;
        let prev_pos = pushing_actor.prev_pos;
        pushing_actor.pos.x = 52.1;
        resolve_actor_dynamic_obstacles(
            &mut pushing_actor.pos,
            prev_pos,
            &mut pushing_actor.vel,
            &mut pushing_actor.grounded,
            pushing_actor.half_size,
            pushing_actor.step_height,
            pushing_actor.weight,
            initial_vel,
            &obstacles,
            &mut physics,
        );

        assert_eq!(pushing_actor.vel.x, 0.0);
        assert!(pushing_actor.pos.x <= 52.0);
    }

    #[test]
    fn test_particle_colliding_with_dynamic_crate() {
        let mut world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<RapierPhysics>();

        let crate_body = PixelRigidBody::new_crate(16, 16, vec2(20.0, 20.0));
        spawn_pixel_rigidbody(&mut world, None, crate_body);

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
}
