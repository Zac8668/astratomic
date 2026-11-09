//! 4-pass deterministic cellular automata simulation step.

use std::{collections::HashMap, f32::consts::PI, ops::Range};

use bones_framework::prelude::{IVec2, Vec2, ivec2};

use crate::{
    atom::{Atom, AtomState, Material, Materials},
    chunk_group::{ChunkGroup, global_to_local, local_to_global, update_chunk_groups},
    consts::{ATOM_TERM_VEL, CHUNK_LENGHT, FRAMES_SLEEP, GRAVITY, HALF_CHUNK_LENGHT},
    geom_tools::Line,
    world::{ChunkPos, ChunkWorld, DirtyRects, URect, update_dirty_rects, update_dirty_rects_3x3},
};

pub struct UpdateChunksType<'a, 'b> {
    pub group: &'b mut ChunkGroup<'a>,
    pub dirty_update_rects: &'b mut HashMap<IVec2, URect>,
    pub dirty_render_rects: &'b mut HashMap<IVec2, URect>,
    pub materials: &'b Materials,
    pub tick: u64,
    pub world_seed: u32,
}

/// Deterministic 64-bit coordinate hash combining world seed, simulation tick, and cell coordinates.
#[inline]
pub fn coord_seed(world_seed: u32, tick: u64, x: i32, y: i32) -> u64 {
    let mut h = (world_seed as u64).wrapping_mul(0x517cc1b727220a95);
    h = h.wrapping_add(tick.wrapping_mul(0x9e3779b97f4a7c15));
    h ^= (x as u32 as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    h = h.rotate_left(27);
    h ^= (y as u32 as u64).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h
}

#[inline]
fn in_group_bounds(pos: IVec2) -> bool {
    const MIN_B: i32 = HALF_CHUNK_LENGHT as i32;
    const MAX_B: i32 = (CHUNK_LENGHT * 2 + HALF_CHUNK_LENGHT) as i32;
    pos.x >= MIN_B && pos.x < MAX_B && pos.y >= MIN_B && pos.y < MAX_B
}

#[inline]
fn push_unique(vec: &mut Vec<IVec2>, pos: IVec2) {
    if !vec.contains(&pos) {
        vec.push(pos);
    }
}

/// Swap two atoms from global 3x3 chunk group positions.
pub fn swap(chunks: &mut UpdateChunksType, pos1: IVec2, pos2: IVec2, dt: u8) {
    if !in_group_bounds(pos1) || !in_group_bounds(pos2) {
        return;
    }
    let (Some(&a0), Some(&a1)) = (chunks.group.get_global(pos1), chunks.group.get_global(pos2))
    else {
        return;
    };

    let mut next1 = if a1.is_object() { Atom::default() } else { a1 };
    let mut next2 = if a0.is_object() { Atom::default() } else { a0 };
    next1.set_updated_at(dt);
    next2.set_updated_at(dt);

    chunks.group[pos1] = next1;
    chunks.group[pos2] = next2;

    let local1 = global_to_local(pos1);
    let local2 = global_to_local(pos2);

    for (pos, idx) in [local1, local2] {
        let chunk = ChunkGroup::group_to_chunk(chunks.group.center_pos, idx);
        update_dirty_rects(
            chunks.dirty_render_rects,
            ChunkPos::new(pos.as_uvec2(), chunk),
        );
    }
}

/// Check if `pos` is swappable (void, or a swappable material that has not yet been updated this step).
#[inline]
pub fn swapable(
    chunks: &UpdateChunksType,
    pos: IVec2,
    ids: &[(u8, f32)],
    atom: Atom,
    dt: u8,
) -> bool {
    if let Some(atom2) = chunks.group.get_global(pos) {
        atom2.is_void()
            || (!ids.is_empty()
                && atom2.updated_at() != dt
                && ids
                    .iter()
                    .any(|&(id, prob)| id == atom2.material_id && fastrand::f32() < prob))
            || (atom2.is_object() && atom.is_fluid())
    } else {
        false
    }
}

/// Check if an atom can be displaced by falling/sliding powder (air or lower density fluid).
#[inline]
pub fn can_displace(target: Option<&Atom>, _dt: u8) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.is_void() || target.is_fluid()
}

/// Gets downward neighbors from a 3x3 chunk group position.
#[inline]
pub fn down_neigh(
    chunks: &UpdateChunksType,
    pos: IVec2,
    ids: &[(u8, f32)],
    dt: u8,
) -> [(bool, IVec2); 3] {
    let mut neigh = [(false, IVec2::ZERO); 3];

    let material = get_atom(chunks, pos);
    let to_check: &[i32] = if material.is_powder() && !get_moving(chunks, pos) {
        &[0]
    } else {
        &[0, -1, 1]
    };

    for (slot, &x) in neigh.iter_mut().zip(to_check.iter()) {
        slot.0 = swapable(chunks, pos + IVec2::new(x, 1), ids, material, dt);
        slot.1 = IVec2::new(x, 1);
    }

    if fastrand::bool() {
        neigh.swap(1, 2);
    }

    neigh
}

/// Gets horizontal neighbors from a 3x3 chunk group position.
#[inline]
pub fn side_neigh(
    chunks: &UpdateChunksType,
    pos: IVec2,
    ids: &[(u8, f32)],
    dt: u8,
) -> [(bool, IVec2); 2] {
    let mut neigh = [(false, IVec2::ZERO); 2];

    let atom = get_atom(chunks, pos);
    for (slot, x) in neigh.iter_mut().zip([-1, 1]) {
        slot.0 = swapable(chunks, pos + IVec2::new(x, 0), ids, atom, dt);
        slot.1 = IVec2::new(x, 0);
    }

    if fastrand::bool() {
        neigh.swap(0, 1);
    }

    neigh
}

#[inline]
pub fn get_speed(chunks: &UpdateChunksType, pos: IVec2) -> u8 {
    chunks.group[pos].vy.max(0) as u8
}

#[inline]
pub fn set_speed(chunks: &mut UpdateChunksType, pos: IVec2, speed: u8) {
    chunks.group[pos].vy = speed.min(i8::MAX as u8) as i8;
}

#[inline]
pub fn get_vel(chunks: &UpdateChunksType, pos: IVec2) -> IVec2 {
    let atom = chunks.group[pos];
    ivec2(atom.vx as i32, atom.vy as i32)
}

#[inline]
pub fn set_vel(chunks: &mut UpdateChunksType, pos: IVec2, vel: IVec2) {
    let atom = &mut chunks.group[pos];
    atom.vx = vel.x.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    atom.vy = vel.y.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
}

#[inline]
pub fn _get_material(chunks: &UpdateChunksType, pos: IVec2) -> Material {
    chunks.materials.0[chunks.group[pos].material_id as usize]
}

#[inline]
pub fn get_atom(chunks: &UpdateChunksType, pos: IVec2) -> Atom {
    chunks.group[pos]
}

#[inline]
pub fn set_moving(chunks: &mut UpdateChunksType, pos: IVec2, inertial_resistance: f32) {
    for x_off in [-1, 1] {
        if fastrand::f32() > inertial_resistance {
            if let Some(neigh) = chunks.group.get_mut_global(pos + ivec2(x_off, 0)) {
                neigh.set_moving(true);
            }
        } else {
            break;
        }
    }
}

#[inline]
pub fn get_moving(chunks: &UpdateChunksType, pos: IVec2) -> bool {
    chunks.group[pos].moving()
}

/// Checks if an atom is eligible to update in the current `dt` step.
#[inline]
pub fn dt_updatable(chunks: &UpdateChunksType, pos: IVec2, dt: u8) -> bool {
    if let Some(atom) = chunks.group.get_global(pos) {
        atom.updated_at() != dt || atom.is_void()
    } else {
        false
    }
}

/// Returns a randomly shuffled `Vec<i32>` covering `range`.
pub fn rand_range(range: Range<i32>) -> Vec<i32> {
    let mut vec: Vec<i32> = range.collect();
    fastrand::shuffle(&mut vec);
    vec
}

/// Inspects immediate supporting neighbors of `(x, y)`: `(x - 1, y)`, `(x + 1, y)`, `(x, y - 1)`,
/// `(x - 1, y - 1)`, and `(x + 1, y - 1)`.
/// Settled powder neighbors roll `rng.f32() > neighbor_mat.inertial_resistance` to wake up.
pub fn try_wake_neighbors(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    _dt: u8,
    awakened: &mut Vec<IVec2>,
    rng: &mut fastrand::Rng,
) {
    let neighbor_offsets = [
        ivec2(-1, 0),
        ivec2(1, 0),
        ivec2(0, -1),
        ivec2(-1, -1),
        ivec2(1, -1),
    ];

    for offset in neighbor_offsets {
        let n_pos = pos + offset;
        if !in_group_bounds(n_pos) {
            continue;
        }

        let Some(neighbor) = chunks.group.get_mut_global(n_pos) else {
            continue;
        };

        if neighbor.is_powder() && !neighbor.is_free_falling() {
            let neighbor_mat = &chunks.materials[neighbor.material_id];
            if rng.f32() > neighbor_mat.inertial_resistance {
                neighbor.set_free_falling(true);
                neighbor.set_f_idle(0);

                let local = global_to_local(n_pos);
                let chunk = ChunkGroup::group_to_chunk(chunks.group.center_pos, local.1);
                let chunk_pos = ChunkPos::new(local.0.as_uvec2(), chunk);
                update_dirty_rects_3x3(chunks.dirty_update_rects, chunk_pos);
                update_dirty_rects(chunks.dirty_render_rects, chunk_pos);
                push_unique(awakened, n_pos);
            }
        }
    }
}

/// Updates a powder atom using Noita-style inertial resistance and wakes neighbors.
pub fn update_powder(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    dt: u8,
    awakened: &mut Vec<IVec2>,
    rng: &mut fastrand::Rng,
) {
    let mut atom = chunks.group[pos];
    let mat = &chunks.materials[atom.material_id];
    let down_y = pos.y + 1;
    let down_pos = ivec2(pos.x, down_y);

    // 1. Check Straight Down First (Always Active, Even When Settled):
    // Check if (x, down_y) is Air or a liquid/gas with lower density than this powder.
    if can_displace(chunks.group.get_global(down_pos), dt) {
        atom.set_free_falling(true);
        atom.vy = (atom.vy.saturating_add(1)).min(6);
        atom.set_updated_at(dt);
        chunks.group[pos] = atom;
        swap(chunks, pos, down_pos, dt);
        push_unique(awakened, pos);
        push_unique(awakened, down_pos);
        try_wake_neighbors(chunks, pos, dt, awakened, rng);
        return;
    }

    // 2. Impact Momentum Conversion (When Straight Down Is Blocked):
    // If straight down (x, down_y) is blocked by a solid/dense atom and atom.vy > 1:
    if atom.vy > 1 {
        let splash = atom.vy;
        atom.vx = if rng.bool() { splash } else { -splash };
        atom.vy = 0;
        atom.set_free_falling(true);
    } else {
        atom.vy = 0;
    }

    // 3. Gate Diagonal & Horizontal Sliding Behind atom.is_free_falling():
    // CRITICAL: If !atom.is_free_falling(), STOP HERE and return! A settled atom with
    // solid ground directly underneath it must NEVER roll random diagonal slides every frame
    // (which would cause gravel piles to slowly melt flat over time).
    if !atom.is_free_falling() {
        chunks.group[pos] = atom;
        return;
    }

    // Horizontal Momentum Hop (atom.vx != 0):
    if atom.vx != 0 {
        let step_x = atom.vx.signum() as i32;
        let mut decay = 1;
        if rng.f32() < mat.inertial_resistance {
            decay += 1;
        }
        if atom.vx > 0 {
            atom.vx = (atom.vx - decay).max(0);
        } else {
            atom.vx = (atom.vx + decay).min(0);
        }

        let hop_down = ivec2(pos.x + step_x, down_y);
        let hop_side = ivec2(pos.x + step_x, pos.y);

        let target_hop = if can_displace(chunks.group.get_global(hop_down), dt) {
            Some(hop_down)
        } else if can_displace(chunks.group.get_global(hop_side), dt) {
            Some(hop_side)
        } else {
            None
        };

        if let Some(target) = target_hop {
            atom.set_updated_at(dt);
            chunks.group[pos] = atom;
            swap(chunks, pos, target, dt);
            push_unique(awakened, pos);
            push_unique(awakened, target);
            try_wake_neighbors(chunks, pos, dt, awakened, rng);
            return;
        } else {
            atom.vx = 0;
        }
    }

    // Diagonal Slope Slide ((x ± 1, down_y)):
    // Slope Stopping Friction (Why Gravel Forms Steeper Piles than Sand):
    // Before sliding diagonally, if the atom has already been sliding (atom.vy == 0),
    // roll if rng.f32() < (mat.inertial_resistance * 0.45).
    // If true, the atom catches on the rough slope: set free_falling(false), vx = 0, and return.
    if atom.vy == 0 && rng.f32() < (mat.inertial_resistance * 0.45) {
        atom.set_free_falling(false);
        atom.vx = 0;
        chunks.group[pos] = atom;
        return;
    }

    // Otherwise, pick primary/secondary diagonal directions (dx1, dx2)
    let (dx1, dx2) = if rng.bool() { (-1, 1) } else { (1, -1) };
    let diag1 = ivec2(pos.x + dx1, down_y);
    let diag2 = ivec2(pos.x + dx2, down_y);

    let diag_target = if can_displace(chunks.group.get_global(diag1), dt) {
        Some(diag1)
    } else if can_displace(chunks.group.get_global(diag2), dt) {
        Some(diag2)
    } else {
        None
    };

    if let Some(target) = diag_target {
        atom.set_free_falling(true);
        atom.set_updated_at(dt);
        chunks.group[pos] = atom;
        swap(chunks, pos, target, dt);
        push_unique(awakened, pos);
        push_unique(awakened, target);
        try_wake_neighbors(chunks, pos, dt, awakened, rng);
        return;
    }

    // 4. Settle to Sleep When Blocked:
    // If the atom could not move straight down, horizontally, or diagonally,
    // set atom.set_free_falling(false), atom.vx = 0, atom.vy = 0, and write it back to (x, y).
    atom.set_free_falling(false);
    atom.vx = 0;
    atom.vy = 0;
    chunks.group[pos] = atom;
}

/// Updates a liquid atom and appends awakened 3x3 chunk group positions into `awakened`.
pub fn update_liquid(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    flow: u8,
    dt: u8,
    awakened: &mut Vec<IVec2>,
) {
    let mut cur_pos = pos;

    let mut speed = get_speed(chunks, pos);
    if speed < ATOM_TERM_VEL {
        speed += GRAVITY;
        set_speed(chunks, pos, speed);
    }

    let mut swapped = false;
    for _ in 0..speed {
        let neigh = down_neigh(chunks, cur_pos, &[], dt);
        for neigh in neigh {
            if neigh.0 {
                swap(chunks, cur_pos, cur_pos + neigh.1, dt);
                push_unique(awakened, cur_pos);
                cur_pos += neigh.1;
                push_unique(awakened, cur_pos);
                swapped = true;
                break;
            }
        }
    }

    if !swapped {
        set_speed(chunks, cur_pos, 0);
        let neigh = side_neigh(chunks, cur_pos, &[], dt);
        let side = if neigh[0].0 {
            Some(neigh[0].1.x)
        } else if neigh[1].0 {
            Some(neigh[1].1.x)
        } else {
            None
        };

        if let Some(side) = side {
            for _ in 0..flow {
                let next_pos = cur_pos + IVec2::new(side, 0);
                if !in_group_bounds(next_pos) {
                    break;
                }
                let atom = get_atom(chunks, cur_pos);
                if !swapable(chunks, next_pos, &[], atom, dt) {
                    break;
                }

                swap(chunks, cur_pos, next_pos, dt);
                push_unique(awakened, cur_pos);
                cur_pos = next_pos;
                push_unique(awakened, cur_pos);
            }
        }
    }
}

/// Updates an atom with a 2D vector velocity (`vx != 0` or `vy < 0`).
pub fn update_atom(chunks: &mut UpdateChunksType, pos: IVec2, dt: u8, awakened: &mut Vec<IVec2>) {
    let mut cur_pos = pos;

    let mut vel = get_vel(chunks, cur_pos);
    if vel.y < ATOM_TERM_VEL as i32 {
        vel += GRAVITY as i32 * IVec2::Y;
        set_vel(chunks, cur_pos, vel);
    }

    for next_pos in Line::new(cur_pos, vel) {
        if !in_group_bounds(next_pos) || chunks.group.get_global(next_pos).is_none() {
            set_vel(chunks, cur_pos, IVec2::ZERO);
            break;
        }

        push_unique(awakened, cur_pos);
        let atom = get_atom(chunks, cur_pos);
        if swapable(chunks, next_pos, &[], atom, dt) {
            swap(chunks, cur_pos, next_pos, dt);
            cur_pos = next_pos;
            push_unique(awakened, cur_pos);
        } else if get_atom(chunks, next_pos).is_liquid() && get_atom(chunks, cur_pos).is_liquid() {
            push_unique(awakened, next_pos);
            set_vel(chunks, next_pos, vel * 4 / 5);
            set_vel(chunks, cur_pos, vel / 5);
            break;
        } else {
            if vel.abs().x > 4 && vel.abs().y > 4 {
                set_vel(
                    chunks,
                    cur_pos,
                    (Vec2::from_angle(PI).rotate(vel.as_vec2()) * 0.5).as_ivec2(),
                );
            } else if !swapable(chunks, cur_pos + IVec2::Y, &[], atom, dt) {
                set_vel(chunks, cur_pos, IVec2::ZERO);
            }
            break;
        }
    }
}

pub fn update_chunks(chunks: &mut UpdateChunksType, dt: u8, dirty_rect: &URect) {
    puffin::profile_function!();

    let max_coord = (CHUNK_LENGHT - 1) as u32;
    let min_x = dirty_rect.min.x.min(max_coord) as i32;
    let max_x = dirty_rect.max.x.min(max_coord) as i32;
    let min_y = dirty_rect.min.y.min(max_coord) as i32;
    let max_y = dirty_rect.max.y.min(max_coord) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }

    let mut xs = [0_i32; CHUNK_LENGHT];
    let x_len = (max_x - min_x + 1) as usize;
    for (i, v) in (min_x..=max_x).enumerate() {
        xs[i] = v;
    }
    let chunk_seed = coord_seed(
        chunks.world_seed,
        chunks.tick,
        chunks.group.center_pos.x,
        chunks.group.center_pos.y,
    );
    let mut chunk_rng = fastrand::Rng::with_seed(chunk_seed);
    chunk_rng.shuffle(&mut xs[..x_len]);

    let mut ys = [0_i32; CHUNK_LENGHT];
    let y_len = (max_y - min_y + 1) as usize;
    for (i, v) in (min_y..=max_y).enumerate() {
        ys[i] = v;
    }
    chunk_rng.shuffle(&mut ys[..y_len]);

    let materials = chunks.materials;
    let mut awakened = Vec::with_capacity(32);

    for &y in &ys[..y_len] {
        for &x in &xs[..x_len] {
            let mut awake_self = false;
            let id;
            let speed;
            let state;
            {
                let atom = chunks.group.center_atom_mut(x as usize, y as usize);
                if atom.is_void() {
                    continue;
                }
                if atom.updated_at() == dt {
                    continue;
                }
                atom.set_updated_at(0);

                speed = (atom.vx, atom.vy);
                state = atom.state();
                if matches!(state, AtomState::Solid | AtomState::Object)
                    && speed.0 == 0
                    && speed.1 == 0
                {
                    continue;
                }

                id = atom.material_id;
                let f_idle = atom.f_idle();
                if f_idle < FRAMES_SLEEP {
                    atom.set_f_idle(f_idle + 1);
                    awake_self = true;
                }
            }

            let local_pos = (ivec2(x, y), 4);
            let pos = local_to_global(local_pos);
            awakened.clear();

            let world_atom_x = chunks.group.center_pos.x * CHUNK_LENGHT as i32 + x;
            let world_atom_y = chunks.group.center_pos.y * CHUNK_LENGHT as i32 + y;
            let atom_seed = coord_seed(chunks.world_seed, chunks.tick, world_atom_x, world_atom_y);
            let mut rng = fastrand::Rng::with_seed(atom_seed);

            let vector = match state {
                AtomState::Powder if speed.1 >= 0 => {
                    update_powder(chunks, pos, dt, &mut awakened, &mut rng);
                    false
                }
                AtomState::Liquid if speed.0 == 0 && speed.1 >= 0 => {
                    update_liquid(chunks, pos, materials[id].flow, dt, &mut awakened);
                    false
                }
                _ => {
                    if speed.0 != 0 || speed.1 != 0 {
                        update_atom(chunks, pos, dt, &mut awakened);
                        true
                    } else {
                        false
                    }
                }
            };

            let atom = chunks.group.center_atom_mut(x as usize, y as usize);
            let mut self_only_awakened = false;
            if awakened.contains(&pos) {
                atom.set_f_idle(0);
                if !atom.is_powder() {
                    atom.set_moving(true);
                }
            } else if vector {
                atom.set_f_idle(0);
                push_unique(&mut awakened, pos);
                atom.set_moving(false);
            } else if awake_self {
                if !atom.is_powder() || atom.is_free_falling() {
                    push_unique(&mut awakened, pos);
                    self_only_awakened = true;
                }
            }

            for &awoke in &awakened {
                let local = global_to_local(awoke);
                let chunk = ChunkGroup::group_to_chunk(chunks.group.center_pos, local.1);
                let chunk_pos = ChunkPos::new(local.0.as_uvec2(), chunk);
                let awake_surrounding = !(self_only_awakened && awoke == pos);
                if awake_surrounding {
                    update_dirty_rects_3x3(chunks.dirty_update_rects, chunk_pos);
                } else {
                    update_dirty_rects(chunks.dirty_update_rects, chunk_pos);
                }
            }
        }
    }
}

/// Runs one full 4-pass checkerboard cellular automata step across all dirty chunks in `chunk_world`.
pub fn step_chunk_simulation(
    chunk_world: &mut ChunkWorld,
    dirty_rects_resource: &mut DirtyRects,
    materials: &Materials,
) {
    puffin::profile_function!();

    if dirty_rects_resource.current.is_empty() || materials.0.is_empty() {
        return;
    }

    // Cycle `dt` through 1..=3 so `0` always means "not updated in the current frame".
    chunk_world.dt = (chunk_world.dt % 3) + 1;
    chunk_world.tick = chunk_world.tick.wrapping_add(1);
    let dt = chunk_world.dt;
    let tick = chunk_world.tick;
    let world_seed = chunk_world.seed;
    let manager_pos = chunk_world.pos;

    let DirtyRects {
        current: dirty_rects,
        new: new_dirty_rects,
        render: render_dirty_rects,
    } = dirty_rects_resource;

    new_dirty_rects.clear();

    let mut pass_rng = fastrand::Rng::with_seed(
        (world_seed as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ tick,
    );
    let mut x_passes = [0_i32, 1];
    let mut y_passes = [0_i32, 1];
    if pass_rng.bool() {
        x_passes.swap(0, 1);
    }
    if pass_rng.bool() {
        y_passes.swap(0, 1);
    }

    for y_toff in y_passes {
        for x_toff in x_passes {
            update_chunk_groups(
                &mut chunk_world.chunks,
                (x_toff, y_toff),
                dirty_rects,
                manager_pos,
                new_dirty_rects,
                render_dirty_rects,
                (dt, materials, tick, world_seed),
            );
        }
    }

    dirty_rects_resource.swap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chunk::Chunk,
        world::global_to_chunk,
    };

    fn test_world() -> (ChunkWorld, DirtyRects, Materials) {
        let materials = Materials::default_builtin();
        let mut chunk_world = ChunkWorld::default();
        let dirty_rects = DirtyRects::default();
        for cx in -1..=1 {
            for cy in -1..=1 {
                chunk_world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }
        (chunk_world, dirty_rects, materials)
    }

    #[test]
    fn newly_spawned_powder_atoms_start_free_falling() {
        let materials = Materials::default_builtin();
        let sand = Atom::new(2, &materials);
        let gravel = Atom::new(4, &materials);
        assert!(sand.is_free_falling(), "new sand must be free falling");
        assert!(gravel.is_free_falling(), "new gravel must be free falling");
        assert_eq!(materials[2].inertial_resistance, 0.10);
        assert_eq!(materials[4].inertial_resistance, 0.60);
    }

    #[test]
    fn settled_powders_never_slide_diagonally_on_their_own() {
        let (mut chunk_world, mut dirty_rects, materials) = test_world();

        // Solid rock floor at y = 20
        for x in 0..=30 {
            chunk_world.paint_at_global(ivec2(x, 20), 8, false, &materials, &mut dirty_rects);
        }

        // Place a 3-atom flat row of settled gravel and sand at y = 19
        for x in 10..=12 {
            let mut sand = Atom::new(2, &materials);
            sand.set_free_falling(false);
            sand.vx = 0;
            sand.vy = 0;
            chunk_world.set_atom(global_to_chunk(ivec2(x, 19)), sand);

            let mut gravel = Atom::new(4, &materials);
            gravel.set_free_falling(false);
            gravel.vx = 0;
            gravel.vy = 0;
            chunk_world.set_atom(global_to_chunk(ivec2(x + 10, 19)), gravel);
        }

        dirty_rects.current.insert(ivec2(0, 0), URect::full_chunk());

        // Step simulation for 20 frames
        for _ in 0..20 {
            step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
        }

        // Neither sand nor gravel should have slid diagonally or melted flat
        for x in 10..=12 {
            let a = chunk_world.get_atom(&global_to_chunk(ivec2(x, 19))).unwrap();
            assert_eq!(a.material_id, 2, "sand should remain in place at ({x}, 19)");
            assert!(!a.is_free_falling(), "settled sand must stay asleep");
        }
        for x in 20..=22 {
            let a = chunk_world.get_atom(&global_to_chunk(ivec2(x, 19))).unwrap();
            assert_eq!(a.material_id, 4, "gravel should remain in place at ({x}, 19)");
            assert!(!a.is_free_falling(), "settled gravel must stay asleep");
        }
        // Verify adjacent cells (9, 19) and (13, 19) are still void (no melting flat)
        assert!(chunk_world.get_atom(&global_to_chunk(ivec2(9, 19))).unwrap().is_void());
        assert!(chunk_world.get_atom(&global_to_chunk(ivec2(13, 19))).unwrap().is_void());
        assert!(chunk_world.get_atom(&global_to_chunk(ivec2(19, 19))).unwrap().is_void());
        assert!(chunk_world.get_atom(&global_to_chunk(ivec2(23, 19))).unwrap().is_void());
    }

    #[test]
    fn undermining_settled_powder_triggers_avalanche() {
        let (mut chunk_world, mut dirty_rects, materials) = test_world();

        // Solid rock floor with a single supporting block at (15, 20)
        chunk_world.paint_at_global(ivec2(15, 20), 8, false, &materials, &mut dirty_rects);

        // Place a vertical stack of 3 sand atoms on top of (15, 20): (15, 19), (15, 18), (15, 17)
        for y in 17..=19 {
            let mut sand = Atom::new(2, &materials);
            sand.set_free_falling(false);
            sand.vx = 0;
            sand.vy = 0;
            chunk_world.set_atom(global_to_chunk(ivec2(15, y)), sand);
        }

        // Floor lower down at y = 25 so collapsing sand has a catch floor
        for x in 10..=20 {
            chunk_world.paint_at_global(ivec2(x, 25), 8, false, &materials, &mut dirty_rects);
        }

        dirty_rects.current.insert(ivec2(0, 0), URect::full_chunk());

        // Step once to verify stable
        step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
        assert_eq!(chunk_world.get_atom(&global_to_chunk(ivec2(15, 19))).unwrap().material_id, 2);

        // Undermine: dig out the supporting rock block at (15, 20)
        chunk_world.set_atom(global_to_chunk(ivec2(15, 20)), Atom::default());
        update_dirty_rects_3x3(&mut dirty_rects.current, global_to_chunk(ivec2(15, 20)));

        // Step simulation for 10 frames
        for _ in 0..10 {
            step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
        }

        // The top sand atom must have collapsed downward
        assert!(
            chunk_world.get_atom(&global_to_chunk(ivec2(15, 17))).unwrap().is_void(),
            "top sand should have collapsed downward"
        );
        let fallen = (10..=20).any(|x| {
            (20..=25).any(|y| {
                chunk_world
                    .get_atom(&global_to_chunk(ivec2(x, y)))
                    .is_some_and(|a| a.material_id == 2)
            })
        });
        assert!(fallen, "undermined sand must collapse into avalanche");
    }

    #[test]
    fn slope_stopping_friction_causes_gravel_to_settle_on_slopes() {
        let (mut chunk_world, mut dirty_rects, materials) = test_world();

        // Build a diagonal slope from (10, 10) to (30, 30)
        for i in 10..=30 {
            chunk_world.paint_at_global(ivec2(i, i), 8, false, &materials, &mut dirty_rects);
            // Fill solid rock underneath
            for dy in 1..=5 {
                chunk_world.paint_at_global(ivec2(i, i + dy), 8, false, &materials, &mut dirty_rects);
            }
        }

        // Place gravel at (11, 10) directly on top of the slope at (11, 11)
        let gravel = Atom::new(4, &materials);
        chunk_world.set_atom(global_to_chunk(ivec2(11, 10)), gravel);
        dirty_rects.current.insert(ivec2(0, 0), URect::full_chunk());

        // Step simulation for 30 frames
        for _ in 0..30 {
            step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
        }

        // Gravel should catch on the rough slope and settle before sliding all the way to the bottom (30, 29)
        let gravel_pos = (10..=30).find_map(|x| {
            let a = chunk_world.get_atom(&global_to_chunk(ivec2(x, x - 1)))?;
            (a.material_id == 4).then_some(ivec2(x, x - 1))
        });
        assert!(
            gravel_pos.is_some(),
            "gravel must remain on the slope"
        );
        let pos = gravel_pos.unwrap();
        let atom = chunk_world.get_atom(&global_to_chunk(pos)).unwrap();
        assert!(
            !atom.is_free_falling(),
            "gravel must have caught and settled to sleep on the slope"
        );
        assert!(
            pos.x < 30,
            "gravel with 0.60 inertial resistance should catch and stop on the slope before reaching the end, stopped at {:?}",
            pos
        );
    }
}
