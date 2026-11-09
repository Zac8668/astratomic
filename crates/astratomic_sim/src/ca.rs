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
            || (atom2.is_object() && atom.is_liquid())
    } else {
        false
    }
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

/// Updates a powder atom and appends awakened 3x3 chunk group positions into `awakened`.
pub fn update_powder(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    dt: u8,
    inertial_resistance: f32,
    awakened: &mut Vec<IVec2>,
) {
    let mut cur_pos = pos;

    let mut speed = get_speed(chunks, cur_pos);
    if speed < ATOM_TERM_VEL {
        speed += GRAVITY;
        set_speed(chunks, cur_pos, speed);
    }

    for _ in 0..speed {
        let neigh = down_neigh(chunks, cur_pos, &[(3, 0.2)], dt);
        let mut swapped = false;
        if get_moving(chunks, cur_pos) {
            set_moving(chunks, cur_pos, inertial_resistance);
        }
        for neigh in neigh {
            if neigh.0 {
                set_moving(chunks, cur_pos, inertial_resistance);
                swap(chunks, cur_pos, cur_pos + neigh.1, dt);
                push_unique(awakened, cur_pos);
                cur_pos += neigh.1;
                push_unique(awakened, cur_pos);
                swapped = true;
                break;
            }
        }

        if !swapped {
            let vel = Vec2::new(0.0, -(speed as f32));
            let angle = (fastrand::f32() - 0.5) * PI;
            set_vel(
                chunks,
                cur_pos,
                Vec2::from_angle(angle)
                    .rotate(vel * 0.3 * (1.0 - inertial_resistance))
                    .as_ivec2(),
            );
            break;
        }
    }
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
    fastrand::shuffle(&mut xs[..x_len]);

    let mut ys = [0_i32; CHUNK_LENGHT];
    let y_len = (max_y - min_y + 1) as usize;
    for (i, v) in (min_y..=max_y).enumerate() {
        ys[i] = v;
    }
    fastrand::shuffle(&mut ys[..y_len]);

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

            let vector = if speed.0 == 0 && speed.1 >= 0 {
                match state {
                    AtomState::Powder => {
                        update_powder(
                            chunks,
                            pos,
                            dt,
                            materials[id].inertial_resistance,
                            &mut awakened,
                        );
                    }
                    AtomState::Liquid => {
                        update_liquid(chunks, pos, materials[id].flow, dt, &mut awakened);
                    }
                    _ => {}
                }
                false
            } else {
                update_atom(chunks, pos, dt, &mut awakened);
                true
            };

            let atom = chunks.group.center_atom_mut(x as usize, y as usize);
            let mut self_only_awakened = false;
            if awakened.contains(&pos) {
                atom.set_f_idle(0);
                atom.set_moving(true);
            } else if vector {
                atom.set_f_idle(0);
                push_unique(&mut awakened, pos);
                atom.set_moving(false);
            } else if awake_self {
                push_unique(&mut awakened, pos);
                self_only_awakened = true;
                atom.set_moving(false);
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
    let dt = chunk_world.dt;
    let manager_pos = chunk_world.pos;

    let DirtyRects {
        current: dirty_rects,
        new: new_dirty_rects,
        render: render_dirty_rects,
    } = dirty_rects_resource;

    new_dirty_rects.clear();

    let mut x_passes = [0_i32, 1];
    let mut y_passes = [0_i32, 1];
    if fastrand::bool() {
        x_passes.swap(0, 1);
    }
    if fastrand::bool() {
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
                (dt, materials),
            );
        }
    }

    dirty_rects_resource.swap();
}
