use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::panic;
use std::sync::{Arc, Mutex};

use bevy::math::ivec2;
use rand::Rng;

use crate::prelude::*;

// Parallel reference for image and chunk data
pub type TexturesHash = HashMap<usize, HashSet<IVec2>>;
pub type ParTexturesHash = Arc<Mutex<TexturesHash>>;
pub type DirtyRectHash = HashMap<usize, HashSet<IVec2>>;
pub type ParDirtyRectHash = Arc<Mutex<DirtyRectHash>>;
pub type UpdateChunksType<'a> = (ChunkGroup<'a>, &'a ParTexturesHash, &'a ParDirtyRectHash);

/// Swap two atoms from global 3x3 chunks positions
pub fn swap(chunks: &mut UpdateChunksType, pos1: IVec2, pos2: IVec2, dt: f32) {
    let local1 = global_to_local(pos1);
    let local2 = global_to_local(pos2);

    let chunk_group = &mut chunks.0;
    {
        let temp = *chunk_group.get_local(local1).unwrap();
        chunk_group[local1] = chunk_group[local2];
        chunk_group[local2] = temp;

        chunk_group[local1].updated_at = dt;
        chunk_group[local2].updated_at = dt;
    }

    let mut hash = chunks.1.lock().unwrap();
    let local1_manager_idx = ChunkGroup::group_to_manager_idx(chunk_group.center_index, local1.1);
    if local1.1 == local2.1 {
        hash.entry(local1_manager_idx)
            .or_default()
            .extend([local1.0, local2.0]);
    } else {
        let local2_manager_idx =
            ChunkGroup::group_to_manager_idx(chunk_group.center_index, local2.1);
        hash.entry(local1_manager_idx).or_default().insert(local1.0);
        hash.entry(local2_manager_idx).or_default().insert(local2.0);
    }
}

/// Transforms global 3x3 chunk position to local 3x3 chunks position
pub fn global_to_local(pos: IVec2) -> (IVec2, i32) {
    let range = 0..CHUNK_LENGHT as i32 * 3;
    if !range.contains(&pos.x) || !range.contains(&pos.y) {
        panic!("Invalid position on global_to_local.")
    }

    let chunk_lenght = CHUNK_LENGHT as i32;

    let chunk_x = pos.x % (chunk_lenght * 3) / chunk_lenght;
    let chunk_y = pos.y / chunk_lenght;

    let local_x = pos.x - chunk_x * chunk_lenght;
    let local_y = pos.y - chunk_y * chunk_lenght;

    let chunk_index = chunk_y * 3 + chunk_x;

    (IVec2::new(local_x, local_y), chunk_index)
}

/// Transforms local 3x3 chunk position to global 3x3 chunks position
pub fn local_to_global(pos: (IVec2, i32)) -> IVec2 {
    let range = 0..CHUNK_LENGHT as i32;
    if !range.contains(&pos.0.x) || !range.contains(&pos.0.y) || !(0..9).contains(&pos.1) {
        panic!("Invalid position on local_to_global.")
    }

    let chunk_size = CHUNK_LENGHT as i32;

    let chunk_index = pos.1;

    let chunk_x = chunk_index % 3;
    let chunk_y = chunk_index / 3;

    let global_x = pos.0.x + chunk_size * chunk_x;
    let global_y = pos.0.y + chunk_size * chunk_y;

    IVec2::new(global_x, global_y)
}

/// See if position is swapable, that means it sees if the position is a void
/// or if it's a swapable state and has been not updated
pub fn swapable(chunks: &UpdateChunksType, pos: IVec2, states: &[(State, f32)], dt: f32) -> bool {
    if let Some(atom) = chunks.0.get_global(pos) {
        atom.state == State::Void
            || (states.iter().any(|&(state, prob)| {
                state == atom.state && rand::thread_rng().gen_range(0.0..1.0) < prob
            }) && atom.updated_at != dt)
    } else {
        false
    }
}

/// Gets down neighbours from a global pos
pub fn down_neigh(
    chunks: &UpdateChunksType,
    pos: IVec2,
    states: &[(State, f32)],
    dt: f32,
) -> [(bool, IVec2); 3] {
    let mut neigh = [(false, IVec2::ZERO); 3];

    for (neigh, x) in neigh.iter_mut().zip([0, -1, 1]) {
        neigh.0 = swapable(chunks, pos + IVec2::new(x, 1), states, dt);
        neigh.1 = IVec2::new(x, 1);
    }

    if rand::thread_rng().gen() {
        neigh.swap(1, 2)
    }

    neigh
}

/// Gets side neighbours from a global pos
pub fn side_neigh(
    chunks: &UpdateChunksType,
    pos: IVec2,
    states: &[(State, f32)],
    dt: f32,
) -> [(bool, IVec2); 2] {
    let mut neigh = [(false, IVec2::ZERO); 2];

    for (neigh, x) in neigh.iter_mut().zip([-1, 1]) {
        neigh.0 = swapable(chunks, pos + IVec2::new(x, 0), states, dt);
        neigh.1 = IVec2::new(x, 0);
    }

    if rand::thread_rng().gen() {
        neigh.swap(0, 1)
    }

    neigh
}

/// Gets velocity from a global pos
pub fn get_vel(chunks: &UpdateChunksType, pos: IVec2) -> Option<IVec2> {
    chunks.0[pos].velocity
}

/// Sets velocity from a global pos
pub fn set_vel(chunks: &mut UpdateChunksType, pos: IVec2, velocity: IVec2) {
    chunks.0[pos].velocity = if velocity == IVec2::ZERO {
        None
    } else {
        Some(velocity)
    }
}

/// Gets fall speed from a global pos
pub fn get_fspeed(chunks: &UpdateChunksType, pos: IVec2) -> u8 {
    chunks.0[pos].fall_speed
}

/// Sets fall speed from a global pos
pub fn set_fspeed(chunks: &mut UpdateChunksType, pos: IVec2, fall_speed: u8) {
    chunks.0[pos].fall_speed = fall_speed
}

/// Checks if atom is able to update this frame from a global pos
pub fn dt_updatable(chunks: &UpdateChunksType, pos: IVec2, dt: f32) -> bool {
    if let Some(atom) = chunks.0.get_global(pos) {
        atom.updated_at != dt || atom.state == State::Void
    } else {
        false
    }
}

pub fn extend_rect_if_needed(rect: &mut Rect, pos: &Vec2) {
    if pos.x < rect.min.x {
        rect.min.x = (pos.x).clamp(0., 63.)
    } else if pos.x > rect.max.x {
        rect.max.x = (pos.x).clamp(0., 63.)
    }

    if pos.y < rect.min.y {
        rect.min.y = (pos.y).clamp(0., 63.)
    } else if pos.y > rect.max.y {
        rect.max.y = (pos.y).clamp(0., 63.)
    }
}

// Shuflles range
pub fn rand_range(vec: Range<usize>) -> Vec<usize> {
    let mut vec: Vec<usize> = vec.collect();
    fastrand::shuffle(&mut vec);
    vec
}

// Transform pos to chunk coords
pub fn transform_to_chunk(pos: Vec2) -> Option<(IVec2, i32)> {
    if pos.x < 0. || pos.y < 0. {
        return None;
    }

    let (width, height) = (CHUNKS_WIDTH, CHUNKS_HEIGHT);

    let (chunk_x, chunk_y) = (
        (pos.x / (CHUNK_LENGHT * ATOM_SIZE) as f32) as usize,
        (pos.y / (CHUNK_LENGHT * ATOM_SIZE) as f32) as usize,
    );

    if chunk_x >= width || chunk_y >= height {
        return None;
    }

    let (atom_x, atom_y) = (
        ((pos.x / ATOM_SIZE as f32) % CHUNK_LENGHT as f32) as i32,
        ((pos.y / ATOM_SIZE as f32) % CHUNK_LENGHT as f32) as i32,
    );

    let local = (ivec2(atom_x, atom_y), (chunk_y * width + chunk_x) as i32);

    Some(local)
}

pub trait D1 {
    fn d1(&self) -> usize;
}

impl D1 for IVec2 {
    /// Transforms a IVec2 to a index for a chunk atoms vec
    fn d1(&self) -> usize {
        (self.y * CHUNK_LENGHT as i32 + self.x) as usize
    }
}

impl D1 for UVec2 {
    /// Transforms a UVec2 to a index for a chunk atoms vec
    fn d1(&self) -> usize {
        (self.y * CHUNK_LENGHT as u32 + self.x) as usize
    }
}

pub fn split_left_right(
    array: &mut [Atom],
) -> ([&mut Atom; CHUNK_LEN / 2], [&mut Atom; CHUNK_LEN / 2]) {
    let (left, right): (Vec<_>, Vec<_>) = array
        .chunks_mut(CHUNK_LENGHT)
        .flat_map(|chunk| {
            let (left, right) = chunk.split_at_mut(HALF_CHUNK_LENGHT);
            left.iter_mut().zip(right.iter_mut()).collect::<Vec<_>>()
        })
        .unzip();

    (left.try_into().unwrap(), right.try_into().unwrap())
}

pub fn updown_to_leftright(
    array: &mut [Atom],
) -> ([&mut Atom; CHUNK_LEN / 4], [&mut Atom; CHUNK_LEN / 4]) {
    let (left, right): (Vec<_>, Vec<_>) = array
        .chunks_mut(CHUNK_LENGHT)
        .flat_map(|chunk| {
            let (left, right) = chunk.split_at_mut(HALF_CHUNK_LENGHT);
            left.iter_mut().zip(right.iter_mut()).collect::<Vec<_>>()
        })
        .unzip();

    (left.try_into().unwrap(), right.try_into().unwrap())
}

#[derive(Default)]
pub struct MutableReferences<'a> {
    pub centers: Vec<Option<[&'a mut Atom; CHUNK_LEN]>>,
    pub sides: [Vec<Option<[&'a mut Atom; HALF_CHUNK_LEN]>>; 4],
    pub corners: [Vec<Option<[&'a mut Atom; QUARTER_CHUNK_LEN]>>; 4],
}

/// A deferred update message.
/// Indicates that an image or dirty rect should udpate.
#[derive(Debug)]
pub enum DeferredUpdate {
    UpdateImage {
        image_id: AssetId<Image>,
        pos: Vec2,
    },
    UpdateDirtyRect {
        chunk_idx: usize,
        pos: Vec2,
    },
}