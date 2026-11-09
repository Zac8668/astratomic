use core::slice;

use crate::prelude::*;
use async_channel::Sender;
use itertools::Itertools;

pub type ChunkCorners<'a> = [Option<[&'a mut Atom; QUARTER_CHUNK_LEN]>; 4];
pub type ChunkSides<'a> = [Option<[&'a mut Atom; HALF_CHUNK_LEN]>; 4];

pub struct ChunkGroup<'a> {
    pub center: &'a mut [Atom],
    pub corners: ChunkCorners<'a>,
    pub sides: ChunkSides<'a>,
    /// Position of the center chunk.
    pub center_pos: IVec2,
}

impl ChunkGroup<'_> {
    pub fn group_to_chunk(center_pos: IVec2, group_idx: i32) -> IVec2 {
        let x_diff = group_idx % 3 - 1;
        let y_diff = group_idx / 3 - 1;

        let mut pos = center_pos;
        pos.x += x_diff;
        pos.y += y_diff;

        pos
    }

    #[inline]
    pub fn surrounding_idx(idx: i32) -> i32 {
        match idx {
            0 | 1 => 0,
            2 | 3 => 1,
            6 | 5 => 2,
            8 | 7 => 3,
            _ => panic!("Invalid surrounding idx."),
        }
    }

    pub fn get_local(&self, idx: (IVec2, i32)) -> Option<&Atom> {
        let mut pos = idx.0;
        match idx.1 {
            // Center
            4 => Some(&self.center[idx.0.d1()]),
            // Corners
            0 | 2 | 6 | 8 => {
                // Offset position
                if idx.1 == 0 || idx.1 == 6 {
                    pos.x -= HALF_CHUNK_LENGHT as i32;
                }
                if idx.1 == 0 || idx.1 == 2 {
                    pos.y -= HALF_CHUNK_LENGHT as i32;
                }

                let index = pos.y * HALF_CHUNK_LENGHT as i32 + pos.x;
                let corner_index = ChunkGroup::surrounding_idx(idx.1) as usize;

                if let Some(corners) = &self.corners[corner_index] {
                    Some(corners[index as usize])
                } else {
                    None
                }
            }
            // Sides
            1 | 3 | 5 | 7 => {
                // Offset position
                if idx.1 == 1 {
                    pos.y -= HALF_CHUNK_LENGHT as i32
                } else if idx.1 == 3 {
                    pos.x -= HALF_CHUNK_LENGHT as i32
                }

                // Gets width accordingly if side is a top/down or left/right
                let width = if idx.1 == 3 || idx.1 == 5 {
                    HALF_CHUNK_LENGHT
                } else {
                    CHUNK_LENGHT
                } as i32;

                let index = pos.y * width + pos.x;
                let side_index = ChunkGroup::surrounding_idx(idx.1) as usize;

                if let Some(sides) = &self.sides[side_index] {
                    Some(sides[index as usize])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn get_mut_local(&mut self, idx: (IVec2, i32)) -> Option<&mut Atom> {
        let mut pos = idx.0;
        match idx.1 {
            // Center
            4 => Some(&mut self.center[idx.0.d1()]),
            // Corners
            0 | 2 | 6 | 8 => {
                // Offset position
                if idx.1 == 0 || idx.1 == 6 {
                    pos.x -= HALF_CHUNK_LENGHT as i32;
                }
                if idx.1 == 0 || idx.1 == 2 {
                    pos.y -= HALF_CHUNK_LENGHT as i32;
                }

                let index = pos.y * HALF_CHUNK_LENGHT as i32 + pos.x;
                let corner_index = ChunkGroup::surrounding_idx(idx.1) as usize;

                if let Some(corners) = &mut self.corners[corner_index] {
                    Some(corners[index as usize])
                } else {
                    None
                }
            }
            // Sides
            1 | 3 | 5 | 7 => {
                // Offset position
                if idx.1 == 1 {
                    pos.y -= HALF_CHUNK_LENGHT as i32
                } else if idx.1 == 3 {
                    pos.x -= HALF_CHUNK_LENGHT as i32
                }

                // Gets width accordingly if side is a top/down or left/right
                let width = if idx.1 == 3 || idx.1 == 5 {
                    HALF_CHUNK_LENGHT
                } else {
                    CHUNK_LENGHT
                } as i32;

                let index = pos.y * width + pos.x;
                let side_index = ChunkGroup::surrounding_idx(idx.1) as usize;

                if let Some(sides) = &mut self.sides[side_index] {
                    Some(sides[index as usize])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn get_global(&self, idx: IVec2) -> Option<&Atom> {
        let local_idx = global_to_local(idx);
        self.get_local(local_idx)
    }

    pub fn get_mut_global(&mut self, idx: IVec2) -> Option<&mut Atom> {
        let local_idx = global_to_local(idx);
        self.get_mut_local(local_idx)
    }
}

// Two ways to index a ChunkGroup to get an atom, by a global(IVec2) and by a local(IVec2, i32) pos
// Use get_* if you want an Option instead
impl std::ops::Index<IVec2> for ChunkGroup<'_> {
    type Output = Atom;
    #[track_caller]
    fn index(&self, idx: IVec2) -> &Self::Output {
        self.get_global(idx).expect("Invalid index position.")
    }
}
impl std::ops::IndexMut<IVec2> for ChunkGroup<'_> {
    #[track_caller]
    fn index_mut(&mut self, idx: IVec2) -> &mut Self::Output {
        self.get_mut_global(idx).expect("Invalid index position.")
    }
}

impl std::ops::Index<(IVec2, i32)> for ChunkGroup<'_> {
    type Output = Atom;
    #[track_caller]
    fn index(&self, idx: (IVec2, i32)) -> &Self::Output {
        self.get_local(idx).expect("Invalid index position.")
    }
}
impl std::ops::IndexMut<(IVec2, i32)> for ChunkGroup<'_> {
    #[track_caller]
    fn index_mut(&mut self, idx: (IVec2, i32)) -> &mut Self::Output {
        self.get_mut_local(idx).expect("Invalid index position.")
    }
}

/// Transforms global 3x3 chunk position to local 3x3 chunks position
/// Used for chunk multithreaded updates
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
/// Used for chunk multithreaded updates
pub fn local_to_global(pos: (IVec2, i32)) -> IVec2 {
    let range = 0..CHUNK_LENGHT as i32;
    if !range.contains(&pos.0.x) || !range.contains(&pos.0.y) || !(0..9).contains(&pos.1) {
        panic!(
            "Invalid position on local_to_global. {} {} {}",
            pos.0.x, pos.0.y, pos.1
        )
    }

    let chunk_size = CHUNK_LENGHT as i32;

    let chunk_index = pos.1;

    let chunk_x = chunk_index % 3;
    let chunk_y = chunk_index / 3;

    let global_x = pos.0.x + chunk_size * chunk_x;
    let global_y = pos.0.y + chunk_size * chunk_y;

    IVec2::new(global_x, global_y)
}

//This function gets and updates the chunk groups parallely
pub fn update_chunk_groups<'a>(
    chunks: &'a mut HashMap<IVec2, Chunk>,
    (x_toff, y_toff): (i32, i32),
    dirty_rects: &'a HashMap<IVec2, URect>,
    manager_pos: IVec2,
    senders: (
        &'a Sender<DeferredDirtyRectUpdate>,
        &'a Sender<DeferredDirtyRectUpdate>,
    ),
    update: (u8, &'a Materials),
    scope: &Scope<'a, '_, ()>,
) {
    puffin::profile_function!();

    let (dirty_update_rect_send, dirty_render_rect_send) = senders;
    let (dt, materials) = update;

    for chunk_pos in dirty_rects.keys() {
        //Get chunks mutable reference from it's pointer
        let chunks_ptr: *mut HashMap<IVec2, Chunk> = chunks;
        let chunks;
        unsafe {
            chunks = chunks_ptr.as_mut().unwrap();
        }

        scope.spawn(async move {
            //If not a center chunk in our current update step, or we don't have the chunk, continue
            let same_x = (chunk_pos.x + x_toff + manager_pos.x.abs() % 2) % 2 == 0;
            let same_y = (chunk_pos.y + y_toff + manager_pos.y.abs() % 2) % 2 == 0;

            if !same_x || !same_y || !chunks.contains_key(chunk_pos) {
                return;
            }

            //Get center and create group
            let first = chunks.get_mut(chunk_pos).unwrap().atoms.as_mut_ptr();
            let center;
            unsafe {
                center = slice::from_raw_parts_mut(first, CHUNK_LEN);
            }

            let mut chunk_group = ChunkGroup {
                center,
                sides: [None, None, None, None],
                corners: [None, None, None, None],
                center_pos: *chunk_pos,
            };

            //Get chunk surroundings
            for (x_off, y_off) in (-1..=1).cartesian_product(-1..=1) {
                let off = ivec2(x_off, y_off);

                match (x_off, y_off) {
                    // UP and DOWN
                    (0, -1) | (0, 1) => {
                        let Some(chunk) = chunks.get_mut(&(*chunk_pos + off)) else {
                            continue;
                        };

                        let mut start_ptr = chunk.atoms.as_mut_ptr();
                        if y_off == -1 {
                            unsafe {
                                start_ptr = start_ptr.add(HALF_CHUNK_LEN);
                            }
                        }

                        let mut atoms = vec![];
                        for i in 0..HALF_CHUNK_LEN {
                            unsafe {
                                atoms.push(start_ptr.add(i).as_mut().unwrap());
                            }
                        }

                        chunk_group.sides[if y_off == -1 { 0 } else { 3 }] =
                            Some(atoms.try_into().unwrap());
                    }
                    //LEFT and RIGHT
                    (-1, 0) | (1, 0) => {
                        let Some(chunk) = chunks.get_mut(&(*chunk_pos + off)) else {
                            continue;
                        };

                        let start_ptr = chunk.atoms.as_mut_ptr();

                        let mut atoms = vec![];
                        let mut add_off = match x_off {
                            -1 => HALF_CHUNK_LENGHT,
                            1 => 0,
                            _ => unreachable!(),
                        };

                        for i in 0..HALF_CHUNK_LEN {
                            if i % HALF_CHUNK_LENGHT == 0 && i != 0 {
                                add_off += HALF_CHUNK_LENGHT;
                            }

                            unsafe {
                                atoms.push(start_ptr.add(i + add_off).as_mut().unwrap());
                            }
                        }

                        chunk_group.sides[if x_off == -1 { 1 } else { 2 }] =
                            Some(atoms.try_into().unwrap());
                    }
                    //CORNERS
                    (-1, -1) | (1, -1) | (-1, 1) | (1, 1) => {
                        let Some(chunk) = chunks.get_mut(&(*chunk_pos + off)) else {
                            continue;
                        };

                        let start_ptr = chunk.atoms.as_mut_ptr();

                        let mut atoms = vec![];
                        let mut add_off = match (x_off, y_off) {
                            (1, 1) => 0,
                            (-1, 1) => HALF_CHUNK_LENGHT,
                            (1, -1) => HALF_CHUNK_LEN,
                            (-1, -1) => HALF_CHUNK_LEN + HALF_CHUNK_LENGHT,

                            _ => unreachable!(),
                        };

                        for i in 0..QUARTER_CHUNK_LEN {
                            if i % HALF_CHUNK_LENGHT == 0 && i != 0 {
                                add_off += HALF_CHUNK_LENGHT;
                            }

                            unsafe {
                                atoms.push(start_ptr.add(i + add_off).as_mut().unwrap());
                            }
                        }

                        let corner_idx = match (x_off, y_off) {
                            (1, 1) => 3,
                            (-1, 1) => 2,
                            (1, -1) => 1,
                            (-1, -1) => 0,

                            _ => unreachable!(),
                        };

                        chunk_group.corners[corner_idx] = Some(atoms.try_into().unwrap());
                    }
                    // CENTER
                    (0, 0) => { /*We alredy got the center*/ }
                    _ => unreachable!(),
                }
            }

            let rect = dirty_rects.get(&chunk_group.center_pos).unwrap();
            update_chunks(
                &mut UpdateChunksType {
                    group: &mut chunk_group,
                    dirty_update_rect_send,
                    dirty_render_rect_send,
                    materials,
                },
                dt,
                rect,
            )
        });
    }
}

/*#[test]
fn update_test() {
    //Get Chunk Manager
    let mut chunk_manager = ChunkManager::default();

    //Get materials
    let materials = &Materials(vec![
        Material::Void,
        Material::Object,
        Material::Powder {
            inertial_resistance: 0.1,
        },
        Material::Liquid { flow: 5 },
        Material::Powder {
            inertial_resistance: 0.92,
        },
        Material::Liquid { flow: 1 },
        Material::Solid,
        Material::Solid,
        Material::Solid,
    ]);

    //Add some chunks with full dirty rects
    let mut dirty_rects = HashMap::new();
    for (x, y) in (chunk_manager.pos.x..chunk_manager.pos.x + 1)
        .cartesian_product(chunk_manager.pos.y..chunk_manager.pos.y + 1)
    {
        let index = ivec2(x, y);
        chunk_manager
            .chunks
            .insert(index, Chunk::new(Handle::default(), index));
        dirty_rects.insert(index, URect::new(0, 0, 63, 63));
    }

    let manager_pos = ivec2(chunk_manager.pos.x, chunk_manager.pos.y);
    let dt = 0;

    //Update
    // Create channel for sending dirty update rects
    let (dirty_update_rects_send, _) = async_channel::unbounded::<DeferredDirtyRectUpdate>();
    let dirty_update_rect_send = &dirty_update_rects_send;

    // Create channel for sending dirty render rect updates
    let (dirty_render_rects_send, _) = async_channel::unbounded::<DeferredDirtyRectUpdate>();
    let dirty_render_rect_send = &dirty_render_rects_send;

    for (y_toff, x_toff) in rand_range(0..2)
        .into_iter()
        .cartesian_product(rand_range(0..2).into_iter())
    {
        update_chunk_groups(
            &mut chunk_manager.chunks,
            (x_toff, y_toff),
            &dirty_rects,
            manager_pos,
            (dirty_update_rect_send, dirty_render_rect_send),
            (dt, materials),
        );
    }
}*/
