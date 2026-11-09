use std::{collections::HashMap, sync::Arc};

use bones_framework::prelude::*;
use itertools::Itertools;
use noise::{RidgedMulti, SuperSimplex};

use crate::{
    atom::{Atom, Materials},
    chunk::{CHUNK_SIZE, Chunk, ChunkGrid},
    consts::{LOAD_HEIGHT, LOAD_WIDTH},
    world_store::{IoResponse, WorldIoWorker},
};

#[derive(HasSchema, Clone, Default)]
pub struct Generator(pub RidgedMulti<SuperSimplex>, pub f64, pub u32);

#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct URect {
    pub min: UVec2,
    pub max: UVec2,
}

impl URect {
    pub const fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        Self {
            min: UVec2::new(x0, y0),
            max: UVec2::new(x1, y1),
        }
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.max.x.saturating_sub(self.min.x)
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.max.y.saturating_sub(self.min.y)
    }
}

pub trait D1 {
    fn d1(&self) -> usize;
}

impl D1 for IVec2 {
    fn d1(&self) -> usize {
        (self.y * CHUNK_SIZE as i32 + self.x) as usize
    }
}

impl D1 for UVec2 {
    fn d1(&self) -> usize {
        (self.y * CHUNK_SIZE as u32 + self.x) as usize
    }
}

#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ChunkPos {
    pub atom: UVec2,
    pub chunk: IVec2,
}

impl ChunkPos {
    pub fn new(atom: UVec2, chunk: IVec2) -> Self {
        Self { atom, chunk }
    }

    pub fn to_global(self) -> IVec2 {
        let mut atom = self.atom.as_ivec2();
        atom.x += self.chunk.x * CHUNK_SIZE as i32;
        atom.y += self.chunk.y * CHUNK_SIZE as i32;
        atom
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DeferredDirtyRectUpdate {
    pub chunk_pos: ChunkPos,
    pub awake_surrouding: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveDir {
    X(i32),
    Y(i32),
}

/// Active world chunk container for `bones_ecs`.
///
/// Each [`Chunk`] stores its atom grid as an `Arc<ChunkGrid>` so `bones_ecs` rollback snapshots
/// only clone the `Arc` pointer, while mutations use `Arc::make_mut(&mut chunk.grid)` for
/// Copy-on-Write semantics.
#[derive(HasSchema, Clone, Default)]
pub struct ChunkWorld {
    pub chunks: HashMap<IVec2, Chunk>,
    pub pos: IVec2,
    pub dt: u8,
}

pub type ChunkManager = ChunkWorld;

impl ChunkWorld {
    pub fn get_chunk(&self, index: IVec2) -> Option<&Chunk> {
        self.chunks.get(&index)
    }

    pub fn get_chunk_mut(&mut self, index: IVec2) -> Option<&mut Chunk> {
        self.chunks.get_mut(&index)
    }

    pub fn get_atom(&self, pos: &ChunkPos) -> Option<&Atom> {
        let chunk = self.chunks.get(&pos.chunk)?;
        chunk.grid.atoms.get(pos.atom.d1())
    }

    /// Obtain a mutable reference to an atom at `pos`, performing Copy-on-Write on the chunk's
    /// `Arc<ChunkGrid>` via `Arc::make_mut(&mut chunk.grid)`.
    pub fn get_mut_atom(&mut self, pos: ChunkPos) -> Option<&mut Atom> {
        let chunk = self.chunks.get_mut(&pos.chunk)?;
        let grid = Arc::make_mut(&mut chunk.grid);
        grid.atoms.get_mut(pos.atom.d1())
    }

    /// Set the atom at `pos` using Copy-on-Write (`Arc::make_mut(&mut chunk.grid)`).
    pub fn set_atom(&mut self, pos: ChunkPos, atom: Atom) -> bool {
        let Some(slot) = self.get_mut_atom(pos) else {
            return false;
        };
        *slot = atom;
        true
    }

    /// Swap two atoms across potentially different chunks using `Arc::make_mut(&mut chunk.grid)`.
    pub fn swap_atoms(&mut self, pos1: ChunkPos, pos2: ChunkPos, dt: u8) -> bool {
        let (Some(&a1), Some(&a2)) = (self.get_atom(&pos1), self.get_atom(&pos2)) else {
            return false;
        };

        let mut next1 = if a2.is_object() { Atom::default() } else { a2 };
        let mut next2 = if a1.is_object() { Atom::default() } else { a1 };
        next1.set_updated_at(dt);
        next2.set_updated_at(dt);

        if let Some(chunk1) = self.chunks.get_mut(&pos1.chunk) {
            Arc::make_mut(&mut chunk1.grid).atoms[pos1.atom.d1()] = next1;
        }
        if let Some(chunk2) = self.chunks.get_mut(&pos2.chunk) {
            Arc::make_mut(&mut chunk2.grid).atoms[pos2.atom.d1()] = next2;
        }
        true
    }

    pub fn add_chunk(&mut self, index: IVec2, chunk: Chunk) {
        self.chunks.insert(index, chunk);
    }

    pub fn insert_chunk_grid(&mut self, index: IVec2, grid: Arc<ChunkGrid>) {
        self.chunks.insert(index, Chunk::from_grid(grid));
    }

    pub fn remove_chunk(&mut self, index: IVec2) -> Option<Chunk> {
        self.chunks.remove(&index)
    }

    /// Request loading all chunks in the active window `[pos.x .. pos.x + LOAD_WIDTH] x [pos.y .. pos.y + LOAD_HEIGHT]`
    /// through the non-blocking background I/O worker.
    pub fn request_initial_window(&self, io_worker: &mut WorldIoWorker) {
        for x in self.pos.x..(self.pos.x + LOAD_WIDTH) {
            for y in self.pos.y..(self.pos.y + LOAD_HEIGHT) {
                if !self.chunks.contains_key(&ivec2(x, y)) {
                    io_worker.request_load_chunk(x, y);
                }
            }
        }
    }

    /// Shift the active chunk window by 1 step in `move_dir`, batch-saving outgoing chunks via
    /// `IoRequest::SaveBatch` and requesting incoming chunks via `IoRequest::LoadChunk`.
    pub fn move_window(&mut self, move_dir: MoveDir, io_worker: &mut WorldIoWorker) {
        let (for_load, vec_load, dir, is_y) = match move_dir {
            MoveDir::X(dir) => {
                self.pos.x += dir;
                (LOAD_HEIGHT, LOAD_WIDTH, dir, false)
            }
            MoveDir::Y(dir) => {
                self.pos.y += dir;
                (LOAD_WIDTH, LOAD_HEIGHT, dir, true)
            }
        };

        let mut save_batch = Vec::with_capacity(for_load as usize);

        for i in 0..for_load {
            // Unload & queue far chunk for batch saving
            let mut unload_off = ivec2(if dir == -1 { vec_load } else { -1 }, i);
            if is_y {
                unload_off = unload_off.yx();
            }
            let unload_pos = self.pos + unload_off;
            if let Some(removed) = self.chunks.remove(&unload_pos) {
                save_batch.push(((unload_pos.x, unload_pos.y), removed.grid));
            }

            // Queue non-blocking load for entering chunk
            let mut load_off = ivec2(if dir == 1 { vec_load - 1 } else { 0 }, i);
            if is_y {
                load_off = load_off.yx();
            }
            let load_pos = self.pos + load_off;
            if !self.chunks.contains_key(&load_pos) {
                io_worker.request_load_chunk(load_pos.x, load_pos.y);
            }
        }

        io_worker.save_batch(save_batch);
    }

    /// Queue all currently active chunks to be saved as a single batch in the background I/O worker.
    pub fn save_all_chunks(&self, io_worker: &WorldIoWorker) {
        let mut batch: Vec<((i32, i32), Arc<ChunkGrid>)> = self
            .chunks
            .iter()
            .map(|(pos, chunk)| ((pos.x, pos.y), Arc::clone(&chunk.grid)))
            .collect();
        batch.sort_by_key(|(coords, _)| *coords);
        io_worker.save_batch(batch);
    }
}

impl std::ops::Index<ChunkPos> for ChunkWorld {
    type Output = Atom;
    #[track_caller]
    fn index(&self, pos: ChunkPos) -> &Self::Output {
        self.get_atom(&pos).expect("Invalid index position.")
    }
}

impl std::ops::IndexMut<ChunkPos> for ChunkWorld {
    #[track_caller]
    fn index_mut(&mut self, pos: ChunkPos) -> &mut Self::Output {
        self.get_mut_atom(pos).expect("Invalid index position.")
    }
}

#[derive(HasSchema, Clone, Default)]
pub struct DirtyRects {
    /// The current chunk update dirty rects.
    pub current: HashMap<IVec2, URect>,
    /// The new chunk update dirty rects.
    pub new: HashMap<IVec2, URect>,
    /// The dirty render rects.
    pub render: HashMap<IVec2, URect>,
}

impl DirtyRects {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.new);
    }
}

/// Transforms a global world atom coordinate to a [`ChunkPos`].
pub fn global_to_chunk(mut pos: IVec2) -> ChunkPos {
    if pos.x < 0 {
        pos.x -= CHUNK_SIZE as i32;
    }
    if pos.y < 0 {
        pos.y -= CHUNK_SIZE as i32;
    }

    let (mut chunk_x, mut chunk_y) = (pos.x / CHUNK_SIZE as i32, pos.y / CHUNK_SIZE as i32);

    let (x_off, y_off) = (
        (pos.x % CHUNK_SIZE as i32 + CHUNK_SIZE as i32) % CHUNK_SIZE as i32,
        (pos.y % CHUNK_SIZE as i32 + CHUNK_SIZE as i32) % CHUNK_SIZE as i32,
    );

    if pos.x < 0 && x_off == 0 {
        chunk_x += 1;
    }
    if pos.y < 0 && y_off == 0 {
        chunk_y += 1;
    }

    let (x, y) = (
        if pos.x >= 0 {
            pos.x as u32 % CHUNK_SIZE as u32
        } else {
            x_off as u32
        },
        if pos.y >= 0 {
            pos.y as u32 % CHUNK_SIZE as u32
        } else {
            y_off as u32
        },
    );

    ChunkPos::new(uvec2(x, y), ivec2(chunk_x, chunk_y))
}

pub fn extend_rect_if_needed(rect: &mut URect, pos: &UVec2) {
    let max_coord = (CHUNK_SIZE - 1) as u32;
    if pos.x < rect.min.x {
        rect.min.x = pos.x.clamp(0, max_coord);
    } else if pos.x > rect.max.x {
        rect.max.x = pos.x.clamp(0, max_coord);
    }

    if pos.y < rect.min.y {
        rect.min.y = pos.y.clamp(0, max_coord);
    } else if pos.y > rect.max.y {
        rect.max.y = pos.y.clamp(0, max_coord);
    }
}

pub fn update_dirty_rects(dirty_rects: &mut HashMap<IVec2, URect>, pos: ChunkPos) {
    if let Some(rect) = dirty_rects.get_mut(&pos.chunk) {
        extend_rect_if_needed(rect, &pos.atom);
    } else {
        dirty_rects.insert(
            pos.chunk,
            URect::new(pos.atom.x, pos.atom.y, pos.atom.x, pos.atom.y),
        );
    }
}

pub fn update_dirty_rects_3x3(dirty_rects: &mut HashMap<IVec2, URect>, pos: ChunkPos) {
    let atom = pos.atom;
    let mut chunk = pos.chunk;
    let last = (CHUNK_SIZE - 1) as u32;

    if (1..(last - 1)).contains(&atom.x) && (1..(last - 1)).contains(&atom.y) {
        if let Some(rect) = dirty_rects.get_mut(&chunk) {
            extend_rect_if_needed(rect, &(atom + UVec2::ONE));
            extend_rect_if_needed(rect, &(atom - UVec2::ONE));
        } else {
            dirty_rects.insert(
                chunk,
                URect::new(atom.x - 1, atom.y - 1, atom.x + 1, atom.y + 1),
            );
        }
    } else if (atom.x == 0 || atom.x == last) && (1..(last - 1)).contains(&atom.y) {
        if let Some(rect) = dirty_rects.get_mut(&chunk) {
            extend_rect_if_needed(
                rect,
                &(atom - if atom.x == 0 { UVec2::Y } else { UVec2::ONE }),
            );
            extend_rect_if_needed(
                rect,
                &(atom + if atom.x == 0 { UVec2::ONE } else { UVec2::Y }),
            );
        } else {
            dirty_rects.insert(
                chunk,
                URect::new(
                    atom.x - if atom.x == 0 { 0 } else { 1 },
                    atom.y - 1,
                    atom.x + if atom.x == 0 { 1 } else { 0 },
                    atom.y + 1,
                ),
            );
        }

        let x = if atom.x == 0 { last } else { 0 };
        if atom.x == 0 {
            chunk.x -= 1;
        } else {
            chunk.x += 1;
        }
        if let Some(rect) = dirty_rects.get_mut(&chunk) {
            extend_rect_if_needed(rect, &uvec2(x, atom.y + 1));
            extend_rect_if_needed(rect, &uvec2(x, atom.y - 1));
        } else {
            dirty_rects.insert(chunk, URect::new(x, atom.y - 1, x, atom.y + 1));
        }
    } else if (atom.y == 0 || atom.y == last) && (1..(last - 1)).contains(&atom.x) {
        if let Some(rect) = dirty_rects.get_mut(&chunk) {
            extend_rect_if_needed(
                rect,
                &(atom - if atom.y == 0 { UVec2::X } else { UVec2::ONE }),
            );
            extend_rect_if_needed(
                rect,
                &(atom + if atom.y == 0 { UVec2::ONE } else { UVec2::X }),
            );
        } else {
            dirty_rects.insert(
                chunk,
                URect::new(
                    atom.x - 1,
                    atom.y - if atom.y == 0 { 0 } else { 1 },
                    atom.x + 1,
                    atom.y + if atom.y == 0 { 1 } else { 0 },
                ),
            );
        }

        let y = if atom.y == 0 { last } else { 0 };
        if atom.y == 0 {
            chunk.y -= 1;
        } else {
            chunk.y += 1;
        }
        if let Some(rect) = dirty_rects.get_mut(&chunk) {
            extend_rect_if_needed(rect, &uvec2(atom.x + 1, y));
            extend_rect_if_needed(rect, &uvec2(atom.x - 1, y));
        } else {
            dirty_rects.insert(chunk, URect::new(atom.x - 1, y, atom.x + 1, y));
        }
    } else {
        for (x, y) in (-1..=1).cartesian_product(-1..=1) {
            let mut global = pos.to_global();
            global += ivec2(x, y);
            let pos = global_to_chunk(global);

            if let Some(rect) = dirty_rects.get_mut(&pos.chunk) {
                extend_rect_if_needed(rect, &pos.atom);
            } else {
                dirty_rects.insert(
                    pos.chunk,
                    URect::new(pos.atom.x, pos.atom.y, pos.atom.x, pos.atom.y),
                );
            }
        }
    }
}

/// Simulation system that polls completed [`IoResponse::ChunkLoaded`] messages from the
/// deterministic queue in [`WorldIoWorker`] and integrates them into [`ChunkWorld`].
pub fn poll_chunk_io_system(
    mut io_worker: ResMut<WorldIoWorker>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    materials: Option<Res<Materials>>,
    generator: Option<Res<Generator>>,
) {
    let full_rect = URect::new(0, 0, (CHUNK_SIZE - 1) as u32, (CHUNK_SIZE - 1) as u32);

    while let Some(response) = io_worker.poll_response() {
        match response {
            IoResponse::ChunkLoaded((cx, cy), loaded_grid) => {
                let pos = ivec2(cx, cy);
                let chunk = match loaded_grid {
                    Some(grid) => Chunk::from_grid(grid),
                    None => match (materials.as_deref(), generator.as_deref()) {
                        (Some(mats), Some(generator)) if !mats.0.is_empty() => {
                            Chunk::new(Handle::default(), pos, mats, generator)
                        }
                        _ => Chunk::default(),
                    },
                };

                chunk_world.add_chunk(pos, chunk);
                dirty_rects.current.insert(pos, full_rect);
                dirty_rects.render.insert(pos, full_rect);
            }
        }
    }
}
