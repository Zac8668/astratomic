use bones_framework::prelude::{IVec2, UVec2};
use noise::{RidgedMulti, SuperSimplex};

use crate::consts::CHUNK_LENGHT;

#[derive(Clone, Default)]
pub struct Generator(pub RidgedMulti<SuperSimplex>, pub f64, pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
}

pub trait D1 {
    fn d1(&self) -> usize;
}

impl D1 for IVec2 {
    fn d1(&self) -> usize {
        (self.y * CHUNK_LENGHT as i32 + self.x) as usize
    }
}

impl D1 for UVec2 {
    fn d1(&self) -> usize {
        (self.y * CHUNK_LENGHT as u32 + self.x) as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
        atom.x += self.chunk.x * CHUNK_LENGHT as i32;
        atom.y += self.chunk.y * CHUNK_LENGHT as i32;
        atom
    }
}

#[derive(Debug, Default)]
pub struct DeferredDirtyRectUpdate {
    pub chunk_pos: ChunkPos,
    pub awake_surrouding: bool,
}
