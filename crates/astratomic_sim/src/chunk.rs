use std::{collections::HashSet, sync::Arc};

use bones_schema::HasSchema;
use noise::*;

pub use crate::consts::CHUNK_SIZE;
use crate::{prelude::*, rigidbody};

/// Fixed 64x64 grid of 8-byte [`Atom`]s (32,768 bytes total).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ChunkGrid {
    pub atoms: [Atom; CHUNK_SIZE * CHUNK_SIZE],
}

const _: [(); 32_768] = [(); std::mem::size_of::<ChunkGrid>()];

impl Default for ChunkGrid {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl PartialEq for ChunkGrid {
    fn eq(&self, other: &Self) -> bool {
        self.atoms.as_slice() == other.atoms.as_slice()
    }
}

impl Eq for ChunkGrid {}

impl std::fmt::Debug for ChunkGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkGrid")
            .field("len", &self.atoms.len())
            .finish()
    }
}

impl ChunkGrid {
    /// Allocate a zeroed [`Box<ChunkGrid>`] directly on the heap without a 32 KiB stack temporary.
    pub fn new_zeroed_box() -> Box<Self> {
        let layout = std::alloc::Layout::new::<Self>();
        // SAFETY: `ChunkGrid` is `Zeroable` and 32,768 bytes (non-ZST), so zero-initialized
        // heap memory is a valid representation of `ChunkGrid`.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        }
    }

    /// Allocate a zeroed [`Arc<ChunkGrid>`].
    pub fn new_zeroed_arc() -> Arc<Self> {
        Arc::from(Self::new_zeroed_box())
    }

    #[inline]
    pub const fn index(x: usize, y: usize) -> usize {
        y * CHUNK_SIZE + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<&Atom> {
        if x < CHUNK_SIZE && y < CHUNK_SIZE {
            Some(&self.atoms[Self::index(x, y)])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Atom> {
        if x < CHUNK_SIZE && y < CHUNK_SIZE {
            Some(&mut self.atoms[Self::index(x, y)])
        } else {
            None
        }
    }
}

impl std::ops::Index<usize> for ChunkGrid {
    type Output = Atom;
    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        &self.atoms[index]
    }
}

impl std::ops::IndexMut<usize> for ChunkGrid {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.atoms[index]
    }
}

#[derive(Debug)]
pub struct ChunkAutoSave {
    pub coords: (i32, i32),
    pub handle: crate::world_store::WorldSaveHandle,
}

/// Active simulation chunk storing its grid behind an [`Arc<ChunkGrid>`] for O(1) rollback
/// snapshots and Copy-on-Write mutation via [`Arc::make_mut`].
#[derive(HasSchema, Debug)]
pub struct Chunk {
    pub grid: Arc<ChunkGrid>,
    pub texture: Handle<Image>,
    pub entity: Option<Entity>,
    pub auto_save: Option<ChunkAutoSave>,
}

impl Clone for Chunk {
    fn clone(&self) -> Self {
        Self {
            grid: Arc::clone(&self.grid),
            texture: self.texture,
            entity: self.entity,
            // Cloned rollback snapshots do not inherit the auto-save hook so dropping a snapshot
            // never overwrites active world state.
            auto_save: None,
        }
    }
}

impl PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid && self.texture == other.texture && self.entity == other.entity
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            grid: ChunkGrid::new_zeroed_arc(),
            texture: Handle::default(),
            entity: None,
            auto_save: None,
        }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        if let Some(auto_save) = self.auto_save.take() {
            auto_save
                .handle
                .save_chunk(auto_save.coords, Arc::clone(&self.grid));
        }
    }
}

#[derive(Clone, Copy)]
struct YCoord;
impl NoiseFn<f64, 2> for YCoord {
    fn get(&self, point: [f64; 2]) -> f64 {
        point[1]
    }
}

impl Chunk {
    pub fn from_grid(grid: Arc<ChunkGrid>) -> Self {
        Self {
            grid,
            texture: Handle::default(),
            entity: None,
            auto_save: None,
        }
    }

    /// Arm this chunk to automatically persist itself to `handle` at `coords` when dropped.
    pub fn arm_auto_save(
        &mut self,
        coords: (i32, i32),
        handle: crate::world_store::WorldSaveHandle,
    ) {
        self.auto_save = Some(ChunkAutoSave { coords, handle });
    }

    /// Disarm the per-chunk drop hook (used when batch-saving multiple chunks in a single
    /// transaction during window movement or world shutdown).
    pub fn disarm_auto_save(&mut self) {
        self.auto_save = None;
    }

    /// Obtain a mutable reference to the underlying [`ChunkGrid`] via Copy-on-Write (`Arc::make_mut`).
    #[inline]
    pub fn grid_mut(&mut self) -> &mut ChunkGrid {
        Arc::make_mut(&mut self.grid)
    }

    #[inline]
    pub fn atoms(&self) -> &[Atom; CHUNK_SIZE * CHUNK_SIZE] {
        &self.grid.atoms
    }

    /// Obtain a mutable reference to the underlying atom array via Copy-on-Write (`Arc::make_mut`).
    #[inline]
    pub fn atoms_mut(&mut self) -> &mut [Atom; CHUNK_SIZE * CHUNK_SIZE] {
        &mut Arc::make_mut(&mut self.grid).atoms
    }

    #[inline]
    pub fn get_atom(&self, index: usize) -> Option<&Atom> {
        self.grid.atoms.get(index)
    }

    /// Obtain a mutable reference to an atom by 1D index using Copy-on-Write (`Arc::make_mut`).
    #[inline]
    pub fn get_mut_atom(&mut self, index: usize) -> Option<&mut Atom> {
        Arc::make_mut(&mut self.grid).atoms.get_mut(index)
    }

    pub fn generate_grid(
        index: IVec2,
        materials: &Materials,
        generator: &Generator,
    ) -> Arc<ChunkGrid> {
        let mut grid = ChunkGrid::new_zeroed_box();

        let air_limit = -1000.;
        let scale = if generator.1 == 0.0 {
            512.0
        } else {
            generator.1
        };
        let cave = generator.0.clone();

        let cave: Multiply<f64, RidgedMulti<SuperSimplex>, Constant, 2> =
            Multiply::new(cave, Constant::new(0.7));

        let ground_noise = BasicMulti::<SuperSimplex>::new(generator.2);
        let ground = Add::new(
            ScalePoint::new(ground_noise)
                .set_y_scale(0.3)
                .set_x_scale(0.3),
            YCoord,
        );

        let cave_start = 0.5;
        let falloff = 0.3;
        let control = Add::new(ground.clone(), Constant::new(-cave_start));
        let generator = Select::new(cave.clone(), ground.clone(), control.clone())
            .set_bounds(air_limit, cave_start)
            .set_falloff(falloff);

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                let pos = [
                    (index.x as f64 * CHUNK_SIZE as f64 + x as f64) / scale,
                    (index.y as f64 * CHUNK_SIZE as f64 + y as f64) / scale,
                ];
                let ground_val = ground.get(pos);
                let control_val = ground_val - cave_start;
                let noise = if ((air_limit + falloff)..(cave_start - falloff))
                    .contains(&control_val)
                {
                    ground_val
                } else if control_val >= cave_start + falloff || control_val < air_limit - falloff {
                    cave.get(pos)
                } else {
                    generator.get(pos)
                };
                let i = y * CHUNK_SIZE + x;

                let id: u8 = if (air_limit..(cave_start - falloff)).contains(&control_val) {
                    // Ground layer
                    match (noise + 1.) / 2. {
                        ..0.2 => continue,
                        0.2..0.21 => 6,
                        0.21..0.5 => 7,
                        0.5..0.75 => 10,
                        0.75.. => 4,
                        _ => panic!("{noise}"),
                    }
                } else {
                    // Cave layer
                    match (noise + 1.) / 2. {
                        ..0.2 => continue,
                        0.2.. => 8,
                        _ => panic!("{noise}"),
                    }
                };

                grid.atoms[i] = Atom::new(id, materials);
            }
        }

        Arc::from(grid)
    }

    pub fn new(
        texture: Handle<Image>,
        index: IVec2,
        materials: &Materials,
        generator: &Generator,
    ) -> Chunk {
        Chunk {
            grid: Self::generate_grid(index, materials, generator),
            texture,
            entity: None,
            auto_save: None,
        }
    }

    pub fn new_image() -> Image {
        Image::Data(image::DynamicImage::ImageRgba8(image::RgbaImage::new(
            CHUNK_SIZE as u32,
            CHUNK_SIZE as u32,
        )))
    }

    // This uses the CPU and is not used in-game anymore
    pub fn _update_image_positions(&self, image: &mut Image, positions: &HashSet<IVec2>) {
        let Image::Data(dyn_img) = image else {
            return;
        };
        let Some(rgba) = dyn_img.as_mut_rgba8() else {
            return;
        };
        let data = rgba.as_mut();
        for pos in positions {
            let pixel_index = (pos.y as usize * CHUNK_SIZE + pos.x as usize) * 4;
            data[pixel_index..pixel_index + 4].copy_from_slice(&self.grid.atoms[pos.d1()].color);
        }
    }

    pub fn update_all(&self, image: &mut Image) {
        let Image::Data(dyn_img) = image else {
            return;
        };
        let Some(rgba) = dyn_img.as_mut_rgba8() else {
            return;
        };
        let data = rgba.as_mut();
        debug_assert_eq!(data.len(), CHUNK_SIZE * CHUNK_SIZE * 4);

        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let atom_index = y * CHUNK_SIZE + x;
                let pixel_index = atom_index * 4;
                data[pixel_index..pixel_index + 4]
                    .copy_from_slice(&self.grid.atoms[atom_index].color);
            }
        }
    }

    pub fn update_rect(&self, image: &mut Image, rect: &URect) {
        let Image::Data(dyn_img) = image else {
            return;
        };
        let Some(rgba) = dyn_img.as_mut_rgba8() else {
            return;
        };
        let data = rgba.as_mut();
        debug_assert_eq!(data.len(), CHUNK_SIZE * CHUNK_SIZE * 4);

        let max_idx = (CHUNK_SIZE - 1) as u32;
        let min_x = rect.min.x.min(max_idx) as usize;
        let max_x = rect.max.x.min(max_idx) as usize;
        let min_y = rect.min.y.min(max_idx) as usize;
        let max_y = rect.max.y.min(max_idx) as usize;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let atom_index = y * CHUNK_SIZE + x;
                let pixel_index = atom_index * 4;
                data[pixel_index..pixel_index + 4]
                    .copy_from_slice(&self.grid.atoms[atom_index].color);
            }
        }
    }

    pub fn get_collider(&self) -> Option<Collider> {
        rigidbody::get_collider(&self.get_values(), CHUNK_SIZE as u32, CHUNK_SIZE as u32)
    }

    pub fn get_values(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);

        for row in self.grid.atoms.chunks(CHUNK_SIZE) {
            for atom in row {
                if atom.is_solid() {
                    values.push(1.0);
                } else {
                    values.push(0.0);
                }
            }
        }

        values
    }
}
