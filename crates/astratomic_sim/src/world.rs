use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use bones_framework::prelude::*;
use itertools::Itertools;

use crate::{
    atom::{Atom, Materials},
    ca::step_chunk_simulation,
    chunk::{CHUNK_SIZE, Chunk, ChunkGrid},
    consts::{CHUNK_LENGHT, LOAD_HEIGHT, LOAD_WIDTH},
    prefabs::{
        PendingPrefabStamps, PrefabConfigs, PrefabPlaced, apply_pending_stamps_for_chunk,
        prefab_xs_for_chunk, stamp_prefab_at_x, stamp_prefabs_for_new_chunk,
    },
    world_store::{IoResponse, WorldIoWorker, WorldSaveHandle, WorldStorage},
};

pub use crate::worldgen::Generator;

/// Runtime simulation settings and camera focus point for chunk streaming and debug controls.
#[derive(HasSchema, Clone, Debug)]
#[repr(C)]
pub struct SimulationSettings {
    pub paused: bool,
    pub step_once: bool,
    pub steps_per_frame: u32,
    pub stream_chunks: bool,
    /// Camera or focus position in Bones world space (`+Y` up, `-Y` down).
    pub focus_world_pos: Vec2,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            paused: false,
            step_once: false,
            steps_per_frame: 1,
            stream_chunks: true,
            focus_world_pos: Vec2::new(0.0, 200.0),
        }
    }
}

/// Locate the workspace `assets/` directory whether running from the workspace root or a crate subdir.
pub fn resolve_assets_dir() -> PathBuf {
    let local = Path::new("assets");
    if local.exists() {
        return local.to_path_buf();
    }
    let workspace_assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    if workspace_assets.exists() {
        return workspace_assets;
    }
    local.to_path_buf()
}

/// Locate the persistent `saves/world.redb` database path outside `assets/` so the `bones_asset`
/// hot-reload watcher never attempts to parse `.redb` files as game assets.
pub fn resolve_world_db_path() -> PathBuf {
    if Path::new("assets").exists() {
        PathBuf::from(crate::world_store::WORLD_DB_PATH)
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(crate::world_store::WORLD_DB_PATH)
    }
}

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
    pub const fn full_chunk() -> Self {
        Self::new(0, 0, (CHUNK_SIZE - 1) as u32, (CHUNK_SIZE - 1) as u32)
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
#[derive(HasSchema, Default)]
pub struct ChunkWorld {
    pub chunks: HashMap<IVec2, Chunk>,
    pub pos: IVec2,
    pub dt: u8,
    pub save_handle: Option<WorldSaveHandle>,
}

impl Clone for ChunkWorld {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
            pos: self.pos,
            dt: self.dt,
            // Cloned rollback snapshots do not inherit `save_handle` so dropping a snapshot
            // does not overwrite active world state on disk.
            save_handle: None,
        }
    }
}

impl Drop for ChunkWorld {
    fn drop(&mut self) {
        let Some(handle) = self.save_handle.take() else {
            return;
        };
        if self.chunks.is_empty() {
            return;
        }
        let mut batch = Vec::with_capacity(self.chunks.len());
        for (pos, mut chunk) in self.chunks.drain() {
            chunk.disarm_auto_save();
            batch.push(((pos.x, pos.y), Arc::clone(&chunk.grid)));
        }
        handle.save_batch_and_flush(batch);
    }
}

pub type ChunkManager = ChunkWorld;

impl ChunkWorld {
    /// Attach a [`WorldSaveHandle`] so all chunks in [`ChunkWorld`] automatically save themselves
    /// when unloaded, dropped, or when [`ChunkWorld`] is dropped at game exit.
    pub fn attach_save_handle(&mut self, handle: WorldSaveHandle) {
        for (&pos, chunk) in self.chunks.iter_mut() {
            chunk.arm_auto_save((pos.x, pos.y), handle.clone());
        }
        self.save_handle = Some(handle);
    }

    pub fn is_in_window(&self, chunk_pos: IVec2) -> bool {
        (self.pos.x..(self.pos.x + LOAD_WIDTH)).contains(&chunk_pos.x)
            && (self.pos.y..(self.pos.y + LOAD_HEIGHT)).contains(&chunk_pos.y)
    }

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

    /// Paint a single global atom position and mark both update and render dirty rects.
    pub fn paint_at_global(
        &mut self,
        global_pos: IVec2,
        material_id: u8,
        only_air: bool,
        materials: &Materials,
        dirty_rects: &mut DirtyRects,
    ) -> bool {
        let chunk_pos = global_to_chunk(global_pos);
        let Some(existing) = self.get_atom(&chunk_pos) else {
            return false;
        };
        if only_air && material_id != 0 && !existing.is_void() {
            return false;
        }
        if material_id == 0 && existing.is_void() {
            return false;
        }

        let new_atom = Atom::new(material_id, materials);
        if !self.set_atom(chunk_pos, new_atom) {
            return false;
        }

        update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
        update_dirty_rects(&mut dirty_rects.render, chunk_pos);
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

    pub fn add_chunk(&mut self, index: IVec2, mut chunk: Chunk) {
        if let Some(ref handle) = self.save_handle {
            chunk.arm_auto_save((index.x, index.y), handle.clone());
        }
        if let Some(mut old) = self.chunks.insert(index, chunk) {
            old.disarm_auto_save();
        }
    }

    pub fn insert_chunk_grid(&mut self, index: IVec2, grid: Arc<ChunkGrid>) {
        self.add_chunk(index, Chunk::from_grid(grid));
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
            if let Some(mut removed) = self.chunks.remove(&unload_pos) {
                removed.disarm_auto_save();
                save_batch.push(((unload_pos.x, unload_pos.y), Arc::clone(&removed.grid)));
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

    /// Regenerate all chunks in the current active window procedurally in parallel and stamp prefabs.
    pub fn regenerate_window(
        &mut self,
        materials: &Materials,
        generator: &Generator,
        prefab_configs: &mut PrefabConfigs,
        prefab_placed: &mut PrefabPlaced,
        pending_stamps: &mut PendingPrefabStamps,
        dirty_rects: &mut DirtyRects,
    ) {
        prefab_placed.x_ranges.clear();
        pending_stamps.0.clear();

        let coords: Vec<IVec2> = (self.pos.x..(self.pos.x + LOAD_WIDTH))
            .cartesian_product(self.pos.y..(self.pos.y + LOAD_HEIGHT))
            .map(|(x, y)| ivec2(x, y))
            .collect();

        let generated = generate_chunks_parallel(&coords, materials, generator);
        for (pos, grid) in generated {
            if let Some(existing) = self.chunks.get_mut(&pos) {
                existing.grid = grid;
            } else {
                self.add_chunk(pos, Chunk::from_grid(grid));
            }
        }

        stamp_initial_window_prefabs(
            self,
            dirty_rects,
            prefab_configs,
            prefab_placed,
            pending_stamps,
            materials,
            generator.seed,
        );

        let full_rect = URect::full_chunk();
        for &pos in &coords {
            dirty_rects.current.insert(pos, full_rect);
            dirty_rects.render.insert(pos, full_rect);
        }

        if let Some(ref handle) = self.save_handle {
            let batch: Vec<((i32, i32), Arc<ChunkGrid>)> = self
                .chunks
                .iter()
                .map(|(pos, chunk)| ((pos.x, pos.y), Arc::clone(&chunk.grid)))
                .collect();
            handle.save_batch(batch);
        }
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

/// Converts a 2D Bones world coordinate (`+Y` up) into global atom coordinates (`+Y` down).
#[inline]
pub fn world_to_global_atom(world_pos: Vec2) -> IVec2 {
    ivec2(world_pos.x.floor() as i32, (-world_pos.y).floor() as i32)
}

/// Converts global atom coordinates (`+Y` down) into a 2D Bones world coordinate (`+Y` up).
#[inline]
pub fn global_atom_to_world(global_pos: IVec2) -> Vec2 {
    vec2(global_pos.x as f32, -(global_pos.y as f32))
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

fn generate_chunks_parallel(
    coords: &[IVec2],
    materials: &Materials,
    generator: &Generator,
) -> Vec<(IVec2, Arc<ChunkGrid>)> {
    if coords.is_empty() {
        return Vec::new();
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    let chunk_batch_size = coords.len().div_ceil(threads).max(1);

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for slice in coords.chunks(chunk_batch_size) {
            handles.push(s.spawn(move || {
                slice
                    .iter()
                    .map(|&pos| (pos, Chunk::generate_grid(pos, materials, generator)))
                    .collect::<Vec<_>>()
            }));
        }

        let mut out = Vec::with_capacity(coords.len());
        for handle in handles {
            if let Ok(part) = handle.join() {
                out.extend(part);
            }
        }
        out
    })
}

fn stamp_initial_window_prefabs(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    prefab_configs: &mut PrefabConfigs,
    prefab_placed: &mut PrefabPlaced,
    pending_prefab_stamps: &mut PendingPrefabStamps,
    materials: &Materials,
    seed: u32,
) {
    for config in &mut prefab_configs.0 {
        config.positions.clear();
        for x in chunk_world.pos.x..(chunk_world.pos.x + LOAD_WIDTH) {
            config
                .positions
                .extend(prefab_xs_for_chunk(config, x, seed));
        }
        config.positions.sort_unstable();
        config.positions.dedup();
    }

    let min_y = chunk_world.pos.y * CHUNK_LENGHT as i32;
    let max_y = (chunk_world.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;
    let y_seed = (chunk_world.pos.y + LOAD_HEIGHT / 2) * CHUNK_LENGHT as i32;

    for config in &prefab_configs.0 {
        for &x in &config.positions {
            let _ = stamp_prefab_at_x(
                chunk_world,
                dirty_rects,
                (prefab_placed, pending_prefab_stamps),
                materials,
                config,
                (x, y_seed),
                (min_y, max_y),
            );
        }
    }
}

/// Startup system that loads `atoms.ron`, `gen.ron`, `prefabs/*`, opens `world.redb`,
/// populates the initial `LOAD_WIDTH x LOAD_HEIGHT` chunk window, and spawns [`WorldIoWorker`].
pub fn setup_chunk_world_system(
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut materials: ResMut<Materials>,
    mut generator: ResMut<Generator>,
    mut prefab_configs: ResMut<PrefabConfigs>,
    mut prefab_placed: ResMut<PrefabPlaced>,
    mut pending_prefab_stamps: ResMut<PendingPrefabStamps>,
    mut io_worker: ResMut<WorldIoWorker>,
    sim_settings: Res<SimulationSettings>,
) {
    let assets_dir = resolve_assets_dir();

    // 1. Load Materials from `assets/atoms.ron`
    let atoms_path = assets_dir.join("atoms.ron");
    *materials = std::fs::read(&atoms_path)
        .ok()
        .and_then(|bytes| Materials::from_ron_bytes(&bytes).ok())
        .unwrap_or_else(Materials::default_builtin);

    // 2. Load Generator from `assets/gen.ron`
    let gen_path = assets_dir.join("gen.ron");
    *generator = std::fs::read(&gen_path)
        .ok()
        .and_then(|bytes| Generator::from_ron_bytes(&bytes).ok())
        .unwrap_or_else(Generator::default_config);

    // 3. Load PrefabConfigs from `assets/prefabs/*`
    *prefab_configs = PrefabConfigs::load_from_assets_dir(&assets_dir);
    prefab_placed.x_ranges.clear();
    pending_prefab_stamps.0.clear();

    // 4. Center initial chunk window around `sim_settings.focus_world_pos`
    let focus_global = world_to_global_atom(sim_settings.focus_world_pos);
    let focus_chunk = global_to_chunk(focus_global).chunk;
    chunk_world.pos = ivec2(
        focus_chunk.x - LOAD_WIDTH / 2,
        focus_chunk.y - LOAD_HEIGHT / 2,
    );
    chunk_world.chunks.clear();

    // 5. Open `saves/world.redb` (outside `assets/`) and populate the initial window
    let db_path = resolve_world_db_path();
    let storage = WorldStorage::open(&db_path).ok();
    let is_new_world = storage.as_ref().is_none_or(|s| !s.has_any_chunks());

    let mut to_generate = Vec::new();
    for (x, y) in (chunk_world.pos.x..(chunk_world.pos.x + LOAD_WIDTH))
        .cartesian_product(chunk_world.pos.y..(chunk_world.pos.y + LOAD_HEIGHT))
    {
        let pos = ivec2(x, y);
        if let Some(grid) = storage.as_ref().and_then(|s| s.load_chunk(x, y)) {
            chunk_world.add_chunk(pos, Chunk::from_grid(grid));
        } else {
            to_generate.push(pos);
        }
    }

    let generated = generate_chunks_parallel(&to_generate, &materials, &generator);
    for (pos, grid) in generated {
        chunk_world.add_chunk(pos, Chunk::from_grid(grid));
    }

    if is_new_world {
        stamp_initial_window_prefabs(
            &mut chunk_world,
            &mut dirty_rects,
            &mut prefab_configs,
            &mut prefab_placed,
            &mut pending_prefab_stamps,
            &materials,
            generator.seed,
        );
    }

    let full_rect = URect::full_chunk();
    for &pos in chunk_world.chunks.keys() {
        dirty_rects.current.insert(pos, full_rect);
        dirty_rects.render.insert(pos, full_rect);
    }

    if let Some(storage) = storage {
        *io_worker = WorldIoWorker::spawn(storage);
        if let Some(handle) = io_worker.save_handle() {
            chunk_world.attach_save_handle(handle);
        }
        if is_new_world {
            chunk_world.save_all_chunks(&io_worker);
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
    prefab_configs: Option<Res<PrefabConfigs>>,
    mut prefab_placed: Option<ResMut<PrefabPlaced>>,
    mut pending_prefab_stamps: Option<ResMut<PendingPrefabStamps>>,
) {
    let full_rect = URect::full_chunk();
    let mut loaded_from_file = Vec::new();
    let mut missing_to_generate = Vec::new();

    while let Some(response) = io_worker.poll_response() {
        match response {
            IoResponse::ChunkLoaded((cx, cy), loaded_grid) => {
                let pos = ivec2(cx, cy);
                match loaded_grid {
                    Some(grid) => loaded_from_file.push((pos, grid)),
                    None => missing_to_generate.push(pos),
                }
            }
        }
    }

    if loaded_from_file.is_empty() && missing_to_generate.is_empty() {
        return;
    }

    for &(pos, ref grid) in &loaded_from_file {
        chunk_world.add_chunk(pos, Chunk::from_grid(grid.clone()));
        dirty_rects.current.insert(pos, full_rect);
        dirty_rects.render.insert(pos, full_rect);
    }

    if !missing_to_generate.is_empty() {
        match (materials.as_deref(), generator.as_deref()) {
            (Some(mats), Some(generator)) if !mats.0.is_empty() => {
                let generated = generate_chunks_parallel(&missing_to_generate, mats, generator);
                for (pos, grid) in generated {
                    chunk_world.add_chunk(pos, Chunk::from_grid(grid));
                    dirty_rects.current.insert(pos, full_rect);
                    dirty_rects.render.insert(pos, full_rect);
                }
            }
            _ => {
                for &pos in &missing_to_generate {
                    chunk_world.add_chunk(pos, Chunk::default());
                    dirty_rects.current.insert(pos, full_rect);
                    dirty_rects.render.insert(pos, full_rect);
                }
            }
        }
    }

    if let (Some(mats), Some(generator), Some(configs), Some(placed), Some(pending)) = (
        materials.as_deref(),
        generator.as_deref(),
        prefab_configs.as_deref(),
        prefab_placed.as_deref_mut(),
        pending_prefab_stamps.as_deref_mut(),
    ) && !mats.0.is_empty()
    {
        for &(pos, _) in &loaded_from_file {
            apply_pending_stamps_for_chunk(&mut chunk_world, &mut dirty_rects, mats, pending, pos);
        }
        for &pos in &missing_to_generate {
            stamp_prefabs_for_new_chunk(
                &mut chunk_world,
                &mut dirty_rects,
                placed,
                pending,
                configs,
                mats,
                generator.seed,
                pos,
            );
        }
    }
}

/// Streams chunks in/out of [`ChunkWorld`] as [`SimulationSettings::focus_world_pos`] moves.
pub fn stream_chunk_window_system(
    sim_settings: Res<SimulationSettings>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut io_worker: ResMut<WorldIoWorker>,
) {
    if !sim_settings.stream_chunks || chunk_world.chunks.is_empty() {
        return;
    }

    let focus_global = world_to_global_atom(sim_settings.focus_world_pos);
    let focus_chunk = global_to_chunk(focus_global).chunk;

    let diff_x = focus_chunk.x - chunk_world.pos.x - LOAD_WIDTH / 2;
    let diff_y = focus_chunk.y - chunk_world.pos.y - LOAD_HEIGHT / 2;
    let mut diff = ivec2(diff_x, diff_y);

    const MAX_STEPS_PER_FRAME: i32 = 2;
    for _ in 0..MAX_STEPS_PER_FRAME {
        if diff == IVec2::ZERO {
            break;
        }
        if diff.x != 0 {
            let step = diff.x.signum();
            chunk_world.move_window(MoveDir::X(step), &mut io_worker);
            diff.x -= step;
        } else if diff.y != 0 {
            let step = diff.y.signum();
            chunk_world.move_window(MoveDir::Y(step), &mut io_worker);
            diff.y -= step;
        }
    }
}

/// Simulation system that advances the cellular automata world each frame.
pub fn chunk_world_update_system(
    mut sim_settings: ResMut<SimulationSettings>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    materials: Res<Materials>,
) {
    let steps = if sim_settings.paused {
        if sim_settings.step_once {
            sim_settings.step_once = false;
            1
        } else {
            0
        }
    } else {
        sim_settings.steps_per_frame.clamp(1, 8)
    };

    for _ in 0..steps {
        step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
    }
}
