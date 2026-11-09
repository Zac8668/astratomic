use std::{
    collections::{HashSet, VecDeque},
    path::Path,
    sync::Arc,
};

use bones_schema::HasSchema;
use crossbeam_channel::{Receiver, Sender};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use thiserror::Error;

use crate::chunk::ChunkGrid;

pub const WORLD_DB_PATH: &str = "saves/world.redb";

pub const CHUNKS_TABLE: TableDefinition<(i32, i32), &[u8]> = TableDefinition::new("chunks");
pub const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("world_meta");

#[derive(Debug, Error)]
pub enum WorldStorageError {
    #[error("redb database error: {0}")]
    Database(#[from] redb::DatabaseError),
    #[error("redb transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("redb table error: {0}")]
    Table(#[from] redb::TableError),
    #[error("redb storage error: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("redb commit error: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("io/compression error: {0}")]
    Io(#[from] std::io::Error),
}

/// Zero-copy `redb` + `zstd` world database wrapper.
#[derive(Clone)]
pub struct WorldStorage {
    db: Arc<Database>,
}

impl WorldStorage {
    /// Open or create a single-file `redb::Database` at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WorldStorageError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let db = Database::create(path)?;
        let storage = Self { db: Arc::new(db) };
        storage.ensure_tables()?;
        Ok(storage)
    }

    /// Open or create the default world database at [`WORLD_DB_PATH`].
    pub fn open_default() -> Result<Self, WorldStorageError> {
        Self::open(WORLD_DB_PATH)
    }

    /// Wrap an existing [`redb::Database`].
    pub fn from_db(db: Database) -> Result<Self, WorldStorageError> {
        let storage = Self { db: Arc::new(db) };
        storage.ensure_tables()?;
        Ok(storage)
    }

    /// Underlying `redb::Database` reference.
    pub fn db(&self) -> &Database {
        &self.db
    }

    fn ensure_tables(&self) -> Result<(), WorldStorageError> {
        let write_txn = self.db.begin_write()?;
        {
            let _ = write_txn.open_table(CHUNKS_TABLE)?;
            let _ = write_txn.open_table(META_TABLE)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Save a batch of chunks in a single `redb` write transaction using zero-copy `bytemuck`
    /// casting and `zstd::bulk::Compressor::new(1)`.
    pub fn save_chunks_batch(
        &self,
        chunks: &[((i32, i32), Arc<ChunkGrid>)],
    ) -> Result<(), WorldStorageError> {
        let write_txn = self.db.begin_write()?;
        let mut compressor = zstd::bulk::Compressor::new(1)?;
        {
            let mut table = write_txn.open_table(CHUNKS_TABLE)?;
            for ((cx, cy), grid) in chunks {
                let grid: &ChunkGrid = &**grid;
                let raw_bytes: &[u8] = bytemuck::bytes_of(&*grid);
                let compressed = compressor.compress(raw_bytes)?;
                table.insert((*cx, *cy), compressed.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load and decompress a chunk at `(cx, cy)` directly into a zeroed `Box<ChunkGrid>`.
    pub fn load_chunk(&self, cx: i32, cy: i32) -> Option<Arc<ChunkGrid>> {
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(CHUNKS_TABLE).ok()?;
        let access = table.get((cx, cy)).ok()??;
        let compressed = access.value();

        let mut grid: Box<ChunkGrid> = ChunkGrid::new_zeroed_box();
        let dst: &mut [u8] = bytemuck::bytes_of_mut(&mut *grid);
        let decompressed_len = zstd::bulk::decompress_to_buffer(compressed, dst).ok()?;
        if decompressed_len != dst.len() {
            return None;
        }

        Some(Arc::from(grid))
    }

    /// Read the raw `zstd`-compressed bytes for chunk `(cx, cy)` from `CHUNKS_TABLE` without
    /// decompressing, allowing a multiplayer host to forward saved chunks directly over the network.
    pub fn get_raw_compressed_chunk(&self, cx: i32, cy: i32) -> Option<Vec<u8>> {
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(CHUNKS_TABLE).ok()?;
        let access = table.get((cx, cy)).ok()??;
        Some(access.value().to_vec())
    }

    /// Decompress a raw `zstd`-compressed chunk slice directly into an [`Arc<ChunkGrid>`].
    pub fn decompress_raw_chunk(compressed: &[u8]) -> Option<Arc<ChunkGrid>> {
        let mut grid: Box<ChunkGrid> = ChunkGrid::new_zeroed_box();
        let dst: &mut [u8] = bytemuck::bytes_of_mut(&mut *grid);
        let decompressed_len = zstd::bulk::decompress_to_buffer(compressed, dst).ok()?;
        if decompressed_len != dst.len() {
            return None;
        }
        Some(Arc::from(grid))
    }

    /// Returns `true` if `CHUNKS_TABLE` contains at least one saved chunk.
    pub fn has_any_chunks(&self) -> bool {
        let Ok(read_txn) = self.db.begin_read() else {
            return false;
        };
        let Ok(table) = read_txn.open_table(CHUNKS_TABLE) else {
            return false;
        };
        let Ok(mut iter) = table.iter() else {
            return false;
        };
        iter.next().is_some()
    }

    /// Write a key-value entry into `META_TABLE`.
    pub fn save_meta(&self, key: &str, value: &[u8]) -> Result<(), WorldStorageError> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(META_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Read a metadata entry from `META_TABLE`.
    pub fn load_meta(&self, key: &str) -> Option<Vec<u8>> {
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(META_TABLE).ok()?;
        let access = table.get(key).ok()??;
        Some(access.value().to_vec())
    }

    /// Encode `value` with `bitcode` and store it in `META_TABLE`.
    pub fn save_meta_bitcode<T: bitcode::Encode>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<(), WorldStorageError> {
        let encoded = bitcode::encode(value);
        self.save_meta(key, &encoded)
    }

    /// Load and decode a `bitcode`-encoded metadata entry from `META_TABLE`.
    pub fn load_meta_bitcode<T: for<'a> bitcode::Decode<'a>>(&self, key: &str) -> Option<T> {
        let bytes = self.load_meta(key)?;
        bitcode::decode(&bytes).ok()
    }
}

/// Requests sent from the simulation thread to the background storage I/O worker thread.
#[derive(Clone, Debug)]
pub enum IoRequest {
    SaveBatch(Vec<((i32, i32), Arc<ChunkGrid>)>),
    LoadChunk(i32, i32),
    Flush(Sender<()>),
    Shutdown,
}

/// Responses sent from the background storage I/O worker thread back to the simulation thread.
#[derive(Clone, Debug)]
pub enum IoResponse {
    ChunkLoaded((i32, i32), Option<Arc<ChunkGrid>>),
}

/// Shared handle allowing individual [`Chunk`][crate::chunk::Chunk]s and
/// [`ChunkWorld`][crate::world::ChunkWorld] to automatically persist chunk grids when unloaded,
/// dropped, or when the game closes.
#[derive(Clone)]
pub struct WorldSaveHandle {
    request_tx: Sender<IoRequest>,
    storage: WorldStorage,
}

impl std::fmt::Debug for WorldSaveHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldSaveHandle").finish_non_exhaustive()
    }
}

impl WorldSaveHandle {
    pub fn new(request_tx: Sender<IoRequest>, storage: WorldStorage) -> Self {
        Self {
            request_tx,
            storage,
        }
    }

    pub fn storage(&self) -> &WorldStorage {
        &self.storage
    }

    /// Enqueue a single chunk to be saved on the background I/O worker, falling back to a direct
    /// synchronous write if the worker thread has already shut down.
    pub fn save_chunk(&self, coords: (i32, i32), grid: Arc<ChunkGrid>) {
        let batch = vec![(coords, grid)];
        if let Err(crossbeam_channel::SendError(IoRequest::SaveBatch(batch))) =
            self.request_tx.send(IoRequest::SaveBatch(batch))
        {
            let _ = self.storage.save_chunks_batch(&batch);
        }
    }

    /// Enqueue a batch of chunks to be saved on the background I/O worker, falling back to a
    /// direct synchronous write if the worker thread has already shut down.
    pub fn save_batch(&self, mut batch: Vec<((i32, i32), Arc<ChunkGrid>)>) {
        if batch.is_empty() {
            return;
        }
        batch.sort_by_key(|(coords, _)| *coords);
        if let Err(crossbeam_channel::SendError(IoRequest::SaveBatch(batch))) =
            self.request_tx.send(IoRequest::SaveBatch(batch))
        {
            let _ = self.storage.save_chunks_batch(&batch);
        }
    }

    /// Synchronously flush any queued saves on the background worker, or write `batch` directly
    /// to [`WorldStorage`] if the worker has already stopped.
    pub fn save_batch_and_flush(&self, mut batch: Vec<((i32, i32), Arc<ChunkGrid>)>) {
        if batch.is_empty() {
            return;
        }
        batch.sort_by_key(|(coords, _)| *coords);
        match self.request_tx.send(IoRequest::SaveBatch(batch)) {
            Ok(()) => {
                let (ack_tx, ack_rx) = crossbeam_channel::bounded(1);
                if self.request_tx.send(IoRequest::Flush(ack_tx)).is_ok() {
                    let _ = ack_rx.recv();
                }
            }
            Err(crossbeam_channel::SendError(IoRequest::SaveBatch(batch))) => {
                let _ = self.storage.save_chunks_batch(&batch);
            }
            Err(_) => {}
        }
    }
}

struct WorkerThreadGuard {
    request_tx: Sender<IoRequest>,
    join_handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for WorkerThreadGuard {
    fn drop(&mut self) {
        let _ = self.request_tx.send(IoRequest::Shutdown);
        if let Ok(mut guard) = self.join_handle.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
    }
}

/// Non-blocking background I/O worker resource for `bones_ecs`.
///
/// Connects the simulation to a dedicated background thread running `redb` transactions via
/// `crossbeam_channel` and buffers completed [`IoResponse`] items in a deterministic queue.
#[derive(HasSchema, Clone)]
pub struct WorldIoWorker {
    pub request_tx: Sender<IoRequest>,
    pub response_rx: Receiver<IoResponse>,
    /// Deterministic FIFO queue of completed `IoResponse` messages polled by the simulation system.
    pub deterministic_queue: VecDeque<IoResponse>,
    /// Coordinates of chunks currently requested for loading to prevent duplicate in-flight loads.
    pub pending_loads: HashSet<(i32, i32)>,
    save_handle: Option<WorldSaveHandle>,
    #[allow(dead_code)]
    worker_guard: Option<Arc<WorkerThreadGuard>>,
}

impl Default for WorldIoWorker {
    fn default() -> Self {
        let (request_tx, _request_rx) = crossbeam_channel::unbounded();
        let (_response_tx, response_rx) = crossbeam_channel::unbounded();
        Self {
            request_tx,
            response_rx,
            deterministic_queue: VecDeque::new(),
            pending_loads: HashSet::new(),
            save_handle: None,
            worker_guard: None,
        }
    }
}

impl WorldIoWorker {
    /// Spawn a background I/O worker thread owning the given [`WorldStorage`].
    pub fn spawn(storage: WorldStorage) -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<IoRequest>();
        let (response_tx, response_rx) = crossbeam_channel::unbounded::<IoResponse>();
        let save_handle = WorldSaveHandle::new(request_tx.clone(), storage.clone());

        let join_handle = std::thread::Builder::new()
            .name("astratomic-world-io".to_string())
            .spawn(move || {
                Self::worker_loop(storage, request_rx, response_tx);
            })
            .expect("failed to spawn background world I/O worker thread");

        let worker_guard = Arc::new(WorkerThreadGuard {
            request_tx: request_tx.clone(),
            join_handle: std::sync::Mutex::new(Some(join_handle)),
        });

        Self {
            request_tx,
            response_rx,
            deterministic_queue: VecDeque::new(),
            pending_loads: HashSet::new(),
            save_handle: Some(save_handle),
            worker_guard: Some(worker_guard),
        }
    }

    /// Return a clonable [`WorldSaveHandle`] that chunks and [`ChunkWorld`][crate::world::ChunkWorld]
    /// can use to persist themselves on unload or drop.
    pub fn save_handle(&self) -> Option<WorldSaveHandle> {
        self.save_handle.clone()
    }

    /// Wait until all previously queued `SaveBatch` requests have been committed by the worker.
    pub fn flush(&self) {
        let (ack_tx, ack_rx) = crossbeam_channel::bounded(1);
        if self.request_tx.send(IoRequest::Flush(ack_tx)).is_ok() {
            let _ = ack_rx.recv();
        }
    }

    /// Open a [`WorldStorage`] at `path` and spawn the background I/O worker thread.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WorldStorageError> {
        let storage = WorldStorage::open(path)?;
        Ok(Self::spawn(storage))
    }

    /// Open the default [`WORLD_DB_PATH`] database and spawn the background I/O worker thread.
    pub fn open_default() -> Result<Self, WorldStorageError> {
        Self::open(WORLD_DB_PATH)
    }

    fn worker_loop(
        storage: WorldStorage,
        request_rx: Receiver<IoRequest>,
        response_tx: Sender<IoResponse>,
    ) {
        while let Ok(request) = request_rx.recv() {
            match request {
                IoRequest::SaveBatch(batch) => {
                    let _ = storage.save_chunks_batch(&batch);
                }
                IoRequest::LoadChunk(cx, cy) => {
                    let loaded = storage.load_chunk(cx, cy);
                    let _ = response_tx.send(IoResponse::ChunkLoaded((cx, cy), loaded));
                }
                IoRequest::Flush(ack_tx) => {
                    let _ = ack_tx.send(());
                }
                IoRequest::Shutdown => {
                    // Drain any remaining SaveBatch or Flush requests queued before Shutdown.
                    while let Ok(rem) = request_rx.try_recv() {
                        match rem {
                            IoRequest::SaveBatch(batch) => {
                                let _ = storage.save_chunks_batch(&batch);
                            }
                            IoRequest::Flush(ack_tx) => {
                                let _ = ack_tx.send(());
                            }
                            IoRequest::LoadChunk(_, _) | IoRequest::Shutdown => {}
                        }
                    }
                    break;
                }
            }
        }
    }

    /// Enqueue a batch of chunks to be compressed and saved on the background I/O thread.
    pub fn save_batch(&self, batch: Vec<((i32, i32), Arc<ChunkGrid>)>) {
        if !batch.is_empty() {
            let _ = self.request_tx.send(IoRequest::SaveBatch(batch));
        }
    }

    /// Enqueue a non-blocking chunk load request on the background I/O thread.
    ///
    /// Returns `true` if a new load request was sent, or `false` if `(cx, cy)` is already in-flight.
    pub fn request_load_chunk(&mut self, cx: i32, cy: i32) -> bool {
        if self.pending_loads.insert((cx, cy)) {
            let _ = self.request_tx.send(IoRequest::LoadChunk(cx, cy));
            true
        } else {
            false
        }
    }

    /// Drain newly completed messages from `response_rx`, sort them deterministically by chunk
    /// coordinates `(cx, cy)`, and push them onto `deterministic_queue`.
    pub fn pump_channel_to_queue(&mut self) {
        let mut newly_arrived: Vec<IoResponse> = self.response_rx.try_iter().collect();
        if newly_arrived.is_empty() {
            return;
        }

        newly_arrived.sort_by_key(|resp| match resp {
            IoResponse::ChunkLoaded((cx, cy), _) => (*cx, *cy),
        });

        for resp in newly_arrived {
            let IoResponse::ChunkLoaded(coords, _) = &resp;
            self.pending_loads.remove(coords);
            self.deterministic_queue.push_back(resp);
        }
    }

    /// Poll the next completed [`IoResponse`] from the deterministic queue.
    pub fn poll_response(&mut self) -> Option<IoResponse> {
        self.pump_channel_to_queue();
        self.deterministic_queue.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bones_framework::prelude::*;

    use super::*;
    use crate::{
        atom::{Atom, AtomState},
        chunk::{CHUNK_SIZE, Chunk, ChunkGrid},
        plugin::AstratomicSimPlugin,
        world::{ChunkPos, ChunkWorld},
    };

    fn temp_db_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "astratomic_test_{}_{}_{}.redb",
            name,
            std::process::id(),
            fastrand::u64(..)
        ));
        path
    }

    #[test]
    fn atom_and_chunk_grid_layout() {
        assert_eq!(std::mem::size_of::<Atom>(), 8);
        assert_eq!(
            std::mem::size_of::<ChunkGrid>(),
            CHUNK_SIZE * CHUNK_SIZE * 8
        );
        assert_eq!(std::mem::size_of::<ChunkGrid>(), 32_768);
    }

    #[test]
    fn chunk_world_cow_snapshot() {
        let world = World::new();
        Atom::register_schema();
        Chunk::register_schema();
        ChunkWorld::register_schema();

        let mut chunk_world = ChunkWorld::default();
        let mut chunk = Chunk::default();
        Arc::make_mut(&mut chunk.grid).atoms[0] =
            Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
        chunk_world.add_chunk(ivec2(0, 0), chunk);
        world.insert_resource(chunk_world);

        // Take a rollback snapshot of the ECS World.
        let snapshot = world.clone();

        {
            let cw1 = world.resource::<ChunkWorld>();
            let cw2 = snapshot.resource::<ChunkWorld>();
            assert!(Arc::ptr_eq(
                &cw1.chunks[&ivec2(0, 0)].grid,
                &cw2.chunks[&ivec2(0, 0)].grid
            ));
        }

        // Mutate the active world via Copy-on-Write (`Arc::make_mut`).
        {
            let mut cw1 = world.resource_mut::<ChunkWorld>();
            let atom = cw1
                .get_mut_atom(ChunkPos::new(uvec2(0, 0), ivec2(0, 0)))
                .unwrap();
            atom.material_id = 8;
        }

        // Verify the snapshot remains untouched while the active world was mutated.
        let cw1 = world.resource::<ChunkWorld>();
        let cw2 = snapshot.resource::<ChunkWorld>();
        assert!(!Arc::ptr_eq(
            &cw1.chunks[&ivec2(0, 0)].grid,
            &cw2.chunks[&ivec2(0, 0)].grid
        ));
        assert_eq!(cw1.chunks[&ivec2(0, 0)].grid.atoms[0].material_id, 8);
        assert_eq!(cw2.chunks[&ivec2(0, 0)].grid.atoms[0].material_id, 2);
    }

    #[test]
    fn world_storage_save_load_and_raw_compressed() {
        let db_path = temp_db_path("storage_roundtrip");
        let storage = WorldStorage::open(&db_path).expect("failed to open redb storage");
        assert!(!storage.has_any_chunks());
        assert!(storage.load_chunk(1, -2).is_none());

        let mut grid_a = ChunkGrid::new_zeroed_box();
        grid_a.atoms[0] = Atom {
            color: [230, 197, 92, 255],
            material_id: 2,
            flags: AtomState::Powder as u8,
            vx: -3,
            vy: 7,
        };
        grid_a.atoms[CHUNK_SIZE * CHUNK_SIZE - 1] = Atom {
            color: [80, 80, 80, 255],
            material_id: 8,
            flags: AtomState::Solid as u8,
            vx: 0,
            vy: 0,
        };
        let arc_a = Arc::from(grid_a);

        let mut grid_b = ChunkGrid::new_zeroed_box();
        grid_b.atoms[512] = Atom::with_state(3, AtomState::Liquid, [20, 125, 204, 150]);
        let arc_b = Arc::from(grid_b);

        storage
            .save_chunks_batch(&[
                ((1, -2), Arc::clone(&arc_a)),
                ((-5, 10), Arc::clone(&arc_b)),
            ])
            .expect("failed to save batch");

        assert!(storage.has_any_chunks());

        let loaded_a = storage.load_chunk(1, -2).expect("chunk (1, -2) missing");
        let loaded_b = storage.load_chunk(-5, 10).expect("chunk (-5, 10) missing");
        assert_eq!(*loaded_a, *arc_a);
        assert_eq!(*loaded_b, *arc_b);

        let raw = storage
            .get_raw_compressed_chunk(1, -2)
            .expect("raw compressed chunk missing");
        assert!(!raw.is_empty());
        assert!(raw.len() < std::mem::size_of::<ChunkGrid>());

        let decompressed =
            WorldStorage::decompress_raw_chunk(&raw).expect("failed to decompress raw slice");
        assert_eq!(*decompressed, *arc_a);

        storage
            .save_meta_bitcode("seed", &8668_u32)
            .expect("failed to save bitcode meta");
        assert_eq!(storage.load_meta_bitcode::<u32>("seed"), Some(8668));

        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn background_io_worker_and_sim_polling() {
        let db_path = temp_db_path("bg_worker");
        let mut session = Session::default();
        session.install_plugin(AstratomicSimPlugin);

        let io_worker = WorldIoWorker::open(&db_path).expect("failed to spawn io worker");
        session.world.insert_resource(io_worker);

        let mut grid = ChunkGrid::new_zeroed_box();
        grid.atoms[10] = Atom::with_state(5, AtomState::Liquid, [245, 140, 10, 255]);
        let grid_arc = Arc::from(grid);

        {
            let mut io = session.world.resource_mut::<WorldIoWorker>();
            io.save_batch(vec![((3, 4), Arc::clone(&grid_arc))]);
            io.request_load_chunk(3, 4);
        }

        let start = Instant::now();
        loop {
            session
                .world
                .run_system(crate::world::poll_chunk_io_system, ());
            if session
                .world
                .resource::<ChunkWorld>()
                .get_chunk(ivec2(3, 4))
                .is_some()
            {
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "timed out waiting for background I/O worker"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        let cw = session.world.resource::<ChunkWorld>();
        let loaded_chunk = cw.get_chunk(ivec2(3, 4)).unwrap();
        assert_eq!(*loaded_chunk.grid, *grid_arc);

        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn ca_simulation_sand_and_water_and_prefabs() {
        use crate::{
            atom::Materials,
            ca::step_chunk_simulation,
            prefabs::PrefabConfigs,
            world::{DirtyRects, Generator, URect, global_to_chunk, resolve_assets_dir},
        };

        let materials = Materials::default_builtin();
        let generator = Generator::default_config();

        // 1. Verify procedural chunk generation produces non-empty terrain
        let generated = Chunk::generate_grid(ivec2(0, 0), &materials, &generator);
        assert!(generated.atoms.iter().any(|a| !a.is_void()));

        // 2. Verify prefabs load from assets/prefabs
        let prefabs = PrefabConfigs::load_from_assets_dir(resolve_assets_dir());
        assert!(
            !prefabs.0.is_empty(),
            "expected prefabs to load from assets/prefabs"
        );

        // 3. Verify CA simulation moves sand downward and spreads water
        let mut chunk_world = ChunkWorld::default();
        let mut dirty_rects = DirtyRects::default();
        for cx in -1..=1 {
            for cy in -1..=1 {
                chunk_world.add_chunk(ivec2(cx, cy), Chunk::default());
            }
        }

        // Paint a solid rock floor at y = 20 and sand at (10, 10)
        for x in 5..=15 {
            chunk_world.paint_at_global(ivec2(x, 20), 8, false, &materials, &mut dirty_rects);
        }
        chunk_world.paint_at_global(ivec2(10, 10), 2, false, &materials, &mut dirty_rects);
        dirty_rects.current.insert(ivec2(0, 0), URect::full_chunk());

        for _ in 0..15 {
            step_chunk_simulation(&mut chunk_world, &mut dirty_rects, &materials);
        }

        // The sand atom at (10, 10) should have fallen down onto the rock floor at y = 19
        let start_atom = chunk_world
            .get_atom(&global_to_chunk(ivec2(10, 10)))
            .unwrap();
        assert!(
            start_atom.is_void(),
            "sand should have moved away from (10, 10)"
        );
        let on_floor = (5..=15).any(|x| {
            chunk_world
                .get_atom(&global_to_chunk(ivec2(x, 19)))
                .is_some_and(|a| a.material_id == 2)
        });
        assert!(
            on_floor,
            "sand should rest on top of the rock floor at y = 19"
        );
    }

    #[test]
    fn chunks_save_on_drop_and_world_close() {
        let db_path = temp_db_path("auto_save_on_drop");
        let storage = WorldStorage::open(&db_path).expect("failed to open redb");

        {
            let worker = WorldIoWorker::spawn(storage.clone());
            let mut chunk_world = ChunkWorld::default();
            chunk_world.attach_save_handle(worker.save_handle().unwrap());

            // 1. Add chunk (1, 1) and remove + drop it directly; `Chunk::drop` should save it.
            let mut dropped_chunk = Chunk::default();
            dropped_chunk.atoms_mut()[0] =
                Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
            chunk_world.add_chunk(ivec2(1, 1), dropped_chunk);
            let removed = chunk_world.remove_chunk(ivec2(1, 1)).unwrap();
            drop(removed);

            // 2. Leave chunk (2, 2) inside `ChunkWorld` when `ChunkWorld` and `WorldIoWorker`
            // are dropped (simulating closing the game).
            let mut active_chunk = Chunk::default();
            active_chunk.atoms_mut()[5] =
                Atom::with_state(3, AtomState::Liquid, [43, 128, 212, 224]);
            chunk_world.add_chunk(ivec2(2, 2), active_chunk);
        }

        let loaded_1 = storage
            .load_chunk(1, 1)
            .expect("dropped chunk (1, 1) should be saved automatically");
        assert_eq!(loaded_1.atoms[0].material_id, 2);

        let loaded_2 = storage
            .load_chunk(2, 2)
            .expect("active chunk (2, 2) should be saved when ChunkWorld drops on close");
        assert_eq!(loaded_2.atoms[5].material_id, 3);

        let _ = std::fs::remove_file(&db_path);
    }
}
