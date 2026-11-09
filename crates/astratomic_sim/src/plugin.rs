use bones_framework::prelude::*;

use crate::{
    atom::{Atom, AtomState, Material, Materials},
    chunk::Chunk,
    world::{ChunkPos, ChunkWorld, DirtyRects, Generator, URect, poll_chunk_io_system},
    world_store::WorldIoWorker,
};

/// Registers all simulation schemas with the global `bones_schema` registry.
pub fn register_sim_schemas() {
    Atom::register_schema();
    AtomState::register_schema();
    Material::register_schema();
    Materials::register_schema();
    Chunk::register_schema();
    ChunkWorld::register_schema();
    DirtyRects::register_schema();
    Generator::register_schema();
    URect::register_schema();
    ChunkPos::register_schema();
    WorldIoWorker::register_schema();
}

/// Bones plugin for `astratomic_sim` that initializes chunk world state and background I/O polling.
#[derive(Default, Clone, Copy)]
pub struct AstratomicSimPlugin;

impl GamePlugin for AstratomicSimPlugin {
    fn install(self, _game: &mut Game) {
        register_sim_schemas();
    }
}

impl SessionPlugin for AstratomicSimPlugin {
    fn install(self, session: &mut Session) {
        register_sim_schemas();

        session.world.init_resource::<ChunkWorld>();
        session.world.init_resource::<DirtyRects>();
        session.world.init_resource::<Generator>();
        session.world.init_resource::<Materials>();
        session.world.init_resource::<WorldIoWorker>();

        session.add_system_to_stage(CoreStage::PreUpdate, poll_chunk_io_system);
    }
}
