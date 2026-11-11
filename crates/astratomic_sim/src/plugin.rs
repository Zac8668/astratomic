use bones_framework::prelude::*;

use crate::{
    atom::{Atom, AtomState, Material, Materials},
    chunk::Chunk,
    particle::{
        Particle, ParticleActorTarget, ParticleActorTargets, ParticlePool,
        ParticleRigidBodyObstacle, ParticleRigidBodyObstacles, simulate_particles_system,
    },
    world::{
        ChunkGeneratorHook, ChunkPos, ChunkWorld, DirtyRects, MaterialsAssetTracker,
        SimulationSettings, URect, chunk_world_update_system, hot_reload_materials_system,
        poll_chunk_io_system, setup_chunk_world_system, stream_chunk_window_system,
    },
    world_store::WorldIoWorker,
};

/// Registers all simulation schemas with the global `bones_schema` registry.
pub fn register_sim_schemas() {
    Atom::register_schema();
    AtomState::register_schema();
    Material::register_schema();
    Materials::register_schema();
    MaterialsAssetTracker::register_schema();
    Chunk::register_schema();
    ChunkWorld::register_schema();
    DirtyRects::register_schema();
    ChunkGeneratorHook::register_schema();
    SimulationSettings::register_schema();
    URect::register_schema();
    ChunkPos::register_schema();
    WorldIoWorker::register_schema();
    Particle::register_schema();
    ParticleActorTarget::register_schema();
    ParticleActorTargets::register_schema();
    ParticleRigidBodyObstacle::register_schema();
    ParticleRigidBodyObstacles::register_schema();
    ParticlePool::register_schema();
}

/// Bones plugin for `astratomic_sim` that initializes chunk world state, procedural generation,
/// cellular automata stepping, and background I/O polling.
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
        session.world.init_resource::<ChunkGeneratorHook>();
        session.world.init_resource::<SimulationSettings>();
        session.world.init_resource::<Materials>();
        session.world.init_resource::<MaterialsAssetTracker>();
        session.world.init_resource::<WorldIoWorker>();
        session.world.init_resource::<ParticlePool>();
        session.world.init_resource::<ParticleActorTargets>();
        session.world.init_resource::<ParticleRigidBodyObstacles>();

        session
            .add_system_to_stage(CoreStage::Startup, setup_chunk_world_system)
            .add_system_to_stage(CoreStage::PreUpdate, hot_reload_materials_system)
            .add_system_to_stage(CoreStage::PreUpdate, poll_chunk_io_system)
            .add_system_to_stage(CoreStage::PreUpdate, stream_chunk_window_system)
            .add_system_to_stage(CoreStage::Update, chunk_world_update_system)
            .add_system_to_stage(CoreStage::Update, simulate_particles_system);
    }
}
