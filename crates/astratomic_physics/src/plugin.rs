//! Bones session plugin integration for `astratomic_physics`.

use bones_framework::prelude::*;

use crate::actor::{
    ActorSimulationStamp, ActorState, KinematicActor, clear_actor_objects_system,
    kinematic_actor_movement_system, stamp_actor_objects_system,
};

/// Registers all `astratomic_physics` schemas with the global `bones_schema` registry.
pub fn register_physics_schemas() {
    KinematicActor::register_schema();
    ActorState::register_schema();
    ActorSimulationStamp::register_schema();
}

/// Bones plugin that registers [`KinematicActor`] and [`ActorState`] schemas and runs
/// [`kinematic_actor_movement_system`].
#[derive(Default, Clone, Copy)]
pub struct AstratomicPhysicsPlugin;

impl GamePlugin for AstratomicPhysicsPlugin {
    fn install(self, _game: &mut Game) {
        register_physics_schemas();
    }
}

impl SessionPlugin for AstratomicPhysicsPlugin {
    fn install(self, session: &mut Session) {
        register_physics_schemas();
        session.world.init_resource::<ActorSimulationStamp>();
        session
            .add_system_to_stage(CoreStage::PreUpdate, kinematic_actor_movement_system)
            .add_system_to_stage(CoreStage::PreUpdate, stamp_actor_objects_system)
            .add_system_to_stage(CoreStage::Update, clear_actor_objects_system);
    }
}
