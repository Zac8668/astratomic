//! Bones session plugin integration for `astratomic_physics`.

use bones_framework::prelude::*;

use crate::actor::{ActorState, KinematicActor, kinematic_actor_movement_system};

/// Registers all `astratomic_physics` schemas with the global `bones_schema` registry.
pub fn register_physics_schemas() {
    KinematicActor::register_schema();
    ActorState::register_schema();
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
        session.add_system_to_stage(CoreStage::Update, kinematic_actor_movement_system);
    }
}
