//! Bones session plugin integration for `astratomic_physics`.

use bones_framework::prelude::*;

use crate::{
    actor::{ActorSimulationStamp, ActorState, KinematicActor},
    rigidbody::{
        ActorPhysicsHandle, PhysicsBodyHandle, PixelRigidBody, RapierPhysics,
        particle_rigidbody_interaction_system,
    },
};

/// Registers all `astratomic_physics` schemas with the global `bones_schema` registry.
pub fn register_physics_schemas() {
    KinematicActor::register_schema();
    ActorState::register_schema();
    ActorSimulationStamp::register_schema();
    RapierPhysics::register_schema();
    PhysicsBodyHandle::register_schema();
    ActorPhysicsHandle::register_schema();
    PixelRigidBody::register_schema();
    crate::rigidbody::RigidbodySimulationStamp::register_schema();
}

/// Bones plugin that registers [`KinematicActor`], [`RapierPhysics`], and [`PixelRigidBody`] schemas
/// and runs physics systems.
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
        session.world.init_resource::<crate::rigidbody::RigidbodySimulationStamp>();
        session.world.init_resource::<RapierPhysics>();
        session
            .add_system_to_stage(CoreStage::Update, particle_rigidbody_interaction_system);
    }
}
