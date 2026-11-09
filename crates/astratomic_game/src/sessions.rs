//! Bones game sessions and lifecycle management.

use astratomic_render::prelude::AstratomicRenderPlugin;
use astratomic_sim::prelude::AstratomicSimPlugin;
use bones_framework::prelude::*;

use crate::{meta::GameMeta, ui::AstratomicDebugUiPlugin};

/// Named session identifiers (following Jumpy's session convention).
pub mod session_names {
    pub const WORLD: &str = "world";
}

/// Marker component for the player entity.
#[derive(HasSchema, Clone, Copy, Default)]
#[repr(C)]
pub struct Player;

/// Top-level Bones [`GamePlugin`] that registers schemas and starts the initial session.
pub struct AstratomicGamePlugin;

impl GamePlugin for AstratomicGamePlugin {
    fn install(self, game: &mut Game) {
        game.install_plugin(DefaultGamePlugin);

        // Initialize the asset server and register built-in + game schemas.
        game.init_shared_resource::<AssetServer>()
            .register_default_assets();
        GameMeta::register_schema();
        Player::register_schema();

        // Create the main gameplay world session.
        game.sessions
            .create_with(session_names::WORLD, |session: &mut Session| {
                session.install_plugin(world_session_plugin);
            });
    }
}

/// Session plugin for the main world session.
pub fn world_session_plugin(session: &mut Session) {
    session
        .install_plugin(DefaultSessionPlugin)
        .install_plugin(AstratomicSimPlugin)
        .install_plugin(AstratomicRenderPlugin)
        .install_plugin(AstratomicDebugUiPlugin)
        .add_system_to_stage(CoreStage::Startup, setup_world);
}

/// Startup system that spawns the 2D pixel-art camera looking at the terrain surface.
fn setup_world(
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut cameras: CompMut<Camera>,
) {
    let camera_ent = entities.create();
    transforms.insert(
        camera_ent,
        Transform::from_translation(Vec3::new(0.0, 200.0, 1000.0)),
    );
    cameras.insert(
        camera_ent,
        Camera {
            size: CameraSize::FixedHeight(420.0),
            background_color: Set(Color::rgb(0.08, 0.09, 0.13)),
            ..default()
        },
    );
}
