//! Bones game sessions and lifecycle management.

use astratomic_sim::prelude::AstratomicSimPlugin;
use bones_framework::prelude::*;

use crate::meta::GameMeta;

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
        .add_system_to_stage(CoreStage::Startup, setup_world);
}

/// Startup system that spawns a 2D pixel-art camera and the player centered at `(0, 0)`.
fn setup_world(
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut cameras: CompMut<Camera>,
    mut atlas_sprites: CompMut<AtlasSprite>,
    mut animated_sprites: CompMut<AnimatedSprite>,
    mut players: CompMut<Player>,
    meta: Root<GameMeta>,
) {
    // Spawn a 2D camera centered at (0, 0) with a 180px vertical view height (~4x zoom at 720p).
    let camera_ent = entities.create();
    transforms.insert(
        camera_ent,
        Transform::from_translation(Vec3::new(0.0, 0.0, 1000.0)),
    );
    cameras.insert(
        camera_ent,
        Camera {
            size: CameraSize::FixedHeight(180.0),
            background_color: Set(Color::rgb(0.08, 0.09, 0.13)),
            ..default()
        },
    );

    // Spawn the player sprite right at the origin (0, 0, 0) with its 2-frame idle animation.
    let player_ent = entities.create();
    players.insert(player_ent, Player);
    transforms.insert(player_ent, Transform::from_translation(Vec3::ZERO));
    atlas_sprites.insert(player_ent, AtlasSprite::new(meta.player_atlas));
    animated_sprites.insert(
        player_ent,
        AnimatedSprite {
            frames: [0, 1].into_iter().collect(),
            fps: 10.0,
            repeat: true,
            ..default()
        },
    );
}
