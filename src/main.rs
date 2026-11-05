#![allow(clippy::type_complexity)]

use bevy::prelude::*;

mod actors;
mod animation;
mod atom;
mod camera;
mod chunk;
mod chunk_group;
mod chunk_manager;
mod consts;
mod debug;
mod geom_tools;
mod manager_api;
mod materials;
mod menu;
mod particles;
mod player;
mod player_inv;
mod prefabs;
mod puffin_plugin;
mod rigidbody;
mod ui_fonts;
mod world_store;
mod prelude {
    pub use crate::GameState;
    pub use crate::{
        actors::*, animation::*, atom::*, camera::*, chunk::*, chunk_group::*, chunk_manager::*,
        consts::*, debug::*, geom_tools::*, manager_api::*, materials::*, menu::*, particles::*,
        player::*, player_inv::*, prefabs::*, puffin_plugin::*, rigidbody::*, ui_fonts::*,
        world_store::*,
    };
    pub use bevy::input::mouse::MouseScrollUnit;
    pub use bevy::input::mouse::MouseWheel;
    pub use bevy::math::{ivec2, uvec2, vec2, vec3};
    pub use bevy::prelude::*;
    pub use bevy::tasks::*;
    pub use bevy_async_task::*;

    pub use serde::{Deserialize, Serialize};
    pub use serde_big_array::BigArray;

    pub use std::collections::{HashMap, HashSet};
    pub use std::env;
    pub use std::fs::File;
    pub use std::io::Write;
    pub use std::io::{BufReader, BufWriter};
    pub use std::sync::{Arc, RwLock};

    pub use bevy_rapier2d::prelude::*;
    pub use contour::ContourBuilder;

    //pub use bevy_egui::EguiContext;

    pub use bevy_rapier2d::prelude::RigidBody as RapierRigidbody;
}

use prelude::*;

use bevy::asset::AssetPlugin;

fn main() {
    let args: Vec<_> = env::args().collect();

    let mut app = App::new();

    app.add_plugins(
        DefaultPlugins
            .set(ImagePlugin::default_nearest())
            .set(AssetPlugin {
                watch_for_changes_override: Some(false),
                ..default()
            }),
    )
    //local plugins
    .add_plugins((
        ChunkManagerPlugin,
        ActorsPlugin,
        PlayerPlugin,
        animation::AnimationPlugin,
        ParticlesPlugin,
        MaterialsPlugin,
        CameraPlugin,
        RigidbodyPlugin,
        PlayerInvPlugin,
        PrefabsPlugin,
        QuaverFontPlugin,
    ))
    .add_plugins((
        RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.),
        MenuPlugin,
    ))
    .add_systems(Startup, setup)
    .init_state::<GameState>();

    if args.contains(&"-d".to_string()) || args.contains(&"--debug".to_string()) {
        app.add_plugins((DebugPlugin,));
    }

    if args.contains(&"-p".to_string()) || args.contains(&"--profiling".to_string()) {
        app.add_plugins(PuffinPlugin);
    }

    app.run();
}

fn setup(mut commands: Commands, mut time: ResMut<Time<Fixed>>) {
    time.set_timestep_hz(58.);
    commands.spawn((Camera2d, Transform::from_scale(Vec3::new(0.23, 0.23, 1.))));
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, States)]
pub enum GameState {
    Menu,
    Loading,
    Game,
}

impl Default for GameState {
    fn default() -> Self {
        let args: Vec<_> = env::args().collect();

        if args.contains(&"-g".to_string()) || args.contains(&"--game".to_string()) {
            GameState::Loading
        } else {
            GameState::Menu
        }
    }
}
