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
mod puffin_plugin;
mod rigidbody;
mod prelude {
    pub use crate::GameState;
    pub use crate::{
        actors::*, animation::*, atom::*, camera::*, chunk::*, chunk_group::*, chunk_manager::*,
        consts::*, debug::*, geom_tools::*, manager_api::*, materials::*, menu::*, particles::*,
        player::*, puffin_plugin::*, rigidbody::*,
    };
    pub use bevy::input::mouse::MouseScrollUnit;
    pub use bevy::input::mouse::MouseWheel;
    pub use bevy::math::{ivec2, uvec2, vec2, vec3};
    pub use bevy::prelude::*;
    pub use bevy::tasks::*;
    pub use bevy::window::PrimaryWindow;
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

    pub use crate::materials::Material;
    pub use bevy_egui::EguiContext;
}

use prelude::*;

fn main() {
    let args: Vec<_> = env::args().collect();

    let mut app = App::new();

    app.add_state::<GameState>()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
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
        ))
        .add_plugins((
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(6.),
            MenuPlugin,
        ))
        .add_systems(Startup, setup);

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

    let mut camera = Camera2dBundle::default();
    camera.camera.hdr = true;
    camera.transform.scale.x = 0.23;
    camera.transform.scale.y = 0.23;

    commands.spawn(camera);
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, States)]
pub enum GameState {
    Menu,
    Game,
}

impl Default for GameState {
    fn default() -> Self {
        let args: Vec<_> = env::args().collect();

        if args.contains(&"-g".to_string()) || args.contains(&"--game".to_string()) {
            GameState::Game
        } else {
            GameState::Menu
        }
    }
}
