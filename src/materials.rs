use crate::prelude::*;

use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext},
    prelude::*,
    reflect::TypePath,
};
use serde::Deserialize;
use thiserror::Error;

#[derive(Default, Debug, Deserialize, PartialEq, Clone, Copy)]
pub struct Material {
    #[serde(default)]
    pub inertial_resistance: f32,
    #[serde(default)]
    pub flow: u8,
    #[serde(default)]
    pub damage: f32,
    #[serde(default)]
    pub default_state: AtomState,
}

#[derive(Asset, TypePath, Debug, Deserialize, Default)]
pub struct Materials(pub Vec<Material>);

impl Materials {
    pub fn get_from_atom(&self, atom: &Atom) -> &Material {
        &self.0[atom.id as usize]
    }

    pub fn get_from_id(&self, id: u8) -> &Material {
        &self.0[id as usize]
    }
}

impl std::ops::Index<&Atom> for Materials {
    type Output = Material;
    #[track_caller]
    fn index(&self, atom: &Atom) -> &Self::Output {
        self.get_from_atom(atom)
    }
}

impl std::ops::Index<u8> for Materials {
    type Output = Material;
    #[track_caller]
    fn index(&self, id: u8) -> &Self::Output {
        self.get_from_id(id)
    }
}

//Asset stuff

#[derive(Resource, Default)]
pub struct MaterialsHandle(pub Handle<Materials>);

#[derive(Default)]
pub struct MaterialsLoader;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MaterialsLoaderError {
    /// An [IO](std::io) Error
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
    /// A [RON](ron) Error
    #[error("Could not parse RON: {0}")]
    RonSpannedError(#[from] ron::error::SpannedError),
}

impl AssetLoader for MaterialsLoader {
    type Asset = Materials;
    type Settings = ();
    type Error = MaterialsLoaderError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let custom_asset = ron::de::from_bytes::<Materials>(&bytes)?;
        Ok(custom_asset)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}

pub fn materials_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load("atoms.ron");
    commands.insert_resource(MaterialsHandle(handle));
}

pub fn loading_wait(
    asset_server: Res<AssetServer>,
    materials: Res<MaterialsHandle>,
    mut next_state: ResMut<NextState<GameState>>,
    prefab_configs: Res<PrefabConfigs>,
) {
    // Wait for materials, prefab images, and masks
    let mut all_loaded = asset_server.is_loaded(&materials.0);
    if all_loaded {
        for prefab in &prefab_configs.0 {
            let image_state = asset_server.get_load_state(&prefab.image);
            let mask_state = asset_server.get_load_state(&prefab.mask);
            if !matches!(image_state, Some(bevy::asset::LoadState::Loaded))
                || !matches!(mask_state, Some(bevy::asset::LoadState::Loaded))
            {
                all_loaded = false;
                break;
            }
        }
    }
    if all_loaded {
        next_state.set(GameState::Game);
    }
}

pub struct MaterialsPlugin;
impl Plugin for MaterialsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Materials>()
            .init_resource::<MaterialsHandle>()
            .init_asset_loader::<MaterialsLoader>()
            .add_systems(OnEnter(GameState::Loading), materials_setup)
            .add_systems(Update, loading_wait.run_if(in_state(GameState::Loading)));
    }
}
