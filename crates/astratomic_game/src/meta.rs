//! Root metadata assets loaded by the Bones `AssetServer`.

use astratomic_sim::atom::Materials;
use bones_framework::prelude::*;

/// Root game asset loaded from `assets/game.yaml` (via `assets/pack.yaml`).
#[derive(HasSchema, Clone, Default)]
#[repr(C)]
#[type_data(metadata_asset("game"))]
pub struct GameMeta {
    pub player_atlas: Handle<Atlas>,
    pub materials: Handle<Materials>,
}
