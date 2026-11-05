use crate::prelude::*;

#[derive(Resource, Clone)]
pub struct QuaverFont(pub Handle<Font>);

impl FromWorld for QuaverFont {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>().clone();
        QuaverFont(asset_server.load("quaver.ttf"))
    }
}

pub struct QuaverFontPlugin;
impl Plugin for QuaverFontPlugin {
    fn build(&self, app: &mut App) {
        // Simple: always present, loaded once.
        app.init_resource::<QuaverFont>();
    }
}
