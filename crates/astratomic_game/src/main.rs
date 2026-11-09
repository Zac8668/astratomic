use std::path::Path;

use astratomic_game::prelude::*;
use bones_framework::prelude::*;
use bones_wgpu_renderer::BonesWgpuRenderer;

fn main() {
    setup_logs!();

    let mut game = Game::new();
    game.install_plugin(AstratomicGamePlugin);

    let mut renderer = BonesWgpuRenderer::new(game)
        .namespace(("com", "spicylobstergames", "astratomic"))
        .version(Version::new(0, 3, 0))
        .pixel_art(true);

    if !renderer.asset_dir.exists() {
        let workspace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        renderer.asset_dir = workspace_dir.join("assets");
        renderer.packs_dir = workspace_dir.join("packs");
    }

    renderer.run();
}
