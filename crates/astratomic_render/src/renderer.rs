use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use astratomic_sim::{
    chunk::{CHUNK_SIZE, Chunk},
    world::{ChunkWorld, DirtyRects},
};
use bones_framework::prelude::*;

pub const CHUNK_LAYER_Z: f32 = 0.0;

static NEXT_CHUNK_CID: AtomicU64 = AtomicU64::new(1);

fn next_chunk_cid() -> Cid {
    let id = NEXT_CHUNK_CID.fetch_add(1, Ordering::Relaxed);
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(b"ASTRCHNK");
    bytes[8..16].copy_from_slice(&id.to_le_bytes());
    Cid(bytes)
}

/// Finds `bones_wgpu_renderer::LoadedEguiAssets` in the global schema registry so chunk textures
/// can be marked as already processed, preventing `bones_wgpu_renderer` from uploading 64x64
/// simulation chunk textures into `egui`'s GPU texture manager every frame.
fn loaded_egui_assets_schema() -> Option<&'static Schema> {
    static SCHEMA: OnceLock<&'static Schema> = OnceLock::new();
    if let Some(&schema) = SCHEMA.get() {
        return Some(schema);
    }
    type ExpectedMap = HashMap<UntypedHandle, Cid>;
    let found = SCHEMA_REGISTRY.schemas.iter().find(|s| {
        s.full_name.ends_with("LoadedEguiAssets")
            && s.layout().size() == std::mem::size_of::<ExpectedMap>()
            && s.layout().align() == std::mem::align_of::<ExpectedMap>()
    });
    if let Some(schema) = found {
        let _ = SCHEMA.set(schema);
    }
    found
}

fn with_loaded_egui_assets_cids(world: &World, f: impl FnOnce(&mut HashMap<UntypedHandle, Cid>)) {
    let Some(schema) = loaded_egui_assets_schema() else {
        return;
    };
    if !world.resources.untyped().contains(schema.id()) {
        return;
    };
    let cell = world.resources.untyped().get(schema);
    let mut borrow = cell.borrow_mut();
    if let Some(schema_box) = borrow.as_mut() {
        // SAFETY: `loaded_egui_assets_schema` verified that `LoadedEguiAssets` is a single-field
        // wrapper around `HashMap<UntypedHandle, Cid>` with matching size and alignment.
        let cids = unsafe {
            &mut *schema_box
                .as_mut()
                .as_ptr()
                .cast::<HashMap<UntypedHandle, Cid>>()
        };
        f(cids);
    }
}

/// Marker component attached to each rendered chunk sprite entity.
#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ChunkSprite {
    pub chunk_pos: IVec2,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkRenderEntry {
    pub entity: Entity,
    pub handle: Handle<Image>,
    pub cid: Cid,
}

/// Tracks active chunk `Sprite` entities and their dynamic `AssetServer` `Image` handles.
#[derive(HasSchema, Clone, Default, Debug)]
#[schema(opaque)]
pub struct ChunkRenderState {
    pub entries: HashMap<IVec2, ChunkRenderEntry>,
}

/// Registers rendering schemas with the global `bones_schema` registry.
pub fn register_render_schemas() {
    ChunkSprite::register_schema();
    ChunkRenderState::register_schema();
    crate::particles::register_particle_render_schemas();
}

/// Bones plugin that synchronizes `ChunkWorld` chunks and dirty rects to `Sprite` textures.
#[derive(Default, Clone, Copy)]
pub struct AstratomicRenderPlugin;

impl GamePlugin for AstratomicRenderPlugin {
    fn install(self, _game: &mut Game) {
        register_render_schemas();
    }
}

impl SessionPlugin for AstratomicRenderPlugin {
    fn install(self, session: &mut Session) {
        register_render_schemas();
        session.world.init_resource::<ChunkRenderState>();
        session.world.init_resource::<crate::particles::ParticleRenderState>();
        session.world.init_resource::<crate::particles::ParticleRenderConfig>();
        session.add_system_to_stage(CoreStage::PostUpdate, sync_chunk_sprites_system);
        session.add_system_to_stage(CoreStage::PostUpdate, crate::particles::render_particles_system);
    }
}

/// PostUpdate system that spawns/despawns chunk `Sprite` entities and uploads dirty chunk rects
/// into `AssetServer` so `bones_wgpu_renderer` updates its GPU sprite atlas.
pub fn sync_chunk_sprites_system(
    world: &World,
    asset_server: Option<Res<AssetServer>>,
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut sprites: CompMut<Sprite>,
    mut chunk_sprites: CompMut<ChunkSprite>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut render_state: ResMut<ChunkRenderState>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };

    let mut removed_egui_handles = Vec::new();
    let mut updated_egui_handles = Vec::new();

    // 1. Despawn chunk sprites for chunks that were unloaded or removed.
    let stale_positions: Vec<IVec2> = render_state
        .entries
        .keys()
        .copied()
        .filter(|pos| !chunk_world.chunks.contains_key(pos))
        .collect();

    for pos in stale_positions {
        if let Some(removed) = render_state.entries.remove(&pos) {
            entities.kill(removed.entity);
            let untyped = removed.handle.untyped();
            asset_server.store.asset_ids.remove(&untyped);
            asset_server.store.assets.remove(&removed.cid);
            removed_egui_handles.push(untyped);
        }
    }

    // 2. Spawn sprites for newly added chunks and update textures for dirty chunks.
    let half_chunk = CHUNK_SIZE as f32 * 0.5;
    for (&chunk_pos, chunk) in chunk_world.chunks.iter_mut() {
        if let Some(entry) = render_state.entries.get_mut(&chunk_pos) {
            if let Some(rect) = dirty_rects.render.get(&chunk_pos) {
                let old_cid = entry.cid;
                let new_cid = next_chunk_cid();

                if let Some((_, mut loaded_asset)) = asset_server.store.assets.remove(&old_cid) {
                    let image = loaded_asset.data.cast_mut::<Image>();
                    chunk.update_rect(image, rect);
                    loaded_asset.cid = new_cid;
                    asset_server.store.assets.insert(new_cid, loaded_asset);
                } else {
                    let mut image = Chunk::new_image();
                    chunk.update_all(&mut image);
                    asset_server.store.assets.insert(
                        new_cid,
                        LoadedAsset {
                            cid: new_cid,
                            pack_spec: None,
                            loc: AssetLoc {
                                path: PathBuf::new(),
                                pack: None,
                            },
                            dependencies: Vec::new(),
                            data: SchemaBox::new(image),
                        },
                    );
                }

                let untyped = entry.handle.untyped();
                asset_server.store.asset_ids.insert(untyped, new_cid);
                updated_egui_handles.push((untyped, new_cid));
                entry.cid = new_cid;
            }
        } else {
            let mut image = Chunk::new_image();
            chunk.update_all(&mut image);

            let handle = UntypedHandle {
                rid: Ulid::create(),
            }
            .typed::<Image>();
            let untyped = handle.untyped();
            let cid = next_chunk_cid();

            asset_server.store.assets.insert(
                cid,
                LoadedAsset {
                    cid,
                    pack_spec: None,
                    loc: AssetLoc {
                        path: PathBuf::new(),
                        pack: None,
                    },
                    dependencies: Vec::new(),
                    data: SchemaBox::new(image),
                },
            );
            asset_server.store.asset_ids.insert(untyped, cid);
            updated_egui_handles.push((untyped, cid));

            let entity = entities.create();
            let center = Vec3::new(
                chunk_pos.x as f32 * CHUNK_SIZE as f32 + half_chunk,
                -(chunk_pos.y as f32 * CHUNK_SIZE as f32) - half_chunk,
                CHUNK_LAYER_Z,
            );
            transforms.insert(entity, Transform::from_translation(center));
            sprites.insert(
                entity,
                Sprite {
                    color: Color::WHITE,
                    image: handle,
                    flip_x: false,
                    flip_y: false,
                },
            );
            chunk_sprites.insert(entity, ChunkSprite { chunk_pos });

            chunk.texture = handle;
            chunk.entity = Some(entity);

            render_state.entries.insert(
                chunk_pos,
                ChunkRenderEntry {
                    entity,
                    handle,
                    cid,
                },
            );
        }
    }

    if !removed_egui_handles.is_empty() || !updated_egui_handles.is_empty() {
        with_loaded_egui_assets_cids(world, |cids| {
            for handle in removed_egui_handles {
                cids.remove(&handle);
            }
            for (handle, cid) in updated_egui_handles {
                cids.insert(handle, cid);
            }
        });
    }

    dirty_rects.render.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use astratomic_sim::{
        atom::{Atom, AtomState},
        plugin::AstratomicSimPlugin,
        world::URect,
    };

    #[test]
    fn sync_chunk_sprites_spawns_updates_and_despawns() {
        let mut game = Game::new();
        game.install_plugin(DefaultGamePlugin);
        game.init_shared_resource::<AssetServer>()
            .register_default_assets();
        game.install_plugin(AstratomicSimPlugin);
        game.install_plugin(AstratomicRenderPlugin);

        game.sessions.create_with("test", |session: &mut Session| {
            session
                .install_plugin(DefaultSessionPlugin)
                .install_plugin(AstratomicSimPlugin)
                .install_plugin(AstratomicRenderPlugin);
        });

        {
            let session = game.sessions.get_mut("test").unwrap();
            let mut cw = session.world.resource_mut::<ChunkWorld>();
            let mut dr = session.world.resource_mut::<DirtyRects>();
            let mut chunk = Chunk::default();
            chunk.atoms_mut()[0] = Atom::with_state(2, AtomState::Powder, [230, 197, 92, 255]);
            cw.add_chunk(ivec2(0, 0), chunk);
            dr.render.insert(ivec2(0, 0), URect::full_chunk());
        }

        // Run sync_chunk_sprites_system via game shared resources
        {
            let asset_server: AssetServer = (*game.shared_resource::<AssetServer>()).clone();
            let session = game.sessions.get_mut("test").unwrap();
            session.world.insert_resource(asset_server);
            session.world.run_system(sync_chunk_sprites_system, ());
        }

        let initial_cid = {
            let session = game.sessions.get("test").unwrap();
            let rs = session.world.resource::<ChunkRenderState>();
            let entry = rs
                .entries
                .get(&ivec2(0, 0))
                .expect("chunk sprite should exist");
            assert!(session.world.resource::<Entities>().is_alive(entry.entity));
            entry.cid
        };

        // Mark chunk dirty and verify Cid updates
        {
            let session = game.sessions.get_mut("test").unwrap();
            session
                .world
                .resource_mut::<DirtyRects>()
                .render
                .insert(ivec2(0, 0), URect::new(0, 0, 1, 1));
            session.world.run_system(sync_chunk_sprites_system, ());
        }

        {
            let session = game.sessions.get("test").unwrap();
            let rs = session.world.resource::<ChunkRenderState>();
            let entry = rs.entries.get(&ivec2(0, 0)).unwrap();
            assert_ne!(
                entry.cid, initial_cid,
                "dirty rect should advance chunk Cid"
            );
        }

        // Remove chunk and verify entity + asset cleanup
        {
            let session = game.sessions.get_mut("test").unwrap();
            session
                .world
                .resource_mut::<ChunkWorld>()
                .remove_chunk(ivec2(0, 0));
            session.world.run_system(sync_chunk_sprites_system, ());
            let rs = session.world.resource::<ChunkRenderState>();
            assert!(rs.entries.is_empty());
        }
    }
}
