use std::collections::VecDeque;

use bevy::render::render_asset::{RenderAssetDependency, RenderAssetUsages, RenderAssets};
use bevy::render::render_resource::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, TextureDimension, TextureFormat,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Extract, RenderApp, RenderSet};
use bevy::sprite::Anchor;
use itertools::Itertools;
use smallvec::SmallVec;

use noise::{MultiFractal, RidgedMulti, SuperSimplex};

use crate::prelude::*;

#[derive(Resource, Default)]
pub struct WorldDb(pub Option<sled::Db>);

impl WorldDb {
    fn get_or_open(&mut self) -> &sled::Db {
        if self.0.is_none() {
            let db = open_world_db();
            self.0 = Some(db);
        }
        self.0.as_ref().unwrap()
    }
}

/// Updates and do the chunks logic
#[derive(Default, Resource)]
pub struct ChunkManager {
    pub chunks: HashMap<IVec2, Chunk>,
    pub pos: IVec2,
    pub dt: u8,
}

//If true the direction is 1, if false the direction is -1
#[derive(Debug)]
pub enum MoveDir {
    X(i32),
    Y(i32),
}

impl ChunkManager {
    pub fn get_atom(&self, pos: &ChunkPos) -> Option<&Atom> {
        if let Some(chunk) = self.chunks.get(&pos.chunk) {
            chunk.atoms.get(pos.atom.d1())
        } else {
            None
        }
    }

    pub fn get_mut_atom(&mut self, pos: ChunkPos) -> Option<&mut Atom> {
        if let Some(chunk) = self.chunks.get_mut(&pos.chunk) {
            chunk.atoms.get_mut(pos.atom.d1())
        } else {
            None
        }
    }

    //Still needs to save file chunks to file after this function is called
    pub fn move_manager(
        &mut self,
        commands: &mut Commands,
        images: &mut ResMut<Assets<Image>>,
        dirty_rects: &mut DirtyRects,
        prefabs: (&mut PrefabPlaced, &mut PendingPrefabStamps, &PrefabConfigs),
        entities: (&Entity, &Query<(&Parent, Entity, &Sprite)>),
        world_info: (&sled::Db, MoveDir, &Materials, &Generator),
    ) {
        let (prefab_placed, pending_prefab_stamps, prefab_configs) = prefabs;
        let (chunk_textures, image_entities) = entities;
        let (world_db, move_dir, materials, generator) = world_info;

        let (for_load, vec_load, dir, y) = match move_dir {
            MoveDir::X(dir) => {
                self.pos.x += dir;
                (LOAD_HEIGHT, LOAD_WIDTH, dir, false)
            }
            MoveDir::Y(dir) => {
                self.pos.y += dir;
                (LOAD_WIDTH, LOAD_HEIGHT, dir, true)
            }
        };

        let seed = generator.2;
        let min_y = self.pos.y * CHUNK_LENGHT as i32;
        let max_y = (self.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;
        let y_seed = (self.pos.y + LOAD_HEIGHT / 2) * CHUNK_LENGHT as i32;

        let mut images_vec = vec![];
        let mut to_remove = vec![];
        for i in 0..for_load {
            {
                //Save far chunks
                let mut to_add = ivec2(if dir == -1 { vec_load } else { -1 }, i);
                if y {
                    to_add = to_add.yx()
                }
                let pos = self.pos + to_add;

                let changed_chunk = self.chunks.remove(&pos).unwrap();
                to_remove.push(changed_chunk.texture.clone());
                images.remove(&changed_chunk.texture);

                save_chunk_to_db(world_db, pos, &changed_chunk);
            }

            {
                //Load new chunks
                let mut to_add = ivec2(if dir == 1 { vec_load - 1 } else { 0 }, i);
                if y {
                    to_add = to_add.yx()
                }
                let pos = self.pos + to_add;

                let from_file = load_chunk_from_db(world_db, pos).is_some();
                let chunk = if let Some(file_chunk) = load_chunk_from_db(world_db, pos) {
                    file_chunk
                } else {
                    Chunk::new(Handle::default(), pos, materials, generator)
                };

                self.add_chunk(pos, chunk);

                // If any previously queued prefab pixels target this chunk, apply them now.
                apply_pending_stamps_for_chunk(
                    self,
                    dirty_rects,
                    materials,
                    pending_prefab_stamps,
                    pos,
                );

                // Spawn prefabs only for freshly generated chunks.
                if !from_file {
                    let images_ref: &Assets<Image> = &*images;
                    for config in &prefab_configs.0 {
                        let Some((prefab_image, prefab_mask)) = prefab_images(images_ref, config)
                        else {
                            continue;
                        };

                        let xs = prefab_xs_for_chunk(config, pos.x, seed);
                        for x in xs {
                            let _ = stamp_prefab_at_x(
                                self,
                                dirty_rects,
                                (prefab_placed, pending_prefab_stamps),
                                materials,
                                (prefab_image, prefab_mask),
                                (x, y_seed),
                                (min_y, max_y),
                            );
                        }
                    }
                }

                if let Some(chunk) = self.chunks.get_mut(&pos) {
                    images_vec.push(add_chunk_image(commands, images, chunk, pos));
                }
            }
        }

        for (parent, ent, handle) in image_entities.iter() {
            if parent.get() == *chunk_textures && to_remove.contains(&handle.image) {
                commands.get_entity(ent).unwrap().despawn();
            }
        }
        let mut chunk_textures = commands.get_entity(*chunk_textures).unwrap();
        chunk_textures.insert_children(0, &images_vec);
    }

    pub fn add_chunk(&mut self, index: IVec2, chunk: Chunk) {
        self.chunks.insert(index, chunk);
    }
}

impl std::ops::Index<ChunkPos> for ChunkManager {
    type Output = Atom;
    #[track_caller]
    fn index(&self, pos: ChunkPos) -> &Self::Output {
        self.get_atom(&pos).expect("Invalid index position.")
    }
}
impl std::ops::IndexMut<ChunkPos> for ChunkManager {
    #[track_caller]
    fn index_mut(&mut self, pos: ChunkPos) -> &mut Self::Output {
        self.get_mut_atom(pos).expect("Invalid index position.")
    }
}

fn save_chunks_on_exit(
    mut exit_events: EventReader<AppExit>,
    chunk_manager: Res<ChunkManager>,
    world_db: ResMut<WorldDb>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    let Some(db) = world_db.0.as_ref() else {
        return;
    };

    for (pos, chunk) in &chunk_manager.chunks {
        save_chunk_to_db(db, *pos, chunk);
    }

    let _ = db.flush();
}

#[derive(Resource, Default)]
pub struct DirtyRects {
    /// The current chunk update dirty rects
    pub current: HashMap<IVec2, URect>,
    /// The new chunk update dirty rects
    pub new: HashMap<IVec2, URect>,
    /// The dirty render rects
    pub render: HashMap<IVec2, URect>,
}

impl DirtyRects {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.new)
    }
}

#[derive(Component)]
pub struct ChunksParent;

#[derive(Resource, Default)]
pub struct Generator(pub RidgedMulti<SuperSimplex>, pub f64, pub u32);

pub fn manager_setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut chunk_manager: ResMut<ChunkManager>,
    mut dirty_rects: ResMut<DirtyRects>,
    prefabs: (
        ResMut<PrefabPlaced>,
        ResMut<PendingPrefabStamps>,
        ResMut<PrefabConfigs>,
    ),
    materials: (Res<Assets<Materials>>, ResMut<MaterialsHandle>),
    world: (ResMut<Generator>, ResMut<WorldDb>),
) {
    let (mut prefab_placed, mut pending_prefab_stamps, mut prefab_configs) = prefabs;
    let (mut generator, mut world_db) = world;
    let (width, height) = (LOAD_WIDTH, LOAD_HEIGHT);

    let mut images_vec = vec![];
    chunk_manager.pos = ivec2(-16, -16);

    // Reset placed-prefab registry on fresh setup.
    prefab_placed.x_ranges.clear();
    pending_prefab_stamps.0.clear();

    let db = world_db.get_or_open();
    let is_new_world = !db_has_any_chunks(db);

    let bytes = std::fs::read("assets/gen.ron").unwrap();
    let gen_config = ron::de::from_bytes::<[f64; 6]>(&bytes).unwrap();

    generator.0 = RidgedMulti::<SuperSimplex>::new(generator.2)
        .set_octaves(gen_config[1] as usize)
        .set_lacunarity(gen_config[2])
        .set_persistence(gen_config[3])
        .set_attenuation(gen_config[4]);
    generator.1 = gen_config[0];
    //Seed
    generator.2 = gen_config[5] as u32;

    //Get materials (already loaded at this point).
    let materials = materials.0.get(&materials.1 .0).unwrap();

    // Adds chunks
    for (x, y) in (chunk_manager.pos.x..chunk_manager.pos.x + width)
        .cartesian_product(chunk_manager.pos.y..chunk_manager.pos.y + height)
    {
        let index = ivec2(x, y);
        let chunk;
        if let Some(file_chunk) = load_chunk_from_db(db, index) {
            chunk = file_chunk;
        } else {
            chunk = Chunk::new(Handle::default(), index, materials, &generator);
        }

        chunk_manager.add_chunk(index, chunk);
    }

    if is_new_world {
        // Adds prefabs on a new world only.
        // Clear positions before accumulating
        for config in &mut prefab_configs.0 {
            config.positions.clear();
        }

        for x in chunk_manager.pos.x..(chunk_manager.pos.x + width) {
            for config in &mut prefab_configs.0 {
                config
                    .positions
                    .extend(prefab_xs_for_chunk(&*config, x, gen_config[5] as u32));
            }
        }

        // Stable + deduped x positions per prefab type.
        for config in &mut prefab_configs.0 {
            config.positions.sort_unstable();
            config.positions.dedup();
        }

        // Stamp prefabs into the world, aligned to the nearest surface.
        // We seed the y-search from the middle of the currently loaded vertical range.
        {
            let images_ref: &Assets<Image> = &images;
            let min_y = chunk_manager.pos.y * CHUNK_LENGHT as i32;
            let max_y = (chunk_manager.pos.y + height) * CHUNK_LENGHT as i32 - 1;
            let y_seed = (chunk_manager.pos.y + height / 2) * CHUNK_LENGHT as i32;

            for config in &prefab_configs.0 {
                let Some((prefab_image, prefab_mask)) = prefab_images(images_ref, config) else {
                    continue;
                };

                for &x in &config.positions {
                    let _ = stamp_prefab_at_x(
                        &mut chunk_manager,
                        &mut dirty_rects,
                        (&mut prefab_placed, &mut pending_prefab_stamps),
                        materials,
                        (prefab_image, prefab_mask),
                        (x, y_seed),
                        (min_y, max_y),
                    );
                }
            }
        }
    }

    //Add Chunk images
    for (x, y) in (chunk_manager.pos.x..chunk_manager.pos.x + width)
        .cartesian_product(chunk_manager.pos.y..chunk_manager.pos.y + height)
    {
        let index = ivec2(x, y);

        let Some(chunk) = chunk_manager.chunks.get_mut(&index) else {
            continue;
        };
        let ent = add_chunk_image(&mut commands, &mut images, chunk, index);
        images_vec.push(ent);
    }

    commands
        .spawn((
            Name::new("Chunks"),
            Visibility::Visible,
            Transform::from_translation(vec3(0., 0., AUTOMATA_LAYER)),
            ChunksParent,
        ))
        .insert_children(0, &images_vec);
}

pub fn add_colliders(
    mut commands: Commands,
    chunk_manager: Res<ChunkManager>,
    chunks: Query<(Entity, &ChunkComponent), Without<Collider>>,
    has_collider: Res<HasCollider>,
) {
    puffin::profile_function!();

    if has_collider.0.is_empty() {
        return;
    }

    for (ent, pos) in &chunks {
        for rect in has_collider.0.iter() {
            if rect.contains(pos.0) {
                if let Some(chunk) = chunk_manager.chunks.get(&pos.0) {
                    let collider = chunk.get_collider();

                    if let Some(collider) = collider {
                        commands
                            .entity(ent)
                            .insert(collider)
                            .insert(bevy_rapier2d::prelude::RigidBody::Fixed);
                    }
                }

                break;
            }
        }
    }
}

pub fn remove_colliders(
    mut commands: Commands,
    chunks: Query<(Entity, &ChunkComponent), With<Collider>>,
    has_collider: Res<HasCollider>,
) {
    if has_collider.0.is_empty() {
        return;
    }

    for (ent, pos) in &chunks {
        let mut contains = false;
        for rect in has_collider.0.iter() {
            if rect.contains(pos.0) {
                contains = true;
                break;
            }
        }

        if !contains {
            //If none contains, remove collider and go to next chunk
            //Remove collider
            if let Some(mut entity) = commands.get_entity(ent) {
                entity.remove::<Collider>();
                entity.remove::<bevy_rapier2d::prelude::RigidBody>();
            }
        }
    }
}

pub fn update_has_collider(
    mut has_collider: ResMut<HasCollider>,
    rigidbodies: Query<(&Transform, &Rigidbody)>,
) {
    has_collider.0 = vec![];
    for (transform, rigidbody) in &rigidbodies {
        let angle = -transform.rotation.to_euler(EulerRot::XYZ).2;
        let mut center = transform.translation.xy();
        center.y *= -1.;
        center += vec2(rigidbody.width as f32, rigidbody.height as f32)
            .rotate(Vec2::from_angle(angle))
            / 2.;

        let l = std::f32::consts::SQRT_2 * (rigidbody.width as f32).max(rigidbody.height as f32);
        let angle = std::f32::consts::FRAC_PI_4 + std::f32::consts::PI;
        let mut top_left = center;
        top_left += vec2(angle.cos(), angle.sin()) * l / 2.;

        let angle = std::f32::consts::FRAC_PI_4;
        let mut down_right = center;
        down_right += vec2(angle.cos(), angle.sin()) * l / 2.;

        /*{
            //Some debug visualization
            let mut top_left = top_left;
            top_left.y *= -1.;
            gizmos.circle_2d(top_left, 5., Color::RED);

            let mut down_right = down_right;
            down_right.y *= -1.;
            gizmos.circle_2d(down_right, 5., Color::BLACK);
        }*/

        const LOADING_OFF: f32 = 1.5;
        top_left -= vec2(CHUNK_LENGHT as f32, CHUNK_LENGHT as f32) * LOADING_OFF;
        down_right += vec2(CHUNK_LENGHT as f32, CHUNK_LENGHT as f32) * LOADING_OFF;

        top_left /= CHUNK_LENGHT as f32;
        down_right /= CHUNK_LENGHT as f32;

        let bounds_rect = IRect::new(
            top_left.x as i32,
            top_left.y as i32,
            down_right.x as i32,
            down_right.y as i32,
        );

        has_collider.0.push(bounds_rect);
    }
}

pub fn chunk_manager_update(
    mut chunk_manager: ResMut<ChunkManager>,
    mut dirty_rects_resource: ResMut<DirtyRects>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    puffin::profile_function!();

    chunk_manager.dt = chunk_manager.dt.wrapping_add(1);
    let dt = chunk_manager.dt;

    // Get dirty rects
    let DirtyRects {
        current: dirty_rects,
        new: new_dirty_rects,
        render: render_dirty_rects,
    } = &mut *dirty_rects_resource;

    //Get materials
    let materials = &materials.0.get(&materials.1 .0).unwrap();

    let manager_pos = ivec2(chunk_manager.pos.x, chunk_manager.pos.y);

    let compute_pool = ComputeTaskPool::get();

    // Create channel for sending dirty update rects
    let (dirty_update_rects_send, dirty_update_rects_recv) =
        async_channel::unbounded::<DeferredDirtyRectUpdate>();
    let dirty_update_rect_send = &dirty_update_rects_send;

    // Create channel for sending dirty render rect updates
    let (dirty_render_rects_send, dirty_render_rects_recv) =
        async_channel::unbounded::<DeferredDirtyRectUpdate>();
    let dirty_render_rect_send = &dirty_render_rects_send;

    // Create a scope in which we handle deferred updates and update chunks.
    compute_pool.scope(|deferred_scope| {
        // Spawn a task on the deferred scope for handling the deferred dirty update rects.
        deferred_scope.spawn(async move {
            // Clear the new dirty rects so we can update a fresh list
            new_dirty_rects.clear();

            // Loop through deferred tasks
            while let Ok(update) = dirty_update_rects_recv.recv().await {
                if update.awake_surrouding {
                    update_dirty_rects_3x3(new_dirty_rects, update.chunk_pos);
                } else {
                    update_dirty_rects(new_dirty_rects, update.chunk_pos)
                }
            }
        });

        // Spawn a task on the deferred scope for handling deferred dirty render rects.
        deferred_scope.spawn(async move {
            // Loop through deferred tasks
            while let Ok(update) = dirty_render_rects_recv.recv().await {
                update_dirty_rects(render_dirty_rects, update.chunk_pos);
            }
        });

        // Run the 4 update steps in checker like pattern
        for (y_toff, x_toff) in rand_range(0..2)
            .into_iter()
            .cartesian_product(rand_range(0..2))
        {
            puffin::profile_scope!("Update step scope.");

            compute_pool.scope(|scope| {
                update_chunk_groups(
                    &mut chunk_manager.chunks,
                    (x_toff, y_toff),
                    dirty_rects,
                    manager_pos,
                    (dirty_update_rect_send, dirty_render_rect_send),
                    (dt, materials),
                    scope,
                );
            });
        }

        // Close the deferred updates channel so that our deferred update task will complete.
        dirty_update_rect_send.close();
        dirty_render_rect_send.close();
    });

    // Once we are done with our updates, swap the new dirty rects to the current one.
    dirty_rects_resource.swap();
}

pub fn update_chunks(chunks: &mut UpdateChunksType, dt: u8, dirty_rect: &URect) {
    puffin::profile_function!();

    let materials = chunks.materials;

    let x_iter = rand_range(dirty_rect.min.x as i32..dirty_rect.max.x as i32 + 1).into_iter();
    let y_iter = rand_range(dirty_rect.min.y as i32..dirty_rect.max.y as i32 + 1).into_iter();
    for (y, x) in y_iter.cartesian_product(x_iter) {
        let local_pos = (ivec2(x, y), 4);
        let pos = local_to_global(local_pos);

        if !dt_updatable(chunks, pos, dt) {
            continue;
        }

        let mut awake_self = false;
        let id;
        let speed;
        let state;
        {
            let atom = &mut chunks.group[local_pos];
            id = atom.id;
            speed = atom.speed;
            state = atom.state;

            if atom.f_idle < FRAMES_SLEEP && !atom.is_void() && !atom.is_solid() {
                atom.f_idle += 1;
                awake_self = true;
            }
        }

        let (vector, mut awakened) = if speed.0 == 0 && speed.1 >= 0 {
            (
                false,
                match state {
                    AtomState::Powder => {
                        update_powder(chunks, pos, dt, materials[id].inertial_resistance)
                    }
                    AtomState::Liquid => update_liquid(chunks, pos, materials[id].flow, dt),
                    _ => HashSet::new(),
                },
            )
        } else {
            (true, update_atom(chunks, pos, dt))
        };

        let atom = &mut chunks.group[local_pos];
        let mut self_awakened = HashSet::new();
        if awakened.contains(&pos) {
            atom.f_idle = 0;
            atom.moving = true;
        } else if vector {
            atom.f_idle = 0;
            awakened.insert(pos);
            atom.moving = false;
        } else if awake_self {
            awakened.insert(pos);
            self_awakened.insert(pos);
            atom.moving = false;
        }

        for awoke in awakened {
            let local = global_to_local(awoke);
            let chunk = ChunkGroup::group_to_chunk(chunks.group.center_pos, local.1);

            chunks
                .dirty_update_rect_send
                .try_send(DeferredDirtyRectUpdate {
                    chunk_pos: ChunkPos::new(local.0.try_into().unwrap(), chunk),
                    awake_surrouding: !self_awakened.contains(&awoke),
                })
                .unwrap();
        }
    }
}

#[derive(Component)]
pub struct ChunkComponent(pub IVec2);

//Still needs to add the return entity to a parent
pub fn add_chunk_image(
    commands: &mut Commands,
    images: &mut ResMut<Assets<Image>>,
    chunk: &mut Chunk,
    index: IVec2,
) -> Entity {
    let pos = Vec2::new(
        index.x as f32 * CHUNK_LENGHT as f32,
        (-index.y as f32) * CHUNK_LENGHT as f32,
    );

    //Add texture
    chunk.texture = images.add(Chunk::new_image());
    let texture_copy = chunk.texture.clone();

    //Update chunk image
    let image = images.get_mut(&chunk.texture).unwrap();
    chunk.update_all(image);

    //Spawn Image
    let entity = commands
        .spawn((
            Sprite {
                image: texture_copy,

                anchor: Anchor::TopLeft,

                ..Default::default()
            },
            ChunkComponent(index),
        ))
        .insert(Transform::from_xyz(pos.x, pos.y, 0.))
        .id();
    chunk.entity = Some(entity);

    entity
}

pub fn update_manager_pos(
    mut commands: Commands,
    chunk_textures: Query<Entity, With<ChunksParent>>,
    image_entities: Query<(&Parent, Entity, &Sprite)>,
    player: Query<&Actor, With<Player>>,
    resources: (
        ResMut<SavingTask>,
        ResMut<ChunkManager>,
        ResMut<Assets<Image>>,
        Res<Generator>,
        ResMut<DirtyRects>,
        ResMut<PrefabPlaced>,
        ResMut<PendingPrefabStamps>,
        Res<PrefabConfigs>,
        ResMut<WorldDb>,
    ),
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    let materials = materials.0.get(&materials.1 .0).unwrap();

    let (
        mut saving_task,
        mut chunk_manager,
        mut images,
        generator,
        mut dirty_rects,
        mut prefab_placed,
        mut pending_prefab_stamps,
        prefab_configs,
        mut world_db,
    ) = resources;

    let db = world_db.get_or_open().clone();

    let mut player_pos = player.single().pos;
    if player_pos.x < 0 {
        player_pos.x -= CHUNK_LENGHT as i32
    }
    if player_pos.y < 0 {
        player_pos.y -= CHUNK_LENGHT as i32
    }
    player_pos /= CHUNK_LENGHT as i32;

    let diff_x = player_pos.x - chunk_manager.pos.x - LOAD_WIDTH / 2;
    let diff_y = player_pos.y - chunk_manager.pos.y - LOAD_HEIGHT / 2;
    let new_diff = ivec2(diff_x, diff_y);

    // If a flush is in progress, don't queue another one.
    if let Some(task) = &saving_task.0 {
        if task.is_finished() {
            saving_task.0 = None;
        }
    }

    // Apply at most one move step per frame to avoid hitches.
    const MAX_STEPS_PER_FRAME: i32 = 1;
    let mut diff = new_diff;
    if diff == IVec2::ZERO {
        // Nothing to stream; ensure any pending writes hit disk.
        if saving_task.0.is_none() {
            let pool = AsyncComputeTaskPool::get();
            let db_for_task = db.clone();
            saving_task.0 = Some(pool.spawn(async move {
                let _ = db_for_task.flush_async().await;
            }));
        }
        return;
    }

    let chunk_textures = chunk_textures.single();
    for _ in 0..MAX_STEPS_PER_FRAME {
        if diff == IVec2::ZERO {
            break;
        }

        if diff.x != 0 {
            chunk_manager.move_manager(
                &mut commands,
                &mut images,
                &mut dirty_rects,
                (
                    &mut prefab_placed,
                    &mut pending_prefab_stamps,
                    &prefab_configs,
                ),
                (&chunk_textures, &image_entities),
                (&db, MoveDir::X(diff.x.signum()), materials, &generator),
            );
            diff.x -= diff.x.signum();
        } else if diff.y != 0 {
            chunk_manager.move_manager(
                &mut commands,
                &mut images,
                &mut dirty_rects,
                (
                    &mut prefab_placed,
                    &mut pending_prefab_stamps,
                    &prefab_configs,
                ),
                (&chunk_textures, &image_entities),
                (&db, MoveDir::Y(diff.y.signum()), materials, &generator),
            );
            diff.y -= diff.y.signum();
        }
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ExtractedTextureUpdates(pub Vec<ExtractedTextureUpdate>);

#[derive(Clone)]
pub struct ExtractedTextureUpdate {
    pub id: AssetId<Image>,
    // TODO: determine a good size for the data smallvec array.
    // The size of the array determines how many bytes we can store before it overflows and has
    // to make a heap allocation. 256 is enough to store an 8x8 pixel dirty rect.
    pub data: SmallVec<[u8; 256]>,
    pub origin: Origin3d,
    pub size: Extent3d,
}

fn extract_chunk_texture_updates(
    chunk_manager: Extract<Res<ChunkManager>>,
    dirty_rects: Extract<Res<DirtyRects>>,
    mut extracted_updates: ResMut<ExtractedTextureUpdates>,
) {
    for (chunk_pos, chunk) in chunk_manager.chunks.iter() {
        if let Some(rect) = dirty_rects.render.get(chunk_pos) {
            let id = chunk.texture.id();
            let mut data = SmallVec::new();

            for y in rect.min.y..=rect.max.y {
                for x in rect.min.x..=rect.max.x {
                    let pos = IVec2::new(x as i32, y as i32);
                    let color = chunk.atoms[pos.d1()].color;
                    data.extend_from_slice(&color)
                }
            }

            extracted_updates.push(ExtractedTextureUpdate {
                id,
                data,
                origin: Origin3d {
                    x: rect.min.x,
                    y: rect.min.y,
                    z: 0,
                },
                size: Extent3d {
                    width: rect.width() + 1,
                    height: rect.height() + 1,
                    depth_or_array_layers: 1,
                },
            });
        }
    }
}

fn prepare_chunk_gpu_textures(
    queue: Res<RenderQueue>,
    image_render_assets: Res<RenderAssets<GpuImage>>,
    mut extracted_updates: ResMut<ExtractedTextureUpdates>,
) {
    for update in extracted_updates.drain(..) {
        let Some(gpu_image) = image_render_assets.get(update.id) else {
            continue;
        };

        queue.write_texture(
            ImageCopyTexture {
                texture: &gpu_image.texture,
                mip_level: 0,
                origin: update.origin,
                aspect: bevy::render::render_resource::TextureAspect::All,
            },
            &update.data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(update.size.width * 4),
                rows_per_image: None,
            },
            update.size,
        );
    }
}

fn clear_render_rect(mut dirty_rects: ResMut<DirtyRects>) {
    dirty_rects.render = HashMap::new();
}

#[derive(Resource, Default)]
pub struct HasCollider(pub Vec<IRect>);

#[derive(Resource, Default)]
pub struct BrokenAtoms(pub HashSet<IVec2>);

const DETACH_MAX_CELLS: usize = 5000;
const DETACH_MAX_SPAN: i32 = 128;
const DETACH_MIN_CELLS: usize = 16;

#[derive(Debug, Default)]
struct BlobFloodResult {
    cells: Vec<IVec2>,
    min: IVec2,
    max: IVec2,
    touches_anchor: bool,
    touches_unknown: bool,
    anchored_by_threshold: bool,
}

fn flood_fill_blob_component(
    chunk_manager: &ChunkManager,
    start: IVec2,
) -> Option<BlobFloodResult> {
    let start_atom = chunk_manager.get_atom(&global_to_chunk(start))?;
    if !blobable(start_atom) {
        return None;
    }

    let mut out = BlobFloodResult {
        min: start,
        max: start,
        ..Default::default()
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    let neighbor_offsets = [
        IVec2::new(1, 0),
        IVec2::new(-1, 0),
        IVec2::new(0, 1),
        IVec2::new(0, -1),
    ];

    while let Some(pos) = queue.pop_front() {
        out.cells.push(pos);

        out.min.x = out.min.x.min(pos.x);
        out.min.y = out.min.y.min(pos.y);
        out.max.x = out.max.x.max(pos.x);
        out.max.y = out.max.y.max(pos.y);

        if out.cells.len() > DETACH_MAX_CELLS {
            out.anchored_by_threshold = true;
            break;
        }
        if (out.max.x - out.min.x) > DETACH_MAX_SPAN || (out.max.y - out.min.y) > DETACH_MAX_SPAN {
            out.anchored_by_threshold = true;
            break;
        }

        for off in neighbor_offsets {
            let neigh = pos + off;
            if visited.contains(&neigh) {
                continue;
            }

            match chunk_manager.get_atom(&global_to_chunk(neigh)) {
                Some(atom) if blobable(atom) => {
                    visited.insert(neigh);
                    queue.push_back(neigh);
                }
                Some(atom) => {
                    // Neighbor is not part of this blob component.
                    // If it's not air, the boundary isn't "air-only".
                    if !(atom.is_void() || atom.is_liquid() || atom.is_powder()) {
                        out.touches_anchor = true;
                    }
                }
                None => {
                    // Unloaded / missing chunk: we can't prove this blob is isolated.
                    out.touches_unknown = true;
                }
            }
        }
    }

    Some(out)
}

fn spawn_detached_rigidbody(
    commands: &mut Commands,
    images: &mut ResMut<Assets<Image>>,
    chunk_manager: &mut ChunkManager,
    dirty_rects: &mut DirtyRects,
    blob: &BlobFloodResult,
) {
    let size = blob.max - blob.min + IVec2::ONE;
    let width = size.x as usize;
    let height = size.y as usize;
    if width == 0 || height == 0 {
        return;
    }
    if width > u8::MAX as usize || height > u8::MAX as usize {
        return;
    }

    let mut atoms = vec![Atom::default(); width * height];
    let mut values = vec![0.0_f64; width * height];
    let mut image_data = vec![0_u8; width * height * 4];

    let mut affected_chunks = HashSet::new();

    for pos in &blob.cells {
        let chunk_pos = global_to_chunk(*pos);
        let Some(atom) = chunk_manager.get_atom(&chunk_pos) else {
            continue;
        };
        if !blobable(atom) {
            continue;
        }

        let lx = (pos.x - blob.min.x) as usize;
        let ly = (pos.y - blob.min.y) as usize;
        let i = ly * width + lx;

        atoms[i] = *atom;
        values[i] = 1.0;
        let pixel_index = i * 4;
        image_data[pixel_index..pixel_index + 4].copy_from_slice(&atom.color);
    }

    let image = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        image_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    let image_handle = images.add(image);

    let collider = get_collider(&values, width as u32, height as u32);
    let Some(collider) = collider else {
        return;
    };

    // Remove blob cells from world and mark dirty rects.
    for pos in &blob.cells {
        let chunk_pos = global_to_chunk(*pos);
        let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) else {
            continue;
        };

        if blobable(atom) {
            *atom = Atom::default();
            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
            affected_chunks.insert(chunk_pos.chunk);
        }
    }

    for chunk in affected_chunks {
        if let Some(chunk) = chunk_manager.chunks.get(&chunk) {
            if let Some(ent) = chunk.entity {
                commands.entity(ent).remove::<Collider>();
            }
        }
    }

    let rigidbody = Rigidbody {
        atoms,
        width: width as u8,
        height: height as u8,
        filled: vec![],
        ..Default::default()
    };

    commands
        .spawn(collider)
        .insert(rigidbody)
        .insert(RapierRigidbody::Dynamic)
        .insert(bevy_rapier2d::prelude::Velocity::zero())
        .insert(bevy_rapier2d::prelude::ExternalImpulse::default())
        .insert(ReadMassProperties::default())
        .insert(Sprite {
            image: image_handle,
            anchor: bevy::sprite::Anchor::TopLeft,
            ..Default::default()
        })
        .insert(Transform::from_xyz(
            blob.min.x as f32,
            -(blob.min.y as f32),
            RIGIDBODY_LAYER,
        ));
}

pub fn check_connectivity(
    mut commands: Commands,
    mut broken_atoms: ResMut<BrokenAtoms>,
    mut chunk_manager: ResMut<ChunkManager>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut images: ResMut<Assets<Image>>,
) {
    if broken_atoms.0.is_empty() {
        return;
    }

    let broken_positions: Vec<IVec2> = broken_atoms.0.drain().collect();
    let neighbor_offsets = [
        IVec2::new(1, 0),
        IVec2::new(-1, 0),
        IVec2::new(0, 1),
        IVec2::new(0, -1),
    ];

    let mut processed_seeds = HashSet::new();

    for broken_pos in broken_positions {
        for off in neighbor_offsets {
            let seed = broken_pos + off;
            if processed_seeds.contains(&seed) {
                continue;
            }

            let flood = {
                let cm: &ChunkManager = &chunk_manager;
                flood_fill_blob_component(cm, seed)
            };

            let Some(flood) = flood else {
                continue;
            };

            for p in &flood.cells {
                processed_seeds.insert(*p);
            }

            // Anchor rules:
            // - Too big (count or span) => anchored
            // - Boundary touches anything non-void => anchored
            // - Touches unloaded area => anchored (can't prove isolation)
            // - Too small => ignore
            if flood.cells.len() < DETACH_MIN_CELLS {
                continue;
            }
            if flood.anchored_by_threshold || flood.touches_anchor || flood.touches_unknown {
                continue;
            }

            spawn_detached_rigidbody(
                &mut commands,
                &mut images,
                &mut chunk_manager,
                &mut dirty_rects,
                &flood,
            );
        }
    }
}

pub fn blobable(atom: &Atom) -> bool {
    //Add some material constraints too maybe
    atom.is_solid()
}

pub struct ChunkManagerPlugin;
impl Plugin for ChunkManagerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Game), manager_setup)
            .add_systems(
                FixedUpdate,
                chunk_manager_update.run_if(in_state(GameState::Game)),
            )
            .add_systems(Last, save_chunks_on_exit)
            .add_systems(
                Update,
                (
                    update_manager_pos,
                    add_colliders.after(update_has_collider),
                    remove_colliders.after(update_has_collider),
                    update_has_collider,
                )
                    .run_if(in_state(GameState::Game)),
            )
            .add_systems(
                PreUpdate,
                clear_render_rect.run_if(in_state(GameState::Game)),
            )
            .add_systems(
                PostUpdate,
                check_connectivity.run_if(in_state(GameState::Game)),
            )
            .init_resource::<HasCollider>()
            .init_resource::<ChunkManager>()
            .init_resource::<DirtyRects>()
            .init_resource::<Generator>()
            .init_resource::<PrefabPlaced>()
            .init_resource::<PendingPrefabStamps>()
            .init_resource::<WorldDb>()
            .init_resource::<BrokenAtoms>();

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedTextureUpdates>()
                .add_systems(ExtractSchedule, extract_chunk_texture_updates);
            GpuImage::register_system(
                render_app,
                prepare_chunk_gpu_textures.in_set(RenderSet::PrepareAssets),
            )
        }
    }
}
