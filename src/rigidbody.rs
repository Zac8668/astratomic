use crate::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::*;
use density_mesh_core::{
    generator::DensityMeshGenerator,
    map::DensityMap,
    mesh::{points_separation::PointsSeparation, settings::GenerateDensityMeshSettings},
};
use itertools::Itertools;
use smallvec::{SmallVec, ToSmallVec};

#[derive(Component)]
pub struct Rigidbody {
    pub atoms: Vec<Atom>,
    pub width: u8,
    pub height: u8,
    // Positions to remove Object atom later
    pub filled: Vec<ChunkPos>,
    pub texture: Handle<Image>,
    pub texture_ent: Entity,
    pub text_update: Option<ExtractedTextureUpdate>,
}

impl Rigidbody {
    pub fn texture_lenght(&self) -> u32 {
        const OFF: f32 = 1.2;
        ((self.width as f32 * OFF).powi(2) + (self.height as f32 * OFF).powi(2)).sqrt() as u32
    }
}

#[derive(Component, Default)]
pub struct RigidbodyHandle(pub Handle<Image>);

#[derive(Component)]
pub struct Hydrated;

//TODO Add rigidbody file
pub fn load_images(mut commands: Commands, server: Res<AssetServer>) {
    let image: Handle<Image> = server.load("tree.png");
    commands.spawn(RigidbodyHandle(image.clone()));
}

pub fn add_rigidbodies(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    handles: Query<(Entity, &RigidbodyHandle), Without<Hydrated>>,
) {
    for (ent, handle) in &handles {
        let image = images.get(handle.0.clone()).unwrap().clone();

        let lenght = image.width().max(image.height()) * 2;
        let new_image = Image::new(
            Extent3d {
                height: lenght,
                width: lenght,
                ..Default::default()
            },
            TextureDimension::D2,
            vec![0; (lenght as usize).pow(2) * 4],
            TextureFormat::Rgba8UnormSrgb,
        );
        let texture = images.add(new_image);

        let texture_ent = commands
            .spawn(SpriteBundle {
                texture: texture.clone(),
                sprite: Sprite {
                    anchor: bevy::sprite::Anchor::TopLeft,
                    ..Default::default()
                },
                ..Default::default()
            })
            .id();

        let rigidbody = Rigidbody {
            atoms: image_atoms(&image),
            height: image.height() as u8,
            width: image.width() as u8,
            filled: vec![],
            texture,
            text_update: None,
            texture_ent,
        };

        let collider = get_collider(
            &image_values(&image),
            image.width(),
            image.height(),
            (rigidbody.width as f64 / -2., rigidbody.height as f64 / -2.),
        )
        .unwrap();

        commands
            .spawn(collider)
            .insert(rigidbody)
            .insert(bevy_rapier2d::prelude::RigidBody::Dynamic)
            .insert(TransformBundle::default());

        commands.entity(ent).insert(Hydrated);
    }
}

pub fn update_rigidibodies(
    mut chunk_manager: ResMut<ChunkManager>,
    mut rigidbodies: Query<(&Transform, &mut Rigidbody)>,
    mut transforms: Query<&mut Transform, Without<Rigidbody>>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
    mut dirty_rects: ResMut<DirtyRects>,
) {
    let materials = materials.0.get(materials.1 .0.clone()).unwrap();

    for (transform, mut rigidbody) in &mut rigidbodies {
        let mut rotation = -(transform.rotation.to_euler(EulerRot::XYZ).2 as f64).to_degrees();
        if rotation < 0. {
            rotation += 360.;
        }
        let (width, height, rotated) = rotsprite::rotsprite(
            rigidbody.atoms.as_slice(),
            &Atom::default(),
            rigidbody.width as usize,
            rotation,
        )
        .unwrap();

        //New texture data
        let mut data = SmallVec::new();

        let mut off = transform.translation.xy();
        off.y *= -1.;
        let pos = vec2(width as f32 / -2., height as f32 / -2.) + off;

        //This fills the chunks with Object atoms, and also converts the atom data to color data
        for (y, x) in (0..height).cartesian_product(0..width) {
            let pos = pos + vec2(x as f32, y as f32);
            let chunk_pos = global_to_chunk(pos.as_ivec2());

            let rotated_atom = rotated[y * width + x];
            if materials[rotated_atom.id].is_solid() {
                update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);

                if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                    if materials[atom.id].is_void() {
                        *atom = Atom::object();
                        rigidbody.filled.push(chunk_pos);
                    }
                }

                data.extend_from_slice(&rotated_atom.color);
            } else {
                data.extend_from_slice(&[0, 0, 0, 0]);
            }
        }

        //Set new texture and new texture transform
        let mut text_transform = transforms.get_mut(rigidbody.texture_ent).unwrap();
        text_transform.translation.x = pos.as_ivec2().x as f32;
        text_transform.translation.y = -pos.as_ivec2().y as f32;

        let text_update = ExtractedTextureUpdate {
            id: rigidbody.texture.id(),
            data,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            size: Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        };

        rigidbody.text_update = Some(text_update);
    }
}

pub fn unfill_rigidbodies(
    mut chunk_manager: ResMut<ChunkManager>,
    mut rigidbodies: Query<&mut Rigidbody>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    let materials = materials.0.get(materials.1 .0.clone()).unwrap();

    for mut rigidbody in &mut rigidbodies {
        while let Some(chunk_pos) = rigidbody.filled.pop() {
            if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                if materials[atom.id].is_object() {
                    *atom = Atom::default();
                }
            }
        }
    }
}

pub fn extract_images(
    rigidibodies: Extract<Query<&Rigidbody>>,
    mut extracted_updates: ResMut<ExtractedTextureUpdates>,
) {
    for rigibody in &rigidibodies {
        if let Some(text_update) = &rigibody.text_update {
            extracted_updates.push(ExtractedTextureUpdate {
                data: vec![0; (rigibody.texture_lenght().pow(2) * 4) as usize].to_smallvec(),
                id: text_update.id,
                origin: Origin3d::ZERO,
                size: Extent3d {
                    width: rigibody.texture_lenght(),
                    height: rigibody.texture_lenght(),
                    depth_or_array_layers: 1,
                },
            });
            extracted_updates.push(text_update.clone());
        }
    }
}

pub fn image_values(image: &Image) -> Vec<u8> {
    let mut values = vec![];
    for pixel in image.data.chunks_exact(4) {
        if pixel[3] > 0 {
            values.push(255)
        } else {
            values.push(0)
        }
    }

    values
}

pub fn image_atoms(image: &Image) -> Vec<Atom> {
    let mut atoms = vec![];
    for pixel in image.data.chunks_exact(4) {
        if pixel[3] > 0 {
            atoms.push(Atom {
                id: 8,
                color: pixel.try_into().unwrap(),
                ..Default::default()
            });
        } else {
            atoms.push(Atom::default())
        }
    }

    atoms
}

//TODO Add multiple colliders support
pub fn get_collider(
    values: &[u8],
    width: u32,
    height: u32,
    origin: (f64, f64),
) -> Option<Collider> {
    let map = DensityMap::new(width as usize, height as usize, 1, values.to_vec()).unwrap();
    let settings = GenerateDensityMeshSettings {
        points_separation: PointsSeparation::SteepnessMapping(0., 10.),

        ..Default::default()
    };
    let mut mesh = DensityMeshGenerator::new(vec![], map, settings);
    mesh.process_wait().unwrap_or_default();
    let Some(mesh) = mesh.mesh() else {
        return None;
    };

    let mut colliders = vec![];
    for triangle in &mesh.triangles {
        let collider = Collider::triangle(
            vec2(mesh.points[triangle.a].x, -mesh.points[triangle.a].y)
                + vec2(origin.0 as f32, -origin.1 as f32),
            vec2(mesh.points[triangle.b].x, -mesh.points[triangle.b].y)
                + vec2(origin.0 as f32, -origin.1 as f32),
            vec2(mesh.points[triangle.c].x, -mesh.points[triangle.c].y)
                + vec2(origin.0 as f32, -origin.1 as f32),
        );
        colliders.push((Vec2::ZERO, 0.0_f32, collider));
    }

    /*let c = ContourBuilder::new(width, height, false)
        .x_origin(origin.0)
        .y_origin(origin.1);

    let res = c.contours(values, &[0.5]).unwrap();
    let mut colliders = vec![];
    for countour in res {
        let geometry = countour.geometry().simplify(&4.0);

        for polygon in geometry {
            let triangles = polygon.earcut_triangles();
            for triangle in triangles {
                let collider = Collider::triangle(
                    vec2(triangle.0.x as f32, -triangle.0.y as f32),
                    vec2(triangle.1.x as f32, -triangle.1.y as f32),
                    vec2(triangle.2.x as f32, -triangle.2.y as f32),
                );
                colliders.push((Vec2::ZERO, 0.0_f32, collider));
            }
        }
    }*/
    if !colliders.is_empty() {
        Some(Collider::compound(colliders))
    } else {
        None
    }
}

pub struct RigidbodyPlugin;
impl Plugin for RigidbodyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_images)
            .add_systems(Update, add_rigidbodies.run_if(in_state(GameState::Game)))
            .add_systems(
                FixedUpdate,
                (
                    update_rigidibodies.before(chunk_manager_update),
                    unfill_rigidbodies.after(chunk_manager_update),
                )
                    .run_if(in_state(GameState::Game)),
            );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(ExtractSchedule, extract_images);
        }
    }
}
