use crate::prelude::*;
use geo::{SimplifyVwPreserve, TriangulateEarcut};
use itertools::Itertools;

#[derive(Component, Default)]
pub struct Rigidbody {
    pub atoms: Vec<Atom>,
    pub width: u8,
    pub height: u8,
    // Positions to remove Object atom later
    pub filled: Vec<ChunkPos>,
    pub _despawn: DespawnOptions,
}

#[derive(Default)]
pub struct DespawnOptions {}

#[derive(Component, Default)]
pub struct RigidbodyHandle(Handle<Image>, Vec2);

#[derive(Component)]
pub struct Hydrated;

//TODO Add rigidbody file
pub fn _load_images(mut commands: Commands, server: Res<AssetServer>) {
    let image: Handle<Image> = server.load("prefabs/tree/image.png");
    commands.spawn(RigidbodyHandle(image.clone(), vec2(64., -64.)));

    let image: Handle<Image> = server.load("player/player_tool.png");
    commands.spawn(RigidbodyHandle(image.clone(), vec2(0., 0.)));
    let image: Handle<Image> = server.load("player/jetpack.png");
    commands.spawn(RigidbodyHandle(image.clone(), vec2(-64., 0.)));
}

pub fn add_rigidbodies(
    mut commands: Commands,
    images: Res<Assets<Image>>,
    handles: Query<(Entity, &RigidbodyHandle), Without<Hydrated>>,
) {
    for (ent, handle) in &handles {
        let Some(image) = images.get(&handle.0) else {
            continue;
        };

        let rigidbody = Rigidbody {
            atoms: image_atoms(image),
            height: image.height() as u8,
            width: image.width() as u8,
            filled: vec![],
            ..Default::default()
        };

        let collider = get_collider(&image_values(image), image.width(), image.height()).unwrap();

        commands
            .spawn(collider)
            .insert(rigidbody)
            .insert(RapierRigidbody::Dynamic)
            .insert(bevy_rapier2d::prelude::Velocity::zero())
            .insert(bevy_rapier2d::prelude::ExternalImpulse::default())
            .insert(ReadMassProperties::default())
            .insert(Sprite {
                image: handle.0.clone(),

                anchor: bevy::sprite::Anchor::TopLeft,

                ..Default::default()
            })
            .insert(Transform::from_xyz(
                handle.1.x,
                -handle.1.y,
                RIGIDBODY_LAYER,
            ));

        commands.entity(ent).insert(Hydrated);
    }
}

pub fn update_rigidibodies(
    mut commands: Commands,
    mut chunk_manager: ResMut<ChunkManager>,
    mut rigidbodies: Query<(&Transform, &mut Rigidbody, &Velocity, &ReadMassProperties)>,
    mut dirty_rects: ResMut<DirtyRects>,
) {
    puffin::profile_function!();

    for (transform, mut rigidbody, velocity, mass_prop) in &mut rigidbodies {
        let (width, height) = (rigidbody.width as usize, rigidbody.height as usize);
        let angle = -transform.rotation.to_euler(EulerRot::XYZ).2;
        let mut top_left = transform.translation.xy();
        top_left.y *= -1.;

        //let mut stop_count = 0;
        //let mut total = 0;

        //This fills the chunks with Object atoms
        for (y, x) in (0..height).cartesian_product(0..width) {
            let pos = top_left + vec2(x as f32, y as f32).rotate(Vec2::from_angle(angle));
            let chunk_pos = global_to_chunk(pos.round().as_ivec2());

            let rotated_atom = rigidbody.atoms[y * width + x];
            if rotated_atom.is_solid() {
                update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
                //total += 1;

                if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                    if !atom.is_solid() && !atom.is_object() {
                        if atom.is_liquid() || atom.is_powder() {
                            let mut point = pos;
                            point.y *= -1.;

                            let center_of_mass =
                                mass_prop.local_center_of_mass + transform.translation.xy();
                            let vel_point =
                                velocity.linear_velocity_at_point(point, center_of_mass);

                            /*Apply impulse to rigidbody
                            *external_impulse += ExternalImpulse::at_point(
                                vel_point / mass_prop.mass,
                                point,
                                center_of_mass,
                            );*/

                            //Spawn particle
                            commands.spawn(Particle {
                                atom: *atom,
                                velocity: vel_point.normalize_or_zero()
                                    * (vel_point.length() * mass_prop.mass / 1000.).min(16.),
                                pos: pos.round(),
                                ..Default::default()
                            });
                            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
                        }

                        rigidbody.filled.push(chunk_pos);
                        *atom = Atom::object();
                    }
                }
            }
        }

        /*
        let stop_treshold = total as f32 * STOP_RATE;
        let new_gravity = ((stop_treshold - stop_count as f32) / stop_treshold).clamp(0., 1.);

        gravity_scale.0 = new_gravity;
        */
    }
}

pub fn unfill_rigidbodies(
    mut chunk_manager: ResMut<ChunkManager>,
    mut rigidbodies: Query<&mut Rigidbody>,
) {
    for mut rigidbody in &mut rigidbodies {
        while let Some(chunk_pos) = rigidbody.filled.pop() {
            if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                if atom.is_object() {
                    *atom = Atom::default();
                }
            }
        }
    }
}

pub fn image_values(image: &Image) -> Vec<f64> {
    let mut values = vec![];
    for pixel in image.data.as_chunks::<4>().0 {
        if pixel[3] > 0 {
            values.push(1.)
        } else {
            values.push(0.)
        }
    }

    values
}

pub fn image_atoms(image: &Image) -> Vec<Atom> {
    let mut atoms = vec![];
    for pixel in image.data.as_chunks::<4>().0 {
        if pixel[3] > 0 {
            atoms.push(Atom {
                id: 8,
                color: *pixel,
                ..Default::default()
            });
        } else {
            atoms.push(Atom::default())
        }
    }

    atoms
}

pub fn get_collider(values: &[f64], width: u32, height: u32) -> Option<Collider> {
    let c = ContourBuilder::new(width as usize, height as usize, false);

    let res = c.contours(values, &[0.5]).unwrap();
    let mut colliders = vec![];
    for countour in res {
        let geometry = countour.geometry().simplify_vw_preserve(&1.5);

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
    }
    if !colliders.is_empty() {
        Some(Collider::compound(colliders))
    } else {
        None
    }
}

pub struct RigidbodyPlugin;
impl Plugin for RigidbodyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, _load_images)
            .add_systems(Update, add_rigidbodies.run_if(in_state(GameState::Game)))
            .add_systems(
                FixedUpdate,
                (
                    update_rigidibodies
                        .before(chunk_manager_update)
                        .before(update_particles),
                    unfill_rigidbodies.after(chunk_manager_update),
                )
                    .run_if(in_state(GameState::Game)),
            );
    }
}
