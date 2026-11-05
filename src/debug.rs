use crate::prelude::*;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

#[derive(Component)]
struct FpsText;

fn spawn_fps_ui(mut commands: Commands) {
    let text_style = TextFont {
        font_size: 18.0,
        ..Default::default()
    };
    let text_color = TextColor(Color::srgb(0.9, 0.9, 0.9));

    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(10.0),
            top: Val::Px(70.0),
            ..Default::default()
        })
        .with_children(|parent| {
            parent.spawn((Text::new("FPS: --"), text_style, text_color, FpsText));
        });
}

fn update_fps_ui(
    diagnostics: Res<DiagnosticsStore>,
    mut fps_text: Query<&mut Text, With<FpsText>>,
) {
    let Ok(mut text) = fps_text.get_single_mut() else {
        return;
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diagnostic| diagnostic.smoothed())
        .unwrap_or(0.0);

    text.0 = format!("FPS: {fps:.0}");
}

fn brush(
    window: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut chunk_manager: ResMut<ChunkManager>,
    mut dirty_rects: ResMut<DirtyRects>,
    prev_mpos: Res<PreviousMousePos>,
    input: (Res<ButtonInput<MouseButton>>, Res<ButtonInput<KeyCode>>),
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    let materials = materials.0.get(&materials.1 .0).unwrap();

    let atom = if input.0.pressed(MouseButton::Middle) {
        Atom::new(2, materials)
    } else if input.1.pressed(KeyCode::ControlLeft) {
        Atom::new(3, materials)
    } else if input.1.pressed(KeyCode::ShiftLeft) {
        Atom::new(8, materials)
    } else {
        return;
    };

    let (camera, camera_transform) = camera_q.single();
    let window = window.single();

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    // Calculate a world position based on the cursor's position.
    let Ok(mut world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position)
    else {
        return;
    };
    world_position.y *= -1.;

    for v in Line::new(
        prev_mpos.0.unwrap().as_ivec2(),
        world_position.as_ivec2() - prev_mpos.0.unwrap().as_ivec2(),
    ) {
        let pos = global_to_chunk(v);

        //Checks if there is a atom at the pos
        if chunk_manager.get_atom(&pos).is_none() {
            continue;
        }

        chunk_manager[pos] = atom;

        // Update simultation rect
        if let Some(dirty_rect) = dirty_rects.current.get_mut(&pos.chunk) {
            extend_rect_if_needed(dirty_rect, &pos.atom)
        } else {
            dirty_rects.current.insert(
                pos.chunk,
                URect::new(pos.atom.x, pos.atom.y, pos.atom.x, pos.atom.y),
            );
        }

        // Update render rect
        if let Some(dirty_rect) = dirty_rects.render.get_mut(&pos.chunk) {
            extend_rect_if_needed(dirty_rect, &pos.atom)
        } else {
            dirty_rects.render.insert(
                pos.chunk,
                URect::new(pos.atom.x, pos.atom.y, pos.atom.x, pos.atom.y),
            );
        }
    }
}

#[derive(Resource, Default)]
pub struct PreviousMousePos(pub Option<Vec2>);

fn prev_mpos(
    window: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut prev_mpos: ResMut<PreviousMousePos>,
) {
    let (camera, camera_transform) = camera_q.single();
    let window = window.single();

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    // Calculate a world position based on the cursor's position.
    let Ok(mut point) = camera.viewport_to_world_2d(camera_transform, cursor_position) else {
        return;
    };
    point.y *= -1.;

    prev_mpos.0 = Some(point);
}

//Debug Render systems

pub fn render_dirty_rects_gizmos(mut gizmos: Gizmos, dirty_rects: Res<DirtyRects>) {
    let (dirty_update, render_update) = (&dirty_rects.new, &dirty_rects.render);

    let mut i = false;
    for rect in [dirty_update, render_update] {
        for (chunk_pos, rect) in rect {
            let size =
                UVec2::new(rect.max.x - rect.min.x + 1, rect.max.y - rect.min.y + 1).as_vec2();
            let position = ivec2(
                chunk_pos.x * CHUNK_LENGHT as i32 + rect.min.x as i32,
                -(chunk_pos.y * CHUNK_LENGHT as i32) - rect.min.y as i32,
            )
            .as_vec2()
                + vec2(size.x, -(size.y)) / 2.;

            if !i {
                gizmos.rect_2d_gradient(position, 0., size);
            } else {
                gizmos.rect_2d(
                    Isometry2d::from_translation(position),
                    size,
                    Color::srgba(1., 0.25, 1., 1.),
                );
            }
        }
        i = true;
    }
}

/* TODO: Add this to a egui debug menu
pub fn _render_dirty_rects_image(mut commands: Commands, dirty_rects: Res<DirtyRects>) {
    let (dirty_update, render_update) = (&dirty_rects.new, &dirty_rects.render);

    let mut i = 0.;
    for rect in [dirty_update, render_update] {
        for (chunk_pos, rect) in rect {
            commands
                .spawn((
                    Sprite {
                        color: Color::srgba(i, 0.25, if i == 0. { 1. } else { 0. }, 0.1),
                        custom_size: Some(
                            UVec2::new(rect.max.x - rect.min.x + 1, rect.max.y - rect.min.y + 1)
                                .as_vec2(),
                        ),
                        anchor: Anchor::TopLeft,
                        ..default()
                    },
                    Transform::from_translation(
                        IVec3::new(
                            chunk_pos.x * CHUNK_LENGHT as i32 + rect.min.x as i32,
                            -(chunk_pos.y * CHUNK_LENGHT as i32) - rect.min.y as i32,
                            1,
                        )
                        .as_vec3(),
                    ),
                ))
                .insert(DeleteImage);
        }
        i += 1.;
    }
}

pub fn _delete_image(mut commands: Commands, debug_images: Query<Entity, With<DeleteImage>>) {
    for image in debug_images.iter() {
        commands.entity(image).despawn();
    }
}
*/

//#[derive(Component)]
//pub struct DeleteImage;

fn render_actors(mut gizmos: Gizmos, actors: Query<&Actor>) {
    for actor in actors.iter() {
        gizmos.rect_2d(
            Isometry2d::from_translation(
                vec2(actor.pos.x as f32, -actor.pos.y as f32)
                    + vec2(actor.width as f32, -(actor.height as f32)) / 2.,
            ),
            Vec2::new(actor.width as f32, actor.height as f32),
            Color::srgba(0.75, 0.25, 0.25, 0.2),
        );
    }
}

fn _camera(keys: Res<ButtonInput<KeyCode>>, mut camera_q: Query<&mut Transform, With<Camera>>) {
    let x = -(keys.pressed(KeyCode::KeyA) as u8 as f32) + keys.pressed(KeyCode::KeyD) as u8 as f32;
    let y = -(keys.pressed(KeyCode::KeyS) as u8 as f32) + keys.pressed(KeyCode::KeyW) as u8 as f32;

    let v = Vec2::new(x, y).normalize_or_zero().extend(0.);
    camera_q.single_mut().translation += v * _CAMERA_SPEED;
}

pub trait Rect2dGradient {
    fn rect_2d_gradient(&mut self, position: Vec2, rotation: f32, size: Vec2);
}

impl Rect2dGradient for Gizmos<'_, '_> {
    fn rect_2d_gradient(&mut self, position: Vec2, rotation: f32, size: Vec2) {
        let rotation = Mat2::from_angle(rotation);
        let [tl, tr, br, bl] = rect_inner(size).map(|vec2| position + rotation * vec2);
        self.linestrip_gradient_2d([
            (tl, Color::srgb(1., 0., 0.)),
            (tr, Color::srgb(0., 1., 0.)),
            (br, Color::srgb(0., 0., 1.)),
            (bl, Color::srgb(1., 1., 0.)),
            (tl, Color::srgb(0., 1., 0.)),
        ]);
    }
}

fn rect_inner(size: Vec2) -> [Vec2; 4] {
    let half_size = size / 2.;
    let tl = Vec2::new(-half_size.x, half_size.y);
    let tr = Vec2::new(half_size.x, half_size.y);
    let bl = Vec2::new(-half_size.x, -half_size.y);
    let br = Vec2::new(half_size.x, -half_size.y);
    [tl, tr, br, bl]
}

pub fn grab_rigidbodies(
    mut commands: Commands,
    rigidbodies: Query<(&mut Transform, Entity, &Collider), With<Rigidbody>>,
    mut transforms: Query<&mut Transform, (With<ImpulseJoint>, Without<Rigidbody>)>,
    window: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut grabbed: ResMut<Grabbed>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let (camera, camera_transform) = camera_q.single();
    let window = window.single();

    if input.pressed(KeyCode::KeyR) {
        let Some(cursor_position) = window.cursor_position() else {
            return;
        };

        // Calculate a world position based on the cursor's position.
        let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position)
        else {
            return;
        };

        if let Some(grabbed_ent) = grabbed.0 {
            //Update pos
            let mut transform = transforms.get_mut(grabbed_ent).unwrap();
            transform.translation.x = world_position.x;
            transform.translation.y = world_position.y;
        } else {
            //Add joint
            for (transform, entity, collider) in &rigidbodies {
                if collider.contains_point(
                    transform.translation.xy(),
                    transform.rotation.to_euler(EulerRot::XYZ).2,
                    world_position,
                ) {
                    let anchor = (world_position - transform.translation.xy()).rotate(
                        Vec2::from_angle(-transform.rotation.to_euler(EulerRot::XYZ).2),
                    );
                    let joint = RevoluteJointBuilder::new()
                        .local_anchor1(anchor)
                        .local_anchor2(Vec2::ZERO);

                    let move_ent = commands
                        .spawn(RigidBody::KinematicPositionBased)
                        .insert(ImpulseJoint::new(entity, joint))
                        .insert(Transform::from_xyz(world_position.x, world_position.y, 0.))
                        .id();

                    grabbed.0 = Some(move_ent);

                    break;
                }
            }
        }
    } else if let Some(ent) = grabbed.0.take() {
        commands.entity(ent).despawn_recursive()
    }
}

#[derive(Resource, Default)]
pub struct Grabbed(Option<Entity>);

pub struct DebugPlugin;
impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                render_dirty_rects_gizmos.after(update_actors),
                brush.after(chunk_manager_update),
                render_actors.after(update_actors),
                prev_mpos.after(brush),
                grab_rigidbodies,
                //_camera
            )
                .run_if(in_state(GameState::Game)),
        )
        .add_systems(Startup, spawn_fps_ui)
        .add_systems(Update, update_fps_ui)
        .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(RapierDebugRenderPlugin::default())
        //Frame on console
        .add_plugins((LogDiagnosticsPlugin::default(), FrameTimeDiagnosticsPlugin))
        .init_resource::<Grabbed>()
        .init_resource::<PreviousMousePos>();
    }
}
