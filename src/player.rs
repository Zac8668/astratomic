use bevy::color::palettes::css::*;
use bevy::color::palettes::tailwind::*;
use bevy::sprite::Anchor;

use crate::prelude::*;

#[derive(Component)]
pub struct Player {
    fuel: f32,
    state: PlayerState,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            fuel: FUEL_MAX,
            state: PlayerState::default(),
        }
    }
}

impl Drop for Actor {
    fn drop(&mut self) {
        let file = File::create("assets/world/player").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &self.pos).unwrap();
    }
}

#[derive(Default)]
pub enum PlayerState {
    #[default]
    Idle,
    Walking,
    Dead,
    Jumping(f32),
    Jetpack(bool),
}

#[derive(Component, Default)]
pub struct Tool;

#[derive(Component)]
pub struct ToolFront;

#[derive(Component)]
pub struct Life(f32);

pub fn player_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut texture_atlases: ResMut<Assets<TextureAtlasLayout>>,
) {
    let pos: IVec2;
    if let Ok(file) = File::open("assets/world/player") {
        let mut buffered = BufReader::new(file);
        pos = bincode::deserialize_from(&mut buffered).unwrap();
    } else {
        pos = IVec2::new(0, -260);
        let file = File::create("assets/world/player").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &pos).unwrap();
    }

    let player_actor = Actor {
        height: 17,
        width: 10,
        pos,
        vel: vec2(0., 0.),
        colliding: None,
    };

    let player_handle = asset_server.load("player/player_sheet.png");
    let animation_indices = AnimationIndices { first: 0, last: 1 };
    let player_atlas_layout = TextureAtlasLayout::from_grid(UVec2::new(24, 24), 8, 5, None, None);
    let atlas = TextureAtlas {
        index: animation_indices.first,
        layout: texture_atlases.add(player_atlas_layout),
    };
    let player_transform = GlobalTransform::from_xyz(5. * 3., -8. * 3., PLAYER_LAYER);

    let tool_handle = asset_server.load("player/player_tool.png");
    let tool_bundle = Sprite {
        image: tool_handle,

        anchor: Anchor::Custom(vec2(-0.1, 0.)),

        ..Default::default()
    };
    let tool_front_ent = commands
        .spawn((Transform::from_translation(vec3(5., 0., 0.)), ToolFront))
        .id();
    let tool_ent = commands
        .spawn(tool_bundle)
        .insert(Tool)
        .insert(Transform::from_translation(Vec3::new(-1., -3.5, 0.1)))
        .insert_children(0, &[tool_front_ent])
        .id();

    commands
        .spawn((
            player_actor.clone(),
            Player::default(),
            Sprite::from_atlas_image(player_handle, atlas),
            player_transform,
            animation_indices,
            AnimationTimer(Timer::from_seconds(0.1, TimerMode::Repeating)),
            bevy_rapier2d::prelude::RigidBody::Fixed,
            bevy_rapier2d::prelude::LockedAxes::ROTATION_LOCKED,
            bevy_rapier2d::prelude::Collider::cuboid(
                player_actor.width as f32 / 2.,
                player_actor.height as f32 / 2.,
            ),
            Life(100.),
        ))
        .add_child(tool_ent);

    //Life
    commands.spawn((
        Node {
            width: Val::Px(180.),
            height: Val::Px(20.),
            border: UiRect::left(Val::Px(180.)),
            margin: UiRect::all(Val::Px(20.)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(MAROON.into()),
        BorderColor(RED.into()),
        Outline {
            width: Val::Px(6.),
            offset: Val::Px(6.),
            color: Color::WHITE,
        },
        PlayerLifeNode,
    ));

    //Fuel
    commands.spawn((
        Node {
            width: Val::Px(20.),
            height: Val::Px(180.),
            border: UiRect::top(Val::Px(180.)),
            margin: UiRect::all(Val::Px(20.)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            justify_self: JustifySelf::End,
            ..default()
        },
        BackgroundColor(LIGHT_GOLDENROD_YELLOW.into()),
        BorderColor(YELLOW.into()),
        Outline {
            width: Val::Px(6.),
            offset: Val::Px(6.),
            color: Color::WHITE,
        },
        PlayerFuelNode,
    ));
}

#[derive(Event)]
pub struct DamageEvent {
    ent: Entity,
    damage: f32,
}

impl DamageEvent {
    pub fn new(ent: Entity, damage: f32) -> Self {
        Self { ent, damage }
    }
}

#[derive(Component)]
pub struct PlayerLifeNode;

#[derive(Component)]
pub struct PlayerFuelNode;

/// Updates player input stuff
pub fn update_player(
    input: (Res<Inputs>, EventReader<MouseWheel>),
    mut player: Query<(&mut Actor, &mut Player)>,
    chunk_manager: ResMut<ChunkManager>,
    time: Res<Time>,
    mut zoom: ResMut<Zoom>,
    mut inventory: ResMut<Inventory>,
) {
    let (mut actor, mut player) = player.single_mut();
    if matches!(player.state, PlayerState::Dead) {
        return;
    }

    let (inputs, mut scroll_evr) = input;

    // Movement
    let x = inputs.right - inputs.left;
    actor.vel.x = x * RUN_SPEED;

    let on_ground = on_ground(&chunk_manager, &actor);

    // Refuel
    if on_ground {
        player.fuel = (player.fuel + FUEL_REGEN).clamp(0., Player::default().fuel);
    }

    if on_ground {
        if x.abs() > 0. {
            player.state = PlayerState::Walking
        } else {
            player.state = PlayerState::Idle
        }
    }

    // Jump
    if inputs.jump_just_pressed {
        if on_ground {
            actor.vel.y -= JUMP_MAG;
            player.state = PlayerState::Jumping(time.elapsed_secs_wrapped());
        } else {
            player.state = PlayerState::Jetpack(true);
            actor.vel.y = 0.;
        }
    }

    //Jump higher when holding space
    if let PlayerState::Jumping(jump_start) = player.state {
        if inputs.jump_pressed && time.elapsed_secs_wrapped() - jump_start < TIME_JUMP_PRESSED {
            actor.vel.y -= PRESSED_JUMP_MAG
        }
    }

    // Jetpack
    let mut new_up = false;
    if let PlayerState::Jetpack(_) = player.state {
        if player.fuel > 0. && inputs.jump_pressed {
            actor.vel.y = (actor.vel.y - JETPACK_FORCE).clamp(-JETPACK_MAX, f32::MAX);
            player.fuel -= FUEL_COMSUMPTON;
            new_up = true;
        } else {
            new_up = false;
        }
    }

    if let PlayerState::Jetpack(up) = &mut player.state {
        *up = new_up
    };

    //Scroll
    for ev in scroll_evr.read() {
        if ev.unit == MouseScrollUnit::Line && inputs.ctrl {
            zoom.0 *= 0.9_f32.powi(ev.y as i32);
            zoom.0 = zoom.0.clamp(ZOOM_LOWER_BOUND, ZOOM_UPPER_BOUND);
        } else if ev.unit == MouseScrollUnit::Line {
            inventory.selected = ((inventory.selected as f32 + ev.y + 8.) % 8.) as usize;
        }
    }

    //Change shooting atoms
    for (i, &pressed) in inputs.numbers.iter().enumerate() {
        if pressed {
            inventory.selected = i;
            break;
        }
    }
}

// Update player non controllable stuff
pub fn update_player_nc(
    mut nodes: (
        Query<&mut Node, (With<PlayerLifeNode>, Without<PlayerFuelNode>)>,
        Query<&mut Node, (With<PlayerFuelNode>, Without<PlayerLifeNode>)>,
    ),
    mut player: Query<(
        &mut Actor,
        &mut Player,
        &mut AnimationIndices,
        &Life,
        Entity,
    )>,
    mut ev_damage: EventWriter<DamageEvent>,
) {
    let (mut player_life_node, mut player_fuel_node) = (nodes.0.single_mut(), nodes.1.single_mut());
    let (mut actor, mut player, mut anim_idxs, life, ent) = player.single_mut();

    // Gravity
    if actor.vel.y < PLAYER_TERM_VEL as f32 {
        actor.vel.y += 0.6
            * if matches!(player.state, PlayerState::Jetpack { .. }) {
                0.4
            } else {
                0.6
            };
    }

    //Animation
    (anim_idxs.first, anim_idxs.last) = match player.state {
        PlayerState::Idle => (0, 1),
        PlayerState::Walking => (8, 11),
        PlayerState::Jumping { .. } => (16, 23),
        PlayerState::Jetpack(up) => {
            if up {
                (24, 26)
            } else {
                (32, 32)
            }
        }
        PlayerState::Dead => (0, 0), //TODO
    };

    //Fall Damage
    if let Some(speed) = actor.colliding {
        if speed >= 10. {
            ev_damage.send(DamageEvent::new(ent, speed * 3.));
        }
    }

    //Death
    if life.0 <= 0. {
        player.state = PlayerState::Dead;
    }

    //Life
    player_life_node.border.left = Val::Px(180. * life.0 / 100.);

    //Fuel
    player_fuel_node.border.top = Val::Px(180. * player.fuel / FUEL_MAX);
}

#[derive(Component)]
pub struct DamageFeedTimer(Timer);

pub fn damage_feed(
    mut commands: Commands,
    mut sprite_query: Query<(Entity, &mut Sprite, &mut Life), Without<DamageFeedTimer>>,
    mut timer_query: Query<(Entity, &mut Sprite, &mut DamageFeedTimer)>,
    mut ev_damage: EventReader<DamageEvent>,
    time: Res<Time>,
) {
    for ev in ev_damage.read() {
        let sprite = sprite_query.get_mut(ev.ent);
        if let Ok((ent, mut sprite, mut life)) = sprite {
            sprite.color = RED_500.into();
            commands.entity(ent).insert(DamageFeedTimer(Timer::new(
                Duration::from_secs_f32(DAMAGE_FEED_TIME),
                TimerMode::Once,
            )));
            life.0 -= ev.damage;
            if life.0 <= 0. {
                life.0 = 0.;
            }
        }
    }

    for (ent, mut sprite, mut timer) in timer_query.iter_mut() {
        timer.0.tick(time.delta());

        if timer.0.finished() {
            sprite.color = Color::WHITE;
            commands.entity(ent).remove::<DamageFeedTimer>();
        }
    }
}

#[derive(Component)]
pub struct RespawnUI;

#[derive(Component)]
pub enum ButtonType {
    Restart,
    Quit,
}

pub fn dead(
    mut commands: Commands,
    quaver_font: Res<QuaverFont>,
    mut player: Query<(&mut Player, &mut Life, &mut Actor)>,
    respawn_ui: Query<Entity, With<RespawnUI>>,
    mut interaction_query: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
            &ButtonType,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    mut exit: EventWriter<AppExit>,
) {
    let (mut player, mut life, mut actor) = player.single_mut();
    if !matches!(player.state, PlayerState::Dead) {
        return;
    } else if respawn_ui.is_empty() {
        let quaver_font: Handle<Font> = quaver_font.0.clone();
        //Spawn UI
        let button_style = Node {
            width: Val::Px(180.0),
            height: Val::Px(45.0),
            border: UiRect::all(Val::Px(5.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        };

        let ui_style = Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(15.),
            ..default()
        };

        let text_style = TextFont {
            font: quaver_font,
            font_size: 20.0,
            ..Default::default()
        };
        let text_color = TextColor(Color::srgb(0.9, 0.9, 0.9));

        commands
            .spawn((
                ui_style,
                ImageNode::solid_color(Color::srgba(0., 0., 0., 0.9)),
            ))
            .insert(RespawnUI)
            .with_children(|parent| {
                //Restart
                parent
                    .spawn((Button, button_style.clone()))
                    .insert(ButtonType::Restart)
                    .with_children(|parent| {
                        parent.spawn((Text::new("Restart"), text_style.clone(), text_color));
                    });

                //Quit
                parent
                    .spawn((Button, button_style))
                    .insert(ButtonType::Quit)
                    .with_children(|parent| {
                        parent.spawn((Text::new("Quit"), text_style, text_color));
                    });
            });
    }
    for (interaction, mut color, mut border_color, button_type) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
                border_color.0 = Color::srgb(1., 0., 0.);

                match *button_type {
                    ButtonType::Restart => {
                        player.state = PlayerState::Idle;
                        life.0 = 100.;
                        actor.pos = IVec2::new(0, 24);

                        for ent in respawn_ui.iter() {
                            commands.entity(ent).despawn_recursive()
                        }
                        return;
                    }
                    ButtonType::Quit => {
                        exit.send(AppExit::Success);
                    }
                }
            }
            Interaction::Hovered => {
                *color = HOVERED_BUTTON.into();
                border_color.0 = Color::WHITE;
            }
            Interaction::None => {
                *color = NORMAL_BUTTON.into();
                border_color.0 = Color::BLACK;
            }
        }
    }
}

pub fn tool_system(
    mut commands: Commands,
    mut tool: Query<
        (
            &mut Transform,
            &GlobalTransform,
            &mut Sprite,
            &mut Visibility,
        ),
        With<Tool>,
    >,
    mut camera: Query<(&Camera, &GlobalTransform), Without<Tool>>,
    tool_front_ent: Query<Entity, With<ToolFront>>,
    querys: (Query<&Window>, Query<(&mut Sprite, &Player), Without<Tool>>),
    resources: (
        ResMut<ChunkManager>,
        ResMut<DirtyRects>,
        ResMut<BrokenAtoms>,
        Res<Inputs>,
        Res<Inventory>,
        Res<Assets<Materials>>,
        Res<MaterialsHandle>,
    ),
    mut ev_item: EventWriter<ItemEvent>,
) {
    let (mut tool_transform, tool_gtransform, mut tool_sprite, mut visibility) = tool.single_mut();
    let (camera, camera_gtransform) = camera.single_mut();
    let (window, mut player) = querys;
    let (mut textatlas_sprite, player) = player.single_mut();
    let (
        mut chunk_manager,
        mut dirty_rects,
        mut broken_atoms,
        inputs,
        inventory,
        materials,
        materials_handle,
    ) = resources;
    let Ok(window) = window.get_single() else {
        return;
    };
    let materials = materials.get(&materials_handle.0).unwrap();

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    if matches!(player.state, PlayerState::Dead) {
        return;
    }

    // Calculate a world position based on the cursor's position.
    let Ok(world_position) = camera.viewport_to_world_2d(camera_gtransform, cursor_position) else {
        return;
    };

    //Don't use tool if atom is solid or we don't have nothing on selected slot
    if let Some(slot) = inventory.slots[inventory.selected] {
        if let Item::Atom(atom) = slot.item {
            if atom.is_solid() {
                *visibility = Visibility::Hidden;
                return;
            }
        }
    }

    if inventory.slots[inventory.selected].is_none() {
        *visibility = Visibility::Hidden;
        return;
    } else {
        *visibility = Visibility::Visible;
    }

    //Rotate and move sprite
    let center_vec = tool_gtransform.compute_transform().translation.xy();
    let tool_vec = world_position - center_vec;
    let angle = tool_vec.y.atan2(tool_vec.x);
    tool_transform.rotation = Quat::from_rotation_z(angle);

    let flip_bool = angle.abs() > std::f32::consts::FRAC_PI_2;
    textatlas_sprite.flip_x = flip_bool;
    tool_sprite.flip_y = flip_bool;
    tool_transform.translation.x =
        tool_transform.translation.x.abs() * (flip_bool as i8 * 2 - 1) as f32;

    //Don't use tool if atom is solid or we don't have nothing on selected slot
    if let Some(slot) = inventory.slots[inventory.selected] {
        if let Item::Atom(atom) = slot.item {
            if atom.is_solid() {
                *visibility = Visibility::Hidden;
                return;
            }
        }
    }

    if inventory.slots[inventory.selected].is_none() {
        *visibility = Visibility::Hidden;
        return;
    } else {
        *visibility = Visibility::Visible;
    }

    //Tool pulling and pushing atoms
    let mut center_vec_y_flipped = center_vec;
    center_vec_y_flipped.y *= -1.;

    let tool_slope = Vec2::new(angle.cos(), -angle.sin());
    let bound_slope = Vec2::new((angle + std::f32::consts::FRAC_PI_2).cos(), -(angle).cos());
    let tool_front = center_vec_y_flipped + tool_slope * 5.;

    let mut pos_to_update = vec![];
    if inputs.push {
        let new_tool_front = tool_front + tool_slope * 3.5;
        let n = 6;

        for i in 0..=n {
            let rand_angle = fastrand::f32() * std::f32::consts::TAU;

            let mut vec = new_tool_front - bound_slope * 2.
                + bound_slope * 2.5 * i as f32 / n as f32
                + vec2(rand_angle.cos(), rand_angle.sin());

            vec += tool_slope * 7. * angle.sin().max(0.);

            let chunk_pos = global_to_chunk(vec.as_ivec2());
            if let (Some(atom), Some(slot)) = (
                chunk_manager.get_mut_atom(chunk_pos),
                inventory.slots[inventory.selected],
            ) {
                if atom.is_void() || atom.is_object() {
                    if let Item::Atom(atom) = slot.item {
                        if !atom.is_solid() {
                            let angle = fastrand::f32() * 0.5 - 0.25;
                            let vel = (tool_slope * 10. * (fastrand::f32() * 0.2 + 0.8))
                                .rotate(vec2(angle.cos(), angle.sin()));
                            commands.spawn(Particle {
                                atom: Atom::new(atom.id, materials),
                                velocity: vel,
                                pos: vec,
                                ..Default::default()
                            });

                            ev_item.send(ItemEvent::RemoveSelected);
                        }
                    }
                }
            }
        }
    } else if inputs.pull {
        let center_bound = tool_front + tool_slope * TOOL_DISTANCE;

        let bound1 = (center_bound + bound_slope * TOOL_RANGE).as_ivec2();
        let bound2 = (center_bound + -bound_slope * TOOL_RANGE).as_ivec2();

        for bound_vec in Line::new(bound1, bound2 - bound1) {
            for vec in Line::new(
                (tool_front - 4. * tool_slope).as_ivec2(),
                bound_vec - (tool_front - 4. * tool_slope).as_ivec2(),
            ) {
                let chunk_pos = global_to_chunk(vec);
                if (vec.distance_squared((tool_front - 6. * tool_slope).as_ivec2()) as f32).sqrt()
                    < 6.
                {
                    continue;
                }

                if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                    if !atom.is_void() && !atom.is_object() && inventory.can_add(Item::Atom(*atom))
                    {
                        commands.spawn(Particle {
                            atom: *atom,
                            pos: chunk_pos.to_global().as_vec2(),
                            state: PartState::Follow(tool_front_ent.single()),
                            ..Default::default()
                        });

                        if atom.is_solid() {
                            broken_atoms.0.insert(vec);
                        }

                        pos_to_update.push(chunk_pos);
                        *atom = Atom::default();
                        break;
                    }
                }
            }
        }
    }

    let mut chunks = HashSet::new();
    for pos in pos_to_update {
        update_dirty_rects_3x3(&mut dirty_rects.current, pos);
        update_dirty_rects(&mut dirty_rects.render, pos);
        chunks.insert(pos.chunk);
    }

    for chunk in chunks {
        let chunk = chunk_manager.chunks.get(&chunk).unwrap();
        commands.entity(chunk.entity.unwrap()).remove::<Collider>();
    }
}

pub fn update_player_sprite(mut query: Query<(&mut Transform, &Actor), With<Player>>) {
    let (mut transform, actor) = query.single_mut();
    let top_corner_vec = vec3(actor.pos.x as f32, -actor.pos.y as f32, 2.);
    let center_vec = top_corner_vec + vec3(actor.width as f32 / 2., -8., 0.);
    transform.translation = center_vec;
}

#[derive(Resource, Default)]
pub struct SavingTask(pub Option<Task<()>>);

pub fn get_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut inputs: ResMut<Inputs>,
) {
    //TODO Add controller support

    //Jump
    if keys.just_pressed(KeyCode::Space) {
        inputs.jump_just_pressed = true;
        inputs.jump_pressed = true;
    } else if keys.pressed(KeyCode::Space) {
        inputs.jump_pressed = true;
    }

    //Movement
    if keys.pressed(KeyCode::KeyA) {
        inputs.left = 1.;
    }
    if keys.pressed(KeyCode::KeyD) {
        inputs.right = 1.;
    }

    //Tool
    if mouse_buttons.pressed(MouseButton::Left) {
        inputs.pull = true;
    }
    if mouse_buttons.pressed(MouseButton::Right) {
        inputs.push = true;
    }

    //Numbers, to select atoms
    if keys.just_pressed(KeyCode::Digit1) {
        inputs.numbers[0] = true;
    } else if keys.just_pressed(KeyCode::Digit2) {
        inputs.numbers[1] = true;
    } else if keys.just_pressed(KeyCode::Digit3) {
        inputs.numbers[2] = true;
    } else if keys.just_pressed(KeyCode::Digit4) {
        inputs.numbers[3] = true;
    } else if keys.just_pressed(KeyCode::Digit5) {
        inputs.numbers[4] = true;
    } else if keys.just_pressed(KeyCode::Digit6) {
        inputs.numbers[5] = true;
    } else if keys.just_pressed(KeyCode::Digit7) {
        inputs.numbers[6] = true;
    } else if keys.just_pressed(KeyCode::Digit8) {
        inputs.numbers[7] = true;
    }

    //Inventory open and close
    if keys.just_released(KeyCode::KeyE) {
        inputs.inventory_toggle = true;
    }

    //Check if we change selected or zoom
    if keys.pressed(KeyCode::ControlLeft) {
        inputs.ctrl = true;
    }
}

pub fn clear_input(mut inputs: ResMut<Inputs>) {
    *inputs = Inputs::default();
}

#[derive(Resource, Default)]
pub struct Inputs {
    left: f32,
    right: f32,

    pull: bool,
    push: bool,

    jump_pressed: bool,
    jump_just_pressed: bool,

    numbers: [bool; 8],
    ctrl: bool,

    pub inventory_toggle: bool,
}

pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                update_player.before(update_actors),
                update_player_nc,
                damage_feed,
                dead,
                update_player_sprite.after(update_actors),
                tool_system
                    .before(chunk_manager_update)
                    .before(update_particles),
            )
                .run_if(in_state(GameState::Game)),
        )
        .add_systems(PostUpdate, clear_input.run_if(in_state(GameState::Game)))
        .add_systems(PreUpdate, get_input.run_if(in_state(GameState::Game)))
        .init_resource::<SavingTask>()
        .init_resource::<Inputs>()
        .add_event::<DamageEvent>()
        .add_systems(OnEnter(GameState::Game), player_setup.after(manager_setup));
    }
}
