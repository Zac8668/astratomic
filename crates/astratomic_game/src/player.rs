//! Player-specific components, network-ready `PlayerInput` / `MatchInputs`, deterministic
//! player controller & world action systems, `AtlasSprite` animations, and `LocalView` camera.

use std::collections::HashSet;

use astratomic_physics::prelude::{
    ActorState, KinematicActor, find_surface_spawn_pos, kinematic_actor_movement_system,
    world_cell_to_global_atom,
};
pub use astratomic_sim::consts::PLAYER_STEP_HEIGHT;
use astratomic_sim::{
    atom::{Atom, Materials},
    consts::{
        FUEL_COMSUMPTON, FUEL_REGEN, JETPACK_FORCE, JETPACK_MAX, JUMP_MAG, PLAYER_LAYER, RUN_SPEED,
    },
    geom_tools::Line,
    world::{
        ChunkWorld, DirtyRects, SimulationSettings, global_to_chunk, update_dirty_rects,
        update_dirty_rects_3x3,
    },
};
use bones_framework::prelude::*;

use crate::meta::GameMeta;

/// Bitflag for jump input (`Space`).
pub const BTN_JUMP: u8 = 1 << 0;
/// Bitflag for jetpack thrust (`Shift` / `W` or holding jump in mid-air).
pub const BTN_JETPACK: u8 = 1 << 1;
/// Bitflag for primary tool action (excavate / pull atoms).
pub const BTN_PRIMARY: u8 = 1 << 2;
/// Bitflag for secondary tool action (shoot / place selected material).
pub const BTN_SECONDARY: u8 = 1 << 3;
/// Bitflag for world / inventory interaction (`E` / `F`).
pub const BTN_INTERACT: u8 = 1 << 4;

/// Maximum number of players in a match session.
pub const MAX_PLAYERS: usize = 4;

/// Default half-extents of a player's hitbox in atoms (`6x12` atom AABB).
pub const PLAYER_HALF_SIZE: Vec2 = Vec2::new(3.0, 6.0);

/// Downward gravity acceleration per tick in Bones world space (`+Y` up, `-Y` down).
pub const PLAYER_GRAVITY_ACCEL: f32 = 0.36;

/// Terminal downward falling speed in atoms per tick.
pub const PLAYER_TERM_SPEED: f32 = 12.0;

/// Marker component for a player entity.
#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Player;

/// Identifies which slot `0..4` in [`MatchInputs`] drives this entity.
#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PlayerControlled {
    pub player_id: u8,
}

/// Compact 12-byte network-ready player input state for a single simulation tick.
///
/// Simulation systems must ONLY read [`MatchInputs`] and NEVER read raw `winit` keyboard/mouse
/// resources directly.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    HasSchema,
)]
#[schema(opaque)]
#[repr(C)]
pub struct PlayerInput {
    /// Horizontal movement axis (`-1` = Left, `0` = Neutral, `+1` = Right).
    pub move_x: i8,
    /// Vertical movement axis (`-1` = Down, `0` = Neutral, `+1` = Up).
    pub move_y: i8,
    /// Bitflags (`BTN_JUMP`, `BTN_JETPACK`, `BTN_PRIMARY`, `BTN_SECONDARY`, `BTN_INTERACT`).
    pub buttons: u8,
    /// Absolute hotbar slot (`0..=9`), never a relative scroll delta.
    pub selected_slot: u8,
    /// Normalized world-space aim direction relative to the player's position (`+Y` up).
    pub aim_dir: [f32; 2],
}

const _: [(); 12] = [(); std::mem::size_of::<PlayerInput>()];

impl PlayerInput {
    #[inline]
    pub const fn pressed(&self, button_mask: u8) -> bool {
        (self.buttons & button_mask) != 0
    }

    #[inline]
    pub fn aim_vec2(&self) -> Vec2 {
        Vec2::new(self.aim_dir[0], self.aim_dir[1]).normalize_or_zero()
    }
}

/// Per-tick synchronized inputs for all players in the session (`0..4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, HasSchema)]
#[schema(opaque)]
#[repr(C)]
pub struct MatchInputs {
    pub players: [PlayerInput; MAX_PLAYERS],
}

/// Client-only view and camera interpolation resource.
///
/// **Rule:** Simulation systems must NEVER read [`LocalView`].
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct LocalView {
    pub local_player_id: u8,
    pub camera_pos: Vec2,
    pub render_alpha: f32,
}

impl Default for LocalView {
    fn default() -> Self {
        Self {
            local_player_id: 0,
            camera_pos: Vec2::new(0.0, 200.0),
            render_alpha: 1.0,
        }
    }
}

/// Client-only input tracker used by [`gather_local_player_input_system`] to accumulate
/// event-based keyboard/mouse state between frames without polluting simulation state.
#[derive(HasSchema, Clone, Debug)]
#[schema(opaque)]
pub struct LocalInputState {
    pub held_keys: HashSet<KeyCode>,
    pub held_mouse: HashSet<MouseButton>,
    pub selected_slot: u8,
    pub last_aim_dir: [f32; 2],
    /// Whether mouse left/right click should drive `BTN_PRIMARY` / `BTN_SECONDARY` on the player.
    pub mouse_actions_enabled: bool,
    /// Whether `LocalView` should automatically control the scene `Camera` entity.
    pub camera_follow_player: bool,
}

impl Default for LocalInputState {
    fn default() -> Self {
        Self {
            held_keys: HashSet::new(),
            held_mouse: HashSet::new(),
            selected_slot: 1,
            last_aim_dir: [1.0, 0.0],
            mouse_actions_enabled: true,
            camera_follow_player: true,
        }
    }
}

/// Registers all player-specific schemas with the global `bones_schema` registry.
pub fn register_player_schemas() {
    Player::register_schema();
    PlayerControlled::register_schema();
    PlayerInput::register_schema();
    MatchInputs::register_schema();
    LocalView::register_schema();
    LocalInputState::register_schema();
}

/// Bones plugin that installs player spawning, input gathering, deterministic player controller,
/// [`KinematicActor`] movement sweep, world actions, sprite visuals, and [`LocalView`] camera.
#[derive(Default, Clone, Copy)]
pub struct AstratomicPlayerPlugin;

impl GamePlugin for AstratomicPlayerPlugin {
    fn install(self, _game: &mut Game) {
        register_player_schemas();
    }
}

impl SessionPlugin for AstratomicPlayerPlugin {
    fn install(self, session: &mut Session) {
        register_player_schemas();

        session.world.init_resource::<MatchInputs>();
        session.world.init_resource::<LocalView>();
        session.world.init_resource::<LocalInputState>();

        session
            .add_system_to_stage(CoreStage::Startup, setup_player_system)
            .add_system_to_stage(CoreStage::PreUpdate, gather_local_player_input_system)
            .add_system_to_stage(CoreStage::Update, player_controller_system)
            .add_system_to_stage(CoreStage::Update, kinematic_actor_movement_system)
            .add_system_to_stage(CoreStage::Update, player_world_action_system)
            .add_system_to_stage(CoreStage::PostUpdate, update_local_view_camera_system)
            .add_system_to_stage(CoreStage::PostUpdate, sync_player_visuals_system);
    }
}

/// Startup system that spawns the local player on top of the terrain surface using
/// [`PLAYER_STEP_HEIGHT`] for `step_height`.
pub fn setup_player_system(
    chunk_world: Res<ChunkWorld>,
    mut local_view: ResMut<LocalView>,
    mut sim_settings: ResMut<SimulationSettings>,
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut players: CompMut<Player>,
    mut player_controlled: CompMut<PlayerControlled>,
    mut actors: CompMut<KinematicActor>,
    mut actor_states: CompMut<ActorState>,
) {
    let spawn_pos = find_surface_spawn_pos(&chunk_world, 0.0, PLAYER_HALF_SIZE);

    let player_ent = entities.create();
    players.insert(player_ent, Player);
    player_controlled.insert(player_ent, PlayerControlled { player_id: 0 });
    actors.insert(
        player_ent,
        KinematicActor {
            pos: spawn_pos,
            prev_pos: spawn_pos,
            vel: Vec2::ZERO,
            half_size: PLAYER_HALF_SIZE,
            step_height: PLAYER_STEP_HEIGHT,
            grounded: true,
            ground_normal: Vec2::Y,
        },
    );
    actor_states.insert(player_ent, ActorState::default());
    transforms.insert(
        player_ent,
        Transform::from_translation(spawn_pos.extend(PLAYER_LAYER)),
    );

    local_view.local_player_id = 0;
    local_view.camera_pos = spawn_pos;
    local_view.render_alpha = 1.0;
    sim_settings.focus_world_pos = spawn_pos;
}

/// Pre-tick client input system that reads local keyboard/mouse + [`LocalView`] camera position,
/// computes the world-relative `aim_dir` from the local player's position, and writes to
/// `match_inputs.players[local_view.local_player_id as usize]`.
pub fn gather_local_player_input_system(
    keyboard: Option<Res<KeyboardInputs>>,
    mouse: Option<Res<MouseInputs>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    mouse_screen: Option<Res<MouseScreenPosition>>,
    window: Option<Res<Window>>,
    egui_ctx: Option<Res<EguiCtx>>,
    local_view: Res<LocalView>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    mut local_input: ResMut<LocalInputState>,
    mut match_inputs: ResMut<MatchInputs>,
) {
    let player_idx = (local_view.local_player_id as usize).min(MAX_PLAYERS - 1);

    if let Some(ref window) = window
        && !window.focused
    {
        local_input.held_keys.clear();
        local_input.held_mouse.clear();
        match_inputs.players[player_idx] = PlayerInput {
            selected_slot: local_input.selected_slot,
            aim_dir: local_input.last_aim_dir,
            ..PlayerInput::default()
        };
        return;
    }

    if let Some(keyboard) = keyboard {
        for ev in &keyboard.key_events {
            if let Set(code) = ev.key_code {
                if ev.button_state.pressed() {
                    local_input.held_keys.insert(code);
                } else {
                    local_input.held_keys.remove(&code);
                }
            }
        }
    }

    if let Some(mouse) = mouse.as_ref() {
        for ev in &mouse.button_events {
            if ev.state.pressed() {
                local_input.held_mouse.insert(ev.button);
            } else {
                local_input.held_mouse.remove(&ev.button);
            }
        }
    }

    let egui_wants_kb = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_keyboard_input());
    let egui_wants_pointer = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_pointer_input() || ctx.is_pointer_over_area());

    // Hotbar slot selection (`0..=9`)
    if !egui_wants_kb {
        let slot_keys = [
            (KeyCode::Key0, 0_u8),
            (KeyCode::Key1, 1_u8),
            (KeyCode::Key2, 2_u8),
            (KeyCode::Key3, 3_u8),
            (KeyCode::Key4, 4_u8),
            (KeyCode::Key5, 5_u8),
            (KeyCode::Key6, 6_u8),
            (KeyCode::Key7, 7_u8),
            (KeyCode::Key8, 8_u8),
            (KeyCode::Key9, 9_u8),
        ];
        for (key, slot) in slot_keys {
            if local_input.held_keys.contains(&key) {
                local_input.selected_slot = slot;
            }
        }
    }

    let mut move_x: i8 = 0;
    let mut move_y: i8 = 0;
    let mut buttons: u8 = 0;

    if !egui_wants_kb {
        if local_input.held_keys.contains(&KeyCode::A)
            || local_input.held_keys.contains(&KeyCode::Left)
        {
            move_x -= 1;
        }
        if local_input.held_keys.contains(&KeyCode::D)
            || local_input.held_keys.contains(&KeyCode::Right)
        {
            move_x += 1;
        }
        if local_input.held_keys.contains(&KeyCode::S)
            || local_input.held_keys.contains(&KeyCode::Down)
        {
            move_y -= 1;
        }
        if local_input.held_keys.contains(&KeyCode::W)
            || local_input.held_keys.contains(&KeyCode::Up)
        {
            move_y += 1;
        }

        if local_input.held_keys.contains(&KeyCode::Space)
            || local_input.held_keys.contains(&KeyCode::W)
            || local_input.held_keys.contains(&KeyCode::Up)
        {
            buttons |= BTN_JUMP;
        }

        if local_input.held_keys.contains(&KeyCode::ShiftLeft)
            || local_input.held_keys.contains(&KeyCode::ShiftRight)
            || local_input.held_keys.contains(&KeyCode::Space)
        {
            buttons |= BTN_JETPACK;
        }

        if local_input.held_keys.contains(&KeyCode::E)
            || local_input.held_keys.contains(&KeyCode::F)
        {
            buttons |= BTN_INTERACT;
        }
    }

    if !egui_wants_pointer && local_input.mouse_actions_enabled {
        if local_input.held_mouse.contains(&MouseButton::Left) {
            buttons |= BTN_PRIMARY;
        }
        if local_input.held_mouse.contains(&MouseButton::Right) {
            buttons |= BTN_SECONDARY;
        }
    }

    // Find the local player's world position for computing world-relative `aim_dir`.
    let mut player_pos = local_view.camera_pos;
    for (_ent, (ctrl, actor)) in entities.iter_with((&player_controlled, &actors)) {
        if ctrl.player_id == local_view.local_player_id {
            player_pos = actor.pos;
            break;
        }
    }

    // Resolve world-space cursor position from `MouseWorldPosition` or from `LocalView::camera_pos`.
    let cursor_world = mouse_world.and_then(|mw| mw.0).or_else(|| {
        let screen = mouse_screen?.0?;
        let win = window?;
        if win.size.x <= 1.0 || win.size.y <= 1.0 {
            return None;
        }
        let ndc = Vec2::new(
            (screen.x / win.size.x) - 0.5,
            0.5 - (screen.y / win.size.y),
        );
        let aspect = win.size.x / win.size.y;
        let view_height = 420.0;
        Some(
            local_view.camera_pos
                + Vec2::new(ndc.x * view_height * aspect, ndc.y * view_height),
        )
    });

    if let Some(target_world) = cursor_world {
        let delta = target_world - player_pos;
        if delta.length_squared() > 1e-4 {
            let dir = delta.normalize();
            local_input.last_aim_dir = [dir.x, dir.y];
        }
    }

    match_inputs.players[player_idx] = PlayerInput {
        move_x,
        move_y,
        buttons,
        selected_slot: local_input.selected_slot.min(9),
        aim_dir: local_input.last_aim_dir,
    };
}

/// Deterministic simulation system that reads [`MatchInputs`] and updates player velocity,
/// jumping, jetpack thrust, gravity, and integer tick cooldowns.
///
/// Players are collected and sorted by `player_id` (`0, 1, 2, 3`) before executing.
pub fn player_controller_system(
    sim_settings: Option<Res<SimulationSettings>>,
    match_inputs: Res<MatchInputs>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    mut actors: CompMut<KinematicActor>,
    mut states: CompMut<ActorState>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    // Collect and sort deterministically by `player_id` (0, 1, 2, 3).
    let mut ordered_players: Vec<(u8, Entity)> = entities
        .iter_with(&player_controlled)
        .map(|(ent, ctrl)| (ctrl.player_id, ent))
        .collect();
    ordered_players.sort_by_key(|&(player_id, ent)| (player_id, ent));

    for (player_id, ent) in ordered_players {
        let Some(input) = match_inputs.players.get(player_id as usize).copied() else {
            continue;
        };
        let Some(actor) = actors.get_mut(ent) else {
            continue;
        };
        let Some(state) = states.get_mut(ent) else {
            continue;
        };

        // 1. Decrement integer cooldown ticks deterministically.
        state.fire_cooldown_ticks = state.fire_cooldown_ticks.saturating_sub(1);

        // 2. Horizontal movement from `input.move_x`.
        let target_vx = (input.move_x.clamp(-1, 1) as f32) * RUN_SPEED;
        actor.vel.x = target_vx;

        // 3. Refuel jetpack when grounded.
        if actor.grounded {
            state.jetpack_fuel =
                (state.jetpack_fuel + FUEL_REGEN).clamp(0.0, state.max_jetpack_fuel);
        }

        // 4. Jump (allowed when `grounded` or within `coyote_ticks > 0`).
        let can_jump = actor.grounded || state.coyote_ticks > 0;
        let mut jumped_this_tick = false;
        if input.pressed(BTN_JUMP) && can_jump {
            actor.vel.y = JUMP_MAG;
            actor.grounded = false;
            state.coyote_ticks = 0;
            jumped_this_tick = true;
        }

        // 5. Jetpack thrust when airborne and holding `BTN_JETPACK`.
        let jetpack_requested = input.pressed(BTN_JETPACK)
            && !jumped_this_tick
            && !actor.grounded
            && state.coyote_ticks == 0
            && (input.move_y > 0 || actor.vel.y <= 1.2);
        if jetpack_requested && state.jetpack_fuel > 0.0 {
            actor.vel.y = (actor.vel.y + JETPACK_FORCE * 0.45).min(JETPACK_MAX);
            state.jetpack_fuel = (state.jetpack_fuel - FUEL_COMSUMPTON).max(0.0);
        }

        // 6. Apply downward gravity (`+Y` up, so gravity subtracts from `vel.y`).
        actor.vel.y = (actor.vel.y - PLAYER_GRAVITY_ACCEL).max(-PLAYER_TERM_SPEED);
    }
}

/// Maps a hotbar slot (`0..=9`) to a paintable material ID.
#[inline]
pub const fn hotbar_slot_to_material_id(slot: u8) -> u8 {
    match slot {
        0 => 2,  // Sand
        1 => 2,  // Sand
        2 => 3,  // Water
        3 => 4,  // Gravel
        4 => 5,  // Lava
        5 => 6,  // Grass
        6 => 7,  // Dirt
        7 => 8,  // Rock
        8 => 9,  // Acid
        9 => 10, // Wood
        _ => 2,
    }
}

/// Deterministic system where players mutate [`ChunkWorld`] (digging with `BTN_PRIMARY` or
/// placing/shooting atoms with `BTN_SECONDARY` along `aim_dir`).
///
/// Players are collected and sorted by `player_id` (`0, 1, 2, 3`) before executing so world
/// mutations are 100% deterministic across peers.
pub fn player_world_action_system(
    sim_settings: Option<Res<SimulationSettings>>,
    match_inputs: Res<MatchInputs>,
    materials: Res<Materials>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    mut states: CompMut<ActorState>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    let mut ordered_players: Vec<(u8, Entity)> = entities
        .iter_with(&player_controlled)
        .map(|(ent, ctrl)| (ctrl.player_id, ent))
        .collect();
    ordered_players.sort_by_key(|&(player_id, ent)| (player_id, ent));

    for (player_id, ent) in ordered_players {
        let Some(input) = match_inputs.players.get(player_id as usize).copied() else {
            continue;
        };
        let Some(actor) = actors.get(ent) else {
            continue;
        };
        let Some(state) = states.get_mut(ent) else {
            continue;
        };

        let aim = input.aim_vec2();
        if aim == Vec2::ZERO {
            continue;
        }

        // Primary action: excavate atoms along a short beam in front of the player.
        if input.pressed(BTN_PRIMARY) {
            let start_world = actor.pos + aim * 4.0;
            let end_world = actor.pos + aim * 24.0;
            let start_cell =
                ivec2(start_world.x.floor() as i32, start_world.y.floor() as i32);
            let end_cell = ivec2(end_world.x.floor() as i32, end_world.y.floor() as i32);

            for cell in Line::points_inclusive(start_cell, end_cell) {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let global = world_cell_to_global_atom(cell.x + dx, cell.y + dy);
                        let chunk_pos = global_to_chunk(global);
                        if let Some(atom) = chunk_world.get_atom(&chunk_pos)
                            && !atom.is_void()
                        {
                            chunk_world.set_atom(chunk_pos, Atom::default());
                            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
                            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
                        }
                    }
                }
            }
        }

        // Secondary action: deposit/emit atoms of the selected hotbar slot along `aim_dir`.
        if input.pressed(BTN_SECONDARY) && state.fire_cooldown_ticks == 0 {
            let mat_id = hotbar_slot_to_material_id(input.selected_slot);
            let spawn_world = actor.pos + aim * 10.0;
            let base_wx = spawn_world.x.floor() as i32;
            let base_wy = spawn_world.y.floor() as i32;
            let launch_vx = (aim.x * 4.0).round().clamp(-10.0, 10.0) as i8;
            // In `ChunkWorld`, `+vy` is downward, while `aim.y` is upward (`+Y` up).
            let launch_vy = (-aim.y * 4.0).round().clamp(-10.0, 10.0) as i8;

            for oy in -1..=1 {
                for ox in -1..=1 {
                    let global = world_cell_to_global_atom(base_wx + ox, base_wy + oy);
                    let chunk_pos = global_to_chunk(global);
                    if let Some(existing) = chunk_world.get_atom(&chunk_pos)
                        && existing.is_void()
                    {
                        let mut new_atom = Atom::new(mat_id, &materials);
                        if new_atom.is_powder() || new_atom.is_liquid() {
                            new_atom.vx = launch_vx;
                            new_atom.vy = launch_vy;
                            new_atom.set_moving(true);
                        }
                        chunk_world.set_atom(chunk_pos, new_atom);
                        update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
                        update_dirty_rects(&mut dirty_rects.render, chunk_pos);
                    }
                }
            }
            state.fire_cooldown_ticks = 3;
        }
    }
}

/// Post-simulation client-only camera & render interpolation system.
///
/// Finds the entity with `player_id == local_view.local_player_id`, computes
/// `interp_pos = actor.prev_pos.lerp(actor.pos, local_view.render_alpha)`, smoothly updates
/// `local_view.camera_pos`, and updates actor [`Transform`] components for rendering.
pub fn update_local_view_camera_system(
    time: Res<Time>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    local_input: Option<Res<LocalInputState>>,
    mut local_view: ResMut<LocalView>,
    mut sim_settings: Option<ResMut<SimulationSettings>>,
    mut cameras: CompMut<Camera>,
    mut transforms: CompMut<Transform>,
) {
    let alpha = local_view.render_alpha.clamp(0.0, 1.0);
    let dt = time.delta_seconds().clamp(0.0001, 0.1);

    // 1. Update render `Transform` for all `KinematicActor` entities using `prev_pos.lerp(pos, alpha)`.
    let mut local_player_interp_pos = None;
    for (ent, actor) in entities.iter_with(&actors) {
        let interp_pos = actor.prev_pos.lerp(actor.pos, alpha);
        if let Some(transform) = transforms.get_mut(ent) {
            transform.translation.x = interp_pos.x;
            transform.translation.y = interp_pos.y;
        }
        if let Some(ctrl) = player_controlled.get(ent)
            && ctrl.player_id == local_view.local_player_id
        {
            local_player_interp_pos = Some(interp_pos);
        }
    }

    // 2. Smoothly update `local_view.camera_pos` toward the local player's interpolated position.
    if let Some(interp_pos) = local_player_interp_pos {
        let follow_speed = 12.0;
        let t = (1.0 - (-follow_speed * dt).exp()).clamp(0.0, 1.0);
        if (local_view.camera_pos - interp_pos).length_squared() > 250_000.0 {
            local_view.camera_pos = interp_pos;
        } else {
            local_view.camera_pos = local_view.camera_pos.lerp(interp_pos, t);
        }

        let follow_enabled = local_input
            .as_ref()
            .is_none_or(|li| li.camera_follow_player);

        if follow_enabled {
            for (cam_ent, _camera) in entities.iter_with(&mut cameras) {
                if let Some(transform) = transforms.get_mut(cam_ent) {
                    transform.translation.x = local_view.camera_pos.x;
                    transform.translation.y = local_view.camera_pos.y;
                }
            }
            if let Some(ref mut settings) = sim_settings {
                settings.focus_world_pos = local_view.camera_pos;
            }
        }
    }
}

/// Synchronizes `AtlasSprite` animation frames and facing direction for all player entities.
pub fn sync_player_visuals_system(
    time: Res<Time>,
    asset_server: Option<Res<AssetServer>>,
    match_inputs: Res<MatchInputs>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    actor_states: Comp<ActorState>,
    mut atlas_sprites: CompMut<AtlasSprite>,
) {
    let player_atlas_handle = asset_server.and_then(|server| {
        let core_guard = server.store.core_pack.lock();
        let pack = core_guard.as_ref()?;
        let root_handle = pack.root.typed::<GameMeta>();
        let meta = server.try_get(root_handle)?.ok()?;
        Some(meta.player_atlas)
    });

    let anim_tick = (time.elapsed_seconds() * 10.0) as u32;

    for (ent, (ctrl, actor)) in entities.iter_with((&player_controlled, &actors)) {
        if atlas_sprites.get(ent).is_none()
            && let Some(atlas_handle) = player_atlas_handle
        {
            atlas_sprites.insert(ent, AtlasSprite::new(atlas_handle));
        }

        let Some(sprite) = atlas_sprites.get_mut(ent) else {
            continue;
        };

        let input = match_inputs
            .players
            .get(ctrl.player_id as usize)
            .copied()
            .unwrap_or_default();

        if input.aim_dir[0].abs() > 0.05 {
            sprite.flip_x = input.aim_dir[0] < 0.0;
        } else if actor.vel.x.abs() > 0.05 {
            sprite.flip_x = actor.vel.x < 0.0;
        }

        let state = actor_states.get(ent);
        let using_jetpack = !actor.grounded
            && input.pressed(BTN_JETPACK)
            && state.is_some_and(|s| s.jetpack_fuel > 0.0);

        let (first_frame, last_frame): (u32, u32) = if using_jetpack {
            (24, 26)
        } else if !actor.grounded {
            (16, 23)
        } else if actor.vel.x.abs() > 0.1 {
            (8, 11)
        } else {
            (0, 1)
        };

        let span = (last_frame - first_frame + 1).max(1);
        sprite.index = first_frame + (anim_tick % span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_input_is_12_bytes_and_pod() {
        assert_eq!(std::mem::size_of::<PlayerInput>(), 12);
        let input = PlayerInput {
            move_x: -1,
            move_y: 1,
            buttons: BTN_JUMP | BTN_PRIMARY,
            selected_slot: 3,
            aim_dir: [0.6, 0.8],
        };
        let bytes = bytemuck::bytes_of(&input);
        assert_eq!(bytes.len(), 12);
        let roundtrip: PlayerInput = *bytemuck::from_bytes(bytes);
        assert_eq!(roundtrip, input);
    }
}
