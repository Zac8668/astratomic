//! Player-specific components, network-ready `PlayerInput` / `MatchInputs`, deterministic
//! player controller & world action systems, `AtlasSprite` animations, and `LocalView` camera.

use std::collections::HashSet;

use astratomic_physics::prelude::{
    ActorSimulationStamp, ActorState, KinematicActor, clear_actor_objects_system,
    clear_rigidbody_objects_system, find_surface_spawn_pos, kinematic_actor_movement_system,
    stamp_actor_objects_system, stamp_rigidbody_objects_system, step_rapier_physics_system,
};
pub use astratomic_sim::consts::PLAYER_STEP_HEIGHT;
use astratomic_sim::{
    atom::{Atom, Materials},
    consts::{
        FUEL_COMSUMPTON, FUEL_REGEN, JETPACK_FORCE, JETPACK_MAX, JUMP_MAG, PLAYER_LAYER, RUN_SPEED,
        TOOL_DISTANCE, TOOL_RANGE,
    },
    geom_tools::Line,
    particle::{PARTICLE_IGNORE_WORLD_COLLISION, Particle, ParticleActorTargets, ParticlePool},
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

/// Maximum reach of the player tool in world units (atoms).
pub const TOOL_MAX_REACH: f32 = 40.0;

/// Horizontal visual and firing offset of the player tool relative to the actor center.
pub const PLAYER_TOOL_OFFSET_X: f32 = 5.0;

/// Maps a player slot `player_id` (`0..=3`) to its deterministic non-zero `target_actor_id` (`1..=4`).
#[inline]
pub const fn player_actor_id(player_id: u8) -> u8 {
    player_id + 1
}

/// Default half-extents of a player's hitbox in atoms (`6x12` atom AABB).
pub const PLAYER_HALF_SIZE: Vec2 = Vec2::new(4.4, 7.0);

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

/// Marker component on an entity displaying a player's tool sprite.
#[derive(HasSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PlayerTool {
    /// The player entity that owns this tool.
    pub player: Entity,
}

/// Compact 12-byte network-ready player input state for a single simulation tick.
///
/// Simulation systems must ONLY read [`MatchInputs`] and NEVER read raw `winit` keyboard/mouse
/// resources directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable, HasSchema)]
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
    PlayerTool::register_schema();
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
        session.world.init_resource::<ActorSimulationStamp>();

        session
            .add_system_to_stage(CoreStage::Startup, setup_player_system)
            .add_system_to_stage(CoreStage::PreUpdate, gather_local_player_input_system)
            .add_system_to_stage(CoreStage::PreUpdate, player_controller_system)
            .add_system_to_stage(CoreStage::PreUpdate, step_rapier_physics_system)
            .add_system_to_stage(CoreStage::PreUpdate, kinematic_actor_movement_system)
            .add_system_to_stage(CoreStage::PreUpdate, sync_actor_particle_targets_system)
            .add_system_to_stage(CoreStage::PreUpdate, player_world_action_system)
            .add_system_to_stage(CoreStage::PreUpdate, player_throw_particles_system)
            .add_system_to_stage(CoreStage::PreUpdate, stamp_actor_objects_system)
            .add_system_to_stage(CoreStage::PreUpdate, stamp_rigidbody_objects_system)
            .add_system_to_stage(CoreStage::Update, clear_actor_objects_system)
            .add_system_to_stage(CoreStage::Update, clear_rigidbody_objects_system)
            .add_system_to_stage(CoreStage::PostUpdate, update_local_view_camera_system)
            .add_system_to_stage(CoreStage::PostUpdate, sync_pixel_rigidbodies_render_system)
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
    mut player_tools: CompMut<PlayerTool>,
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
            weight: 70.0,
            grounded: true,
            ground_normal: Vec2::Y,
        },
    );
    actor_states.insert(player_ent, ActorState::default());
    transforms.insert(
        player_ent,
        Transform::from_translation(spawn_pos.extend(PLAYER_LAYER)),
    );

    let tool_ent = entities.create();
    player_tools.insert(tool_ent, PlayerTool { player: player_ent });
    transforms.insert(
        tool_ent,
        Transform::from_translation(Vec3::new(
            spawn_pos.x - 1.0,
            spawn_pos.y - 3.5,
            PLAYER_LAYER + 0.1,
        )),
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
        let ndc = Vec2::new((screen.x / win.size.x) - 0.5, 0.5 - (screen.y / win.size.y));
        let aspect = win.size.x / win.size.y;
        let view_height = 420.0;
        Some(local_view.camera_pos + Vec2::new(ndc.x * view_height * aspect, ndc.y * view_height))
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
            state.jumped_this_tick = true;
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

/// PreUpdate system that populates [`ParticleActorTargets`] from active [`PlayerControlled`]
/// [`KinematicActor`] positions and tool nozzles so homing particles navigate deterministically.
pub fn sync_actor_particle_targets_system(
    match_inputs: Option<Res<MatchInputs>>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    mut actor_targets: ResMut<ParticleActorTargets>,
) {
    actor_targets.clear();

    let mut ordered_players: Vec<(u8, Entity)> = entities
        .iter_with(&player_controlled)
        .map(|(ent, ctrl)| (ctrl.player_id, ent))
        .collect();
    ordered_players.sort_by_key(|&(player_id, ent)| (player_id, ent));

    for (player_id, ent) in ordered_players {
        if let Some(actor) = actors.get(ent) {
            // In ChunkWorld, +Y is downward, so actor.pos (Bones world +Y up) maps to [actor.pos.x, -actor.pos.y].
            let aim = match_inputs
                .as_ref()
                .and_then(|inputs| inputs.players.get(player_id as usize))
                .map(|i| i.aim_vec2())
                .unwrap_or(Vec2::ZERO);

            let facing_left = if aim.x.abs() > 0.05 {
                aim.x < 0.0
            } else {
                actor.vel.x < -0.05
            };
            let tool_offset_x = if facing_left { -PLAYER_TOOL_OFFSET_X } else { PLAYER_TOOL_OFFSET_X };
            let tool_origin = Vec2::new(actor.pos.x + tool_offset_x, -actor.pos.y);
            let mut center_pos = [tool_origin.x, tool_origin.y];
            if aim != Vec2::ZERO {
                let tool_slope = Vec2::new(aim.x, -aim.y).normalize();
                let tool_front = tool_origin + tool_slope * 5.0;
                center_pos = [tool_front.x, tool_front.y];
            }
            actor_targets.set_target(player_actor_id(player_id), center_pos, 6.0);
        }
    }
}

/// Deterministic system for throwing matter particles along player aim directions (`BTN_SECONDARY`).
pub fn player_throw_particles_system(
    sim_settings: Option<Res<SimulationSettings>>,
    match_inputs: Res<MatchInputs>,
    materials: Res<Materials>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    chunk_world: Res<ChunkWorld>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
) {
    if sim_settings
        .as_ref()
        .is_some_and(|s| s.paused && !s.step_once)
    {
        return;
    }

    let Some(ref mut pool) = particle_pool else {
        return;
    };

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

        if !input.pressed(BTN_SECONDARY) {
            continue;
        }

        let aim = input.aim_vec2();
        if aim == Vec2::ZERO {
            continue;
        }

        let tool_slope = Vec2::new(aim.x, -aim.y).normalize();
        let bound_slope = Vec2::new(-tool_slope.y, tool_slope.x);
        let tool_offset_x = if aim.x < 0.0 { -PLAYER_TOOL_OFFSET_X } else { PLAYER_TOOL_OFFSET_X };
        let player_center = Vec2::new(actor.pos.x + tool_offset_x, -actor.pos.y);
        let tool_front = player_center + tool_slope * 5.0;
        let new_tool_front = tool_front + tool_slope * 3.5;

        let world_seed = chunk_world.seed as u64;
        let current_tick = chunk_world.tick;
        let mat_id = hotbar_slot_to_material_id(input.selected_slot);

        let n = 4;
        for i in 0..=n {
            let seed = (world_seed ^ (current_tick << 16))
                .wrapping_add((player_id as u64) << 48)
                .wrapping_add((i as u64) << 32)
                .wrapping_add(0x9E3779B97F4A7C15);
            let mut rng = fastrand::Rng::with_seed(seed);

            let rand_angle = rng.f32() * std::f32::consts::TAU;
            let mut spawn_vec = new_tool_front - bound_slope * 2.0
                + bound_slope * (4.0 * (i as f32 / n as f32))
                + Vec2::new(rand_angle.cos(), rand_angle.sin()) * 0.8;

            if tool_slope.y > 0.0 {
                spawn_vec += tool_slope * (4.0 * tool_slope.y);
            }

            let chunk_pos = global_to_chunk(spawn_vec.as_ivec2());
            if let Some(existing) = chunk_world.get_atom(&chunk_pos) {
                if existing.is_air() || existing.is_object() {
                    let angle_offset = rng.f32() * 0.4 - 0.2;
                    let speed = 9.5 * (rng.f32() * 0.25 + 0.85);

                    let cos_a = angle_offset.cos();
                    let sin_a = angle_offset.sin();
                    let vel = [
                        (tool_slope.x * cos_a - tool_slope.y * sin_a) * speed,
                        (tool_slope.x * sin_a + tool_slope.y * cos_a) * speed,
                    ];

                    let mut atom = Atom::new_with_rng(mat_id, &materials, &mut rng);
                    atom.set_free_falling(true);
                    atom.set_life_5bit(0);
                    atom.vx = (vel[0] * 0.5).round().clamp(-6.0, 6.0) as i8;
                    atom.vy = (vel[1] * 0.5).round().clamp(-6.0, 6.0) as i8;

                    pool.push_particle(Particle::new([spawn_vec.x, spawn_vec.y], vel, atom));
                }
            }
        }
    }
}

/// Deterministic system where players operate the matter manipulator tool (sucking atoms
/// from terrain with `BTN_PRIMARY` along `aim_dir`).
///
/// Players are collected and sorted by `player_id` (`0, 1, 2, 3`) before executing so world
/// mutations and particle spawning are 100% deterministic across peers.
pub fn player_world_action_system(
    sim_settings: Option<Res<SimulationSettings>>,
    match_inputs: Res<MatchInputs>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    mut states: CompMut<ActorState>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
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
        let Some(_state) = states.get_mut(ent) else {
            continue;
        };

        if !input.pressed(BTN_PRIMARY) {
            continue;
        }

        let aim = input.aim_vec2();
        if aim == Vec2::ZERO {
            continue;
        }

        // In ChunkWorld, +Y is downward, so Bones world (+Y up) maps to (aim.x, -aim.y).
        let tool_slope = Vec2::new(aim.x, -aim.y).normalize();
        let bound_slope = Vec2::new(-tool_slope.y, tool_slope.x);
        let tool_offset_x = if aim.x < 0.0 { -PLAYER_TOOL_OFFSET_X } else { PLAYER_TOOL_OFFSET_X };
        let player_center = Vec2::new(actor.pos.x + tool_offset_x, -actor.pos.y);
        let tool_front = player_center + tool_slope * 5.0;

        let center_bound = tool_front + tool_slope * TOOL_DISTANCE;
        let bound1 = (center_bound + bound_slope * TOOL_RANGE).as_ivec2();
        let bound2 = (center_bound - bound_slope * TOOL_RANGE).as_ivec2();

        let ray_start = (tool_front - tool_slope * 4.0).as_ivec2();
        let tool_origin_check = tool_front - tool_slope * 6.0;
        let min_dist_sq = 6.0 * 6.0;

        for bound_vec in Line::new(bound1, bound2 - bound1) {
            for vec in Line::new(ray_start, bound_vec - ray_start) {
                if (vec.as_vec2() - tool_origin_check).length_squared() < min_dist_sq {
                    continue;
                }

                let chunk_pos = global_to_chunk(vec);
                if let Some(&atom) = chunk_world.get_atom(&chunk_pos) {
                    if !atom.is_air() && !atom.is_object() {
                        // Eject from ChunkWorld
                        chunk_world.set_atom(chunk_pos, Atom::AIR);
                        update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
                        update_dirty_rects(&mut dirty_rects.render, chunk_pos);

                        // Wake adjacent powder neighbors so terrain caves in naturally
                        for (ndx, ndy) in [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1)] {
                            let n_cp = global_to_chunk(ivec2(vec.x + ndx, vec.y + ndy));
                            if let Some(&neighbor) = chunk_world.get_atom(&n_cp) {
                                if neighbor.is_powder() && !neighbor.is_free_falling() {
                                    let mut waking = neighbor;
                                    waking.set_free_falling(true);
                                    waking.set_moving(true);
                                    chunk_world.set_atom(n_cp, waking);
                                }
                            }
                        }

                        // Spawn homing particle targeting the player nozzle
                        if let Some(ref mut pool) = particle_pool {
                            let mut matter_atom = atom;
                            matter_atom.set_free_falling(true);
                            matter_atom.set_life_5bit(0);
                            matter_atom.vx = 0;
                            matter_atom.vy = 0;

                            pool.push_particle(Particle::new_homing(
                                [vec.x as f32 + 0.5, vec.y as f32 + 0.5],
                                [0.0, 0.0],
                                matter_atom,
                                player_actor_id(player_id),
                                PARTICLE_IGNORE_WORLD_COLLISION,
                            ));
                        }

                        // Break the ray after first non-air atom hit (peels surface layer)
                        break;
                    }
                }
            }
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

static NEXT_RIGIDBODY_CID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_rigidbody_cid() -> Cid {
    let id = NEXT_RIGIDBODY_CID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(b"ASTR_RB_");
    bytes[8..16].copy_from_slice(&id.to_le_bytes());
    Cid(bytes)
}

thread_local! {
    static RIGIDBODY_SCRATCH_BUFFER: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Interpolates, inverse-rasterizes, and synchronizes world-grid aligned textures and transforms
/// for all active [`PhysicsBodyHandle`] entities with zero rotation holes and zero moiré gaps.
pub fn sync_pixel_rigidbodies_render_system(
    local_view: Res<LocalView>,
    asset_server: Option<Res<AssetServer>>,
    entities: Res<Entities>,
    handles: Comp<astratomic_physics::rigidbody::PhysicsBodyHandle>,
    mut pixel_bodies: CompMut<astratomic_physics::rigidbody::PixelRigidBody>,
    mut transforms: CompMut<Transform>,
    mut sprites: CompMut<Sprite>,
) {
    let alpha = local_view.render_alpha.clamp(0.0, 1.0);

    RIGIDBODY_SCRATCH_BUFFER.with(|buf_cell| {
        let mut scratch = buf_cell.borrow_mut();

        for (ent, handle) in entities.iter_with(&handles) {
            let Some(pbody) = pixel_bodies.get_mut(ent) else {
                continue;
            };

            // Interpolated position and shortest-arc rotation angle
            let render_pos = handle.prev_pos.lerp(handle.pos, alpha);
            let diff = (handle.angle - handle.prev_angle + std::f32::consts::PI)
                .rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            let render_angle = handle.prev_angle + diff * alpha;

            let (sin_a, cos_a) = render_angle.sin_cos();
            let half_w = pbody.width as f32 * 0.5;
            let half_h = pbody.height as f32 * 0.5;
            let x_extent = half_w * cos_a.abs() + half_h * sin_a.abs();
            let y_extent = half_w * sin_a.abs() + half_h * cos_a.abs();

            let min_wx = (render_pos.x - x_extent).floor() as i32;
            let max_wx = (render_pos.x + x_extent - 1e-4).floor() as i32;
            let min_wy = (render_pos.y - y_extent).floor() as i32;
            let max_wy = (render_pos.y + y_extent - 1e-4).floor() as i32;

            let bbox_w = (max_wx - min_wx + 1).max(1) as u32;
            let bbox_h = (max_wy - min_wy + 1).max(1) as u32;
            let needed_len = (bbox_w as usize) * (bbox_h as usize) * 4;

            if scratch.len() < needed_len {
                scratch.resize(needed_len, 0);
            }
            scratch[..needed_len].fill(0);

            // Inverse rasterization: sample backward from each integer world cell into local grid
            for wy in min_wy..=max_wy {
                let img_y = (max_wy - wy) as usize;
                for wx in min_wx..=max_wx {
                    if let Some(atom) =
                        pbody.sample_rigidbody_at_world_cell(wx, wy, render_pos, sin_a, cos_a)
                    {
                        if !atom.is_air() {
                            let img_x = (wx - min_wx) as usize;
                            let idx = (img_y * (bbox_w as usize) + img_x) * 4;
                            scratch[idx..idx + 4].copy_from_slice(&atom.color);
                        }
                    }
                }
            }

            // Transform center aligned exactly with the world-space bounding box
            let center_x = min_wx as f32 + bbox_w as f32 * 0.5;
            let center_y = min_wy as f32 + bbox_h as f32 * 0.5;

            if let Some(transform) = transforms.get_mut(ent) {
                transform.translation.x = center_x;
                transform.translation.y = center_y;
                transform.translation.z = 1.0;
                transform.rotation = Quat::IDENTITY;
            }

            if let Some(ref server) = asset_server {
                let new_cid = next_rigidbody_cid();
                let untyped = if let Some(h) = pbody.image_handle {
                    h.untyped()
                } else {
                    let handle = UntypedHandle {
                        rid: Ulid::create(),
                    }
                    .typed::<Image>();
                    pbody.image_handle = Some(handle);
                    sprites.insert(
                        ent,
                        Sprite {
                            color: Color::WHITE,
                            image: handle,
                            flip_x: false,
                            flip_y: false,
                        },
                    );
                    handle.untyped()
                };

                if let Some(old_cid) = pbody.render_cid {
                    server.store.assets.remove(&old_cid);
                }
                pbody.render_cid = Some(new_cid);

                let img_data =
                    image::RgbaImage::from_raw(bbox_w, bbox_h, scratch[..needed_len].to_vec())
                        .unwrap();
                let image = Image::Data(image::DynamicImage::ImageRgba8(img_data));

                server.store.assets.insert(
                    new_cid,
                    LoadedAsset {
                        cid: new_cid,
                        pack_spec: None,
                        loc: AssetLoc {
                            path: std::path::PathBuf::new(),
                            pack: None,
                        },
                        dependencies: Vec::new(),
                        data: SchemaBox::new(image),
                    },
                );
                server.store.asset_ids.insert(untyped, new_cid);
            }
        }
    });
}

/// Synchronizes `AtlasSprite` animation frames and facing direction for all player entities,
/// as well as positioning, rotation, and flipping for the player's tool sprite.
pub fn sync_player_visuals_system(
    time: Res<Time>,
    asset_server: Option<Res<AssetServer>>,
    match_inputs: Res<MatchInputs>,
    mut entities: ResMut<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    actor_states: Comp<ActorState>,
    mut transforms: CompMut<Transform>,
    mut atlas_sprites: CompMut<AtlasSprite>,
    mut player_tools: CompMut<PlayerTool>,
) {
    let (player_atlas_handle, tool_atlas_handle) = asset_server
        .and_then(|server| {
            let core_guard = server.store.core_pack.lock();
            let pack = core_guard.as_ref()?;
            let root_handle = pack.root.typed::<GameMeta>();
            let meta = server.try_get(root_handle)?.ok()?;
            Some((meta.player_atlas, meta.player_tool_atlas))
        })
        .unzip();

    // Clean up orphaned tools whose player entity was destroyed
    let dead_tools: Vec<Entity> = entities
        .iter_with(&player_tools)
        .filter(|(_t_ent, pt)| !entities.is_alive(pt.player))
        .map(|(t_ent, _)| t_ent)
        .collect();
    for t_ent in dead_tools {
        entities.kill(t_ent);
    }

    let anim_tick = (time.elapsed_seconds() * 10.0) as u32;

    let players_info: Vec<(Entity, u8, KinematicActor)> = entities
        .iter_with((&player_controlled, &actors))
        .map(|(ent, (ctrl, actor))| (ent, ctrl.player_id, *actor))
        .collect();

    for (ent, player_id, actor) in players_info {
        if atlas_sprites.get(ent).is_none()
            && let Some(atlas_handle) = player_atlas_handle
        {
            atlas_sprites.insert(ent, AtlasSprite::new(atlas_handle));
        }

        let input = match_inputs
            .players
            .get(player_id as usize)
            .copied()
            .unwrap_or_default();

        let aim = input.aim_vec2();
        let facing_left = if aim.x.abs() > 0.05 {
            aim.x < 0.0
        } else if actor.vel.x.abs() > 0.05 {
            actor.vel.x < 0.0
        } else {
            atlas_sprites.get(ent).is_some_and(|s| s.flip_x)
        };

        if let Some(sprite) = atlas_sprites.get_mut(ent) {
            sprite.flip_x = facing_left;

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

        // --- Tool Sprite Synchronization ---
        let tool_ent = entities
            .iter_with(&player_tools)
            .find(|(_t_ent, pt)| pt.player == ent)
            .map(|(t_ent, _)| t_ent)
            .unwrap_or_else(|| {
                let new_tool = entities.create();
                player_tools.insert(new_tool, PlayerTool { player: ent });
                transforms.insert(new_tool, Transform::default());
                new_tool
            });

        if atlas_sprites.get(tool_ent).is_none()
            && let Some(tool_atlas) = tool_atlas_handle
        {
            atlas_sprites.insert(tool_ent, AtlasSprite::new(tool_atlas));
        }

        let aim_norm = if aim != Vec2::ZERO {
            aim.normalize()
        } else {
            Vec2::new(if facing_left { -1.0 } else { 1.0 }, 0.0)
        };
        let angle = aim_norm.y.atan2(aim_norm.x);
        let flip_bool = angle.abs() > std::f32::consts::FRAC_PI_2;

        if let Some(tool_sprite) = atlas_sprites.get_mut(tool_ent) {
            tool_sprite.flip_y = flip_bool;
        }

        let player_trans = transforms
            .get(ent)
            .map(|t| t.translation)
            .unwrap_or_else(|| actor.pos.extend(PLAYER_LAYER));
        let tool_offset_x = if flip_bool { -PLAYER_TOOL_OFFSET_X } else { PLAYER_TOOL_OFFSET_X };
        let offset_x = (if flip_bool { 1.0 } else { -1.0 }) + tool_offset_x;
        let offset_y = -3.5;
        let tool_pivot = Vec2::new(player_trans.x + offset_x, player_trans.y + offset_y);
        let tool_pos = tool_pivot + aim_norm * 3.0;

        if let Some(tool_trans) = transforms.get_mut(tool_ent) {
            tool_trans.translation = Vec3::new(tool_pos.x, tool_pos.y, PLAYER_LAYER + 0.1);
            tool_trans.rotation = Quat::from_rotation_z(angle);
        }
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

    #[test]
    fn test_tool_excavates_terrain_and_spawns_homing_particles() {
        use astratomic_physics::prelude::world_cell_to_global_atom;
        use astratomic_sim::chunk::Chunk;
        use astratomic_sim::particle::{
            PARTICLE_FOLLOW_ACTOR, step_particle_simulation_with_targets,
        };

        register_player_schemas();
        astratomic_sim::plugin::register_sim_schemas();
        astratomic_physics::plugin::register_physics_schemas();

        let session = Session::default();
        session.world.init_resource::<MatchInputs>();
        session.world.init_resource::<ChunkWorld>();
        session.world.init_resource::<DirtyRects>();
        session.world.init_resource::<Materials>();
        session.world.init_resource::<ParticlePool>();
        session.world.init_resource::<ParticleActorTargets>();

        // Populate chunks
        {
            let mut world = session.world.resource_mut::<ChunkWorld>();
            for cy in -2..=2 {
                for cx in -2..=2 {
                    world.add_chunk(ivec2(cx, cy), Chunk::default());
                }
            }
            // Fill an obstacle patch of sand at wx in 8..=16, wy in -2..=2
            let sand = Atom::with_state(
                2,
                astratomic_sim::atom::AtomState::Powder,
                [230, 197, 92, 255],
            );
            for wx in 8..=16 {
                for wy in -2..=2 {
                    let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                    world.set_atom(cp, sand);
                }
            }
        }

        // Spawn player at (0.0, 0.0)
        let _player_ent = {
            let mut entities = session.world.resource_mut::<Entities>();
            let mut players = session.world.component_mut::<Player>();
            let mut player_controlled = session.world.component_mut::<PlayerControlled>();
            let mut actors = session.world.component_mut::<KinematicActor>();
            let mut states = session.world.component_mut::<ActorState>();

            let ent = entities.create();
            players.insert(ent, Player);
            player_controlled.insert(ent, PlayerControlled { player_id: 0 });
            actors.insert(
                ent,
                KinematicActor {
                    pos: Vec2::new(0.0, 0.0),
                    prev_pos: Vec2::new(0.0, 0.0),
                    vel: Vec2::ZERO,
                    half_size: PLAYER_HALF_SIZE,
                    step_height: PLAYER_STEP_HEIGHT,
                    weight: 70.0,
                    grounded: true,
                    ground_normal: Vec2::Y,
                },
            );
            states.insert(ent, ActorState::default());
            ent
        };

        // Aim player 0 right toward the sand with BTN_PRIMARY
        {
            let mut inputs = session.world.resource_mut::<MatchInputs>();
            inputs.players[0] = PlayerInput {
                move_x: 0,
                move_y: 0,
                buttons: BTN_PRIMARY,
                selected_slot: 1, // radius 3
                aim_dir: [1.0, 0.0],
            };
        }

        // Sync targets and run action system
        session
            .world
            .run_system(sync_actor_particle_targets_system, ());
        session.world.run_system(player_world_action_system, ());

        // Verify that homing matter particles were spawned into ParticlePool
        {
            let pool = session.world.resource::<ParticlePool>();
            let matter_particles: Vec<_> = pool
                .particles()
                .iter()
                .filter(|p| p.atom.life_5bit() == 0)
                .collect();
            assert!(
                !matter_particles.is_empty(),
                "excavator must spawn homing matter particles"
            );
            for p in matter_particles {
                assert_eq!(p.target_actor_id, player_actor_id(0));
                assert_ne!(p.flags & PARTICLE_FOLLOW_ACTOR, 0);
                assert_ne!(p.flags & PARTICLE_IGNORE_WORLD_COLLISION, 0);
            }
        }

        let targets = {
            let t = session.world.resource::<ParticleActorTargets>();
            t.targets.clone()
        };

        for tick in 1..=60 {
            let mut p_pool = session.world.resource_mut::<ParticlePool>();
            let mut c_world = session.world.resource_mut::<ChunkWorld>();
            let mut dirty = session.world.resource_mut::<DirtyRects>();
            step_particle_simulation_with_targets(
                &mut p_pool,
                &mut c_world,
                Some(&mut dirty),
                tick,
                &targets,
            );
            if p_pool.particles().iter().all(|p| p.atom.life_5bit() > 0) {
                break;
            }
        }

        let final_pool = session.world.resource::<ParticlePool>();
        let remaining_matter = final_pool
            .particles()
            .iter()
            .filter(|p| p.atom.life_5bit() == 0)
            .count();
        assert_eq!(
            remaining_matter, 0,
            "all homing matter particles must despawn upon reaching the player"
        );
    }

    #[test]
    fn test_tool_throws_particles_and_settles() {
        use astratomic_physics::prelude::world_cell_to_global_atom;
        use astratomic_sim::chunk::Chunk;
        use astratomic_sim::particle::step_particle_simulation;

        register_player_schemas();
        astratomic_sim::plugin::register_sim_schemas();
        astratomic_physics::plugin::register_physics_schemas();

        let session = Session::default();
        session.world.init_resource::<MatchInputs>();
        session.world.init_resource::<ChunkWorld>();
        session.world.init_resource::<DirtyRects>();
        session.world.init_resource::<Materials>();
        session.world.init_resource::<ParticlePool>();
        session.world.init_resource::<ParticleActorTargets>();

        {
            let mut world = session.world.resource_mut::<ChunkWorld>();
            for cy in -1..=1 {
                for cx in -1..=1 {
                    world.add_chunk(ivec2(cx, cy), Chunk::default());
                }
            }
            // Put a rock wall at wx = 20 (wy in -15..=15) in front of the player
            // and a floor at wy = -10 (wx in 0..=30) to catch falling particles
            let rock = Atom::with_state(
                8,
                astratomic_sim::atom::AtomState::Solid,
                [128, 128, 128, 255],
            );
            for wy in -15..=15 {
                let cp = global_to_chunk(world_cell_to_global_atom(20, wy));
                world.set_atom(cp, rock);
            }
            for wx in 0..=30 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, -10));
                world.set_atom(cp, rock);
            }
        }

        // Spawn player at (0.0, 0.0)
        {
            let mut entities = session.world.resource_mut::<Entities>();
            let mut players = session.world.component_mut::<Player>();
            let mut player_controlled = session.world.component_mut::<PlayerControlled>();
            let mut actors = session.world.component_mut::<KinematicActor>();
            let mut states = session.world.component_mut::<ActorState>();

            let ent = entities.create();
            players.insert(ent, Player);
            player_controlled.insert(ent, PlayerControlled { player_id: 0 });
            actors.insert(
                ent,
                KinematicActor {
                    pos: Vec2::new(0.0, 0.0),
                    prev_pos: Vec2::new(0.0, 0.0),
                    vel: Vec2::ZERO,
                    half_size: PLAYER_HALF_SIZE,
                    step_height: PLAYER_STEP_HEIGHT,
                    weight: 70.0,
                    grounded: true,
                    ground_normal: Vec2::Y,
                },
            );
            states.insert(ent, ActorState::default());
        }

        // Aim right and press BTN_SECONDARY with slot 3 (Gravel, material_id = 4)
        {
            let mut inputs = session.world.resource_mut::<MatchInputs>();
            inputs.players[0] = PlayerInput {
                move_x: 0,
                move_y: 0,
                buttons: BTN_SECONDARY,
                selected_slot: 3, // Gravel
                aim_dir: [1.0, 0.0],
            };
        }

        session.world.run_system(player_throw_particles_system, ());

        // Verify that gravel particles were thrown into ParticlePool
        {
            let pool = session.world.resource::<ParticlePool>();
            assert!(!pool.is_empty(), "secondary action must throw particles");
            for p in pool.particles() {
                assert_eq!(p.atom.material_id, 4);
                assert_eq!(p.atom.life_5bit(), 0);
                assert_eq!(p.target_actor_id, 0); // flying free
                assert_eq!(p.flags, 0); // normal collision & gravity
            }
        }

        // Simulate particles until they hit the rock wall/floor and reinsert
        for tick in 1..=40 {
            let mut pool = session.world.resource_mut::<ParticlePool>();
            let mut world = session.world.resource_mut::<ChunkWorld>();
            let mut dirty = session.world.resource_mut::<DirtyRects>();
            step_particle_simulation(&mut pool, &mut world, Some(&mut dirty), tick);
            if pool.is_empty() {
                break;
            }
        }

        // Verify gravel settled onto the wall/floor
        let world = session.world.resource::<ChunkWorld>();
        let mut gravel_settled = 0;
        for wx in 0..=30 {
            for wy in -15..=15 {
                let cp = global_to_chunk(world_cell_to_global_atom(wx, wy));
                if let Some(atom) = world.get_atom(&cp)
                    && atom.material_id == 4
                {
                    gravel_settled += 1;
                }
            }
        }
        assert!(
            gravel_settled > 0,
            "thrown particles must settle into terrain upon collision"
        );
    }

    #[test]
    fn test_tool_sprite_creation_and_sync() {
        register_player_schemas();
        astratomic_sim::plugin::register_sim_schemas();
        astratomic_physics::plugin::register_physics_schemas();

        let session = Session::default();
        session.world.init_resource::<MatchInputs>();
        session.world.init_resource::<Time>();

        let player_ent = {
            let mut entities = session.world.resource_mut::<Entities>();
            let mut players = session.world.component_mut::<Player>();
            let mut player_controlled = session.world.component_mut::<PlayerControlled>();
            let mut actors = session.world.component_mut::<KinematicActor>();
            let mut states = session.world.component_mut::<ActorState>();
            let mut transforms = session.world.component_mut::<Transform>();

            let ent = entities.create();
            players.insert(ent, Player);
            player_controlled.insert(ent, PlayerControlled { player_id: 0 });
            actors.insert(
                ent,
                KinematicActor {
                    pos: Vec2::new(10.0, 20.0),
                    prev_pos: Vec2::new(10.0, 20.0),
                    vel: Vec2::ZERO,
                    half_size: PLAYER_HALF_SIZE,
                    step_height: PLAYER_STEP_HEIGHT,
                    weight: 70.0,
                    grounded: true,
                    ground_normal: Vec2::Y,
                },
            );
            states.insert(ent, ActorState::default());
            transforms.insert(
                ent,
                Transform::from_translation(Vec3::new(10.0, 20.0, PLAYER_LAYER)),
            );
            ent
        };

        // Aim right
        {
            let mut inputs = session.world.resource_mut::<MatchInputs>();
            inputs.players[0] = PlayerInput {
                move_x: 0,
                move_y: 0,
                buttons: 0,
                selected_slot: 0,
                aim_dir: [1.0, 0.0],
            };
        }

        session.world.run_system(sync_player_visuals_system, ());

        // Verify tool entity was spawned and positioned facing right
        let tool_ent = {
            let entities = session.world.resource::<Entities>();
            let tools = session.world.component::<PlayerTool>();
            let mut found = None;
            for (t_ent, tool) in entities.iter_with(&tools) {
                if tool.player == player_ent {
                    found = Some(t_ent);
                    break;
                }
            }
            found.expect("sync_player_visuals_system must spawn PlayerTool entity")
        };

        {
            let transforms = session.world.component::<Transform>();
            let tool_trans = transforms.get(tool_ent).expect("tool must have Transform");
            // facing right: offset_x = -1.0 + 5.0 = 4.0, aim_norm = (1.0, 0.0) -> pos.x = 10.0 + 4.0 + 3.0 = 17.0
            assert_eq!(tool_trans.translation.x, 17.0);
            assert!((tool_trans.translation.y - (20.0 - 3.5)).abs() < 1e-4);
            assert_eq!(tool_trans.translation.z, PLAYER_LAYER + 0.1);
        }

        // Aim left
        {
            let mut inputs = session.world.resource_mut::<MatchInputs>();
            inputs.players[0].aim_dir = [-1.0, 0.0];
        }

        session.world.run_system(sync_player_visuals_system, ());

        {
            let transforms = session.world.component::<Transform>();
            let tool_trans = transforms.get(tool_ent).unwrap();
            // facing left: offset_x = 1.0 - 5.0 = -4.0, aim_norm = (-1.0, 0.0) -> pos.x = 10.0 - 4.0 - 3.0 = 3.0
            assert!((tool_trans.translation.x - 3.0).abs() < 1e-4);
        }

        // Destroy player -> orphaned tool must be cleaned up
        {
            let mut entities = session.world.resource_mut::<Entities>();
            entities.kill(player_ent);
        }

        session.world.run_system(sync_player_visuals_system, ());

        {
            let entities = session.world.resource::<Entities>();
            assert!(
                !entities.is_alive(tool_ent),
                "orphaned tool must be despawned when player is destroyed"
            );
        }
    }
}
