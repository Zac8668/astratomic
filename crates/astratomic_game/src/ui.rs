//! Egui world debug panel, player/actor inspector, material brush, prefab stamper, camera controls,
//! and wireframe debug overlays.

use std::collections::HashSet;

use astratomic_physics::prelude::{ActorState, KinematicActor, find_surface_spawn_pos};
use astratomic_render::prelude::ParticleRenderConfig;
use astratomic_sim::{
    atom::{Atom, Materials},
    chunk::CHUNK_SIZE,
    geom_tools::{Line, circle_points, square_points},
    particle::{MAX_PARTICLES, Particle, ParticlePool},
    world::{
        ChunkGeneratorHook, ChunkWorld, DirtyRects, SimulationSettings, global_to_chunk,
        world_to_global_atom,
    },
    world_store::WorldIoWorker,
};
use bones_framework::prelude::*;

use crate::{
    player::{LocalInputState, LocalView, MatchInputs, PlayerControlled},
    prefabs::{
        PendingPrefabStamps, PrefabConfigs, PrefabPlaced, stamp_initial_window_prefabs,
        stamp_prefab_manual,
    },
    worldgen::Generator,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, HasSchema)]
#[repr(u8)]
pub enum DebugTab {
    #[default]
    Tools = 0,
    Particles = 1,
    Simulation = 2,
    Player = 3,
    WorldGen = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, HasSchema)]
#[repr(u8)]
pub enum ParticleSpawnType {
    #[default]
    Fx = 0,
    Matter = 1,
    Explosion = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, HasSchema)]
#[repr(u8)]
pub enum BrushShape {
    #[default]
    Circle = 0,
    Square = 1,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, HasSchema)]
#[repr(u8)]
pub enum DebugToolMode {
    #[default]
    PlayerTool = 0,
    Brush = 1,
    PrefabStamp = 2,
    ParticleEmitter = 3,
}

/// Session resource holding the state of the world debug UI, camera, and brush tools.
#[derive(HasSchema, Clone, Debug)]
#[schema(opaque)]
pub struct DebugToolState {
    pub selected_tab: DebugTab,
    pub tool_mode: DebugToolMode,
    pub selected_material_id: u8,
    pub brush_radius: i32,
    pub brush_shape: BrushShape,
    pub only_air: bool,
    pub selected_prefab_idx: usize,
    pub snap_prefab_to_surface: bool,
    pub overwrite_prefab_terrain: bool,
    pub show_dirty_rects: bool,
    pub show_chunk_grid: bool,
    pub show_brush_cursor: bool,
    pub show_actor_debug: bool,
    pub show_particles_overlay: bool,
    pub camera_height: f32,
    pub camera_speed: f32,
    pub reset_camera_requested: bool,
    pub respawn_player_requested: bool,
    // --- Particle Testing Controls ---
    pub particle_spawn_type: ParticleSpawnType,
    pub particle_fx_color: [u8; 4],
    pub particle_fx_lifetime: u8,
    pub particle_fx_bounces: bool,
    pub particle_speed: f32,
    pub particle_angle_deg: f32,
    pub particle_spread_deg: f32,
    pub particle_burst_radius: i32,
    pub particle_continuous_fountain: bool,
    // --- Draft Settings & Telemetry ---
    /// Draft generator settings edited in the UI, applied on "Regenerate Active Window".
    pub gen_draft: Generator,
    pub gen_params_initialized: bool,
    pub smoothed_fps: f32,
    pub prev_cursor_global: Option<IVec2>,
    pub prev_middle_drag_world: Option<Vec2>,
    pub held_keys: HashSet<KeyCode>,
    pub held_mouse: HashSet<MouseButton>,
    pub just_pressed_mouse: HashSet<MouseButton>,
    pub just_pressed_keys: HashSet<KeyCode>,
    pub dirty_rect_path_entity: Option<Entity>,
    pub chunk_grid_path_entity: Option<Entity>,
    pub brush_cursor_path_entity: Option<Entity>,
    pub actor_debug_path_entity: Option<Entity>,
}

impl Default for DebugToolState {
    fn default() -> Self {
        Self {
            selected_tab: DebugTab::Tools,
            tool_mode: DebugToolMode::PlayerTool,
            selected_material_id: 2, // Sand
            brush_radius: 4,
            brush_shape: BrushShape::Circle,
            only_air: false,
            selected_prefab_idx: 0,
            snap_prefab_to_surface: true,
            overwrite_prefab_terrain: false,
            show_dirty_rects: false,
            show_chunk_grid: false,
            show_brush_cursor: true,
            show_actor_debug: true,
            show_particles_overlay: true,
            camera_height: 240.0,
            camera_speed: 320.0,
            reset_camera_requested: false,
            respawn_player_requested: false,
            particle_spawn_type: ParticleSpawnType::Fx,
            particle_fx_color: [255, 180, 40, 255], // spark orange
            particle_fx_lifetime: 20,
            particle_fx_bounces: true,
            particle_speed: 6.0,
            particle_angle_deg: -90.0, // upward in simulation coordinates (+Y is down)
            particle_spread_deg: 25.0,
            particle_burst_radius: 4,
            particle_continuous_fountain: false,
            gen_draft: Generator::default_config(),
            gen_params_initialized: false,
            smoothed_fps: 60.0,
            prev_cursor_global: None,
            prev_middle_drag_world: None,
            held_keys: HashSet::new(),
            held_mouse: HashSet::new(),
            just_pressed_mouse: HashSet::new(),
            just_pressed_keys: HashSet::new(),
            dirty_rect_path_entity: None,
            chunk_grid_path_entity: None,
            brush_cursor_path_entity: None,
            actor_debug_path_entity: None,
        }
    }
}

pub fn register_debug_ui_schemas() {
    BrushShape::register_schema();
    DebugTab::register_schema();
    ParticleSpawnType::register_schema();
    DebugToolMode::register_schema();
    DebugToolState::register_schema();
}

/// Plugin that installs the camera controller, world brush/stamper, debug overlays, and egui UI.
#[derive(Default, Clone, Copy)]
pub struct AstratomicDebugUiPlugin;

impl SessionPlugin for AstratomicDebugUiPlugin {
    fn install(self, session: &mut Session) {
        register_debug_ui_schemas();
        session.world.init_resource::<DebugToolState>();

        session
            .add_system_to_stage(CoreStage::PreUpdate, track_input_state_system)
            .add_system_to_stage(CoreStage::Update, camera_control_system)
            .add_system_to_stage(CoreStage::Update, world_debug_tool_system)
            .add_system_to_stage(CoreStage::PostUpdate, debug_overlays_system)
            .add_system_to_stage(CoreStage::PostUpdate, debug_ui_system);
    }
}

/// Updates held and just-pressed keyboard/mouse sets from per-frame Bones input events.
pub fn track_input_state_system(
    keyboard: Option<Res<KeyboardInputs>>,
    mouse: Option<Res<MouseInputs>>,
    window: Option<Res<Window>>,
    mut local_input: Option<ResMut<LocalInputState>>,
    mut state: ResMut<DebugToolState>,
) {
    state.just_pressed_keys.clear();
    state.just_pressed_mouse.clear();

    if let Some(ref mut li) = local_input {
        li.mouse_actions_enabled = matches!(state.tool_mode, DebugToolMode::PlayerTool);
    }

    if let Some(window) = window
        && !window.focused
    {
        state.held_keys.clear();
        state.held_mouse.clear();
        state.prev_cursor_global = None;
        state.prev_middle_drag_world = None;
        return;
    }

    if let Some(keyboard) = keyboard {
        for ev in &keyboard.key_events {
            if let Set(code) = ev.key_code {
                if ev.button_state.pressed() {
                    if state.held_keys.insert(code) {
                        state.just_pressed_keys.insert(code);
                    }
                } else {
                    state.held_keys.remove(&code);
                }
            }
        }
    }

    if let Some(mouse) = mouse {
        for ev in &mouse.button_events {
            if ev.state.pressed() {
                if state.held_mouse.insert(ev.button) {
                    state.just_pressed_mouse.insert(ev.button);
                }
            } else {
                state.held_mouse.remove(&ev.button);
            }
        }
    }
}

/// Controls camera zoom (scroll wheel), free-fly camera pan when detached from player,
/// and player respawning at the terrain surface.
pub fn camera_control_system(
    time: Res<Time>,
    mouse: Option<Res<MouseInputs>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    egui_ctx: Option<Res<EguiCtx>>,
    chunk_world: Res<ChunkWorld>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    mut actors: CompMut<KinematicActor>,
    mut actor_states: CompMut<ActorState>,
    mut local_view: ResMut<LocalView>,
    mut local_input: Option<ResMut<LocalInputState>>,
    mut cameras: CompMut<Camera>,
    mut transforms: CompMut<Transform>,
    mut sim_settings: ResMut<SimulationSettings>,
    mut state: ResMut<DebugToolState>,
) {
    let dt = time.delta_seconds().clamp(0.0001, 0.1);
    state.smoothed_fps = state.smoothed_fps * 0.92 + (1.0 / dt) * 0.08;

    let egui_wants_kb = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_keyboard_input());
    let egui_wants_pointer = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_pointer_input() || ctx.is_pointer_over_area());

    // Quick hotkeys when egui is not capturing keyboard focus.
    if !egui_wants_kb {
        if state.just_pressed_keys.contains(&KeyCode::P) {
            sim_settings.paused = !sim_settings.paused;
        }
        let hotkey_materials = [
            (KeyCode::Key0, 0_u8),
            (KeyCode::Key1, 2_u8),
            (KeyCode::Key2, 3_u8),
            (KeyCode::Key3, 4_u8),
            (KeyCode::Key4, 5_u8),
            (KeyCode::Key5, 6_u8),
            (KeyCode::Key6, 7_u8),
            (KeyCode::Key7, 8_u8),
            (KeyCode::Key8, 9_u8),
            (KeyCode::Key9, 10_u8),
        ];
        for (key, mat_id) in hotkey_materials {
            if state.just_pressed_keys.contains(&key) {
                state.selected_material_id = mat_id;
            }
        }
    }

    // Handle respawn player request
    if state.respawn_player_requested {
        state.respawn_player_requested = false;
        for (ent, ctrl) in entities.iter_with(&player_controlled) {
            if ctrl.player_id == local_view.local_player_id
                && let Some(actor) = actors.get_mut(ent)
            {
                let spawn_pos = find_surface_spawn_pos(&chunk_world, actor.pos.x, actor.half_size);
                actor.pos = spawn_pos;
                actor.prev_pos = spawn_pos;
                actor.vel = Vec2::ZERO;
                actor.grounded = true;
                if let Some(st) = actor_states.get_mut(ent) {
                    *st = ActorState::default();
                }
                local_view.camera_pos = spawn_pos;
            }
        }
    }

    // Zoom with scroll wheel when pointer is not over egui.
    if !egui_wants_pointer && let Some(mouse) = mouse {
        for wheel in &mouse.wheel_events {
            if wheel.movement.y.abs() > f32::EPSILON {
                let zoom_factor = if wheel.movement.y > 0.0 { 0.88 } else { 1.14 };
                state.camera_height = (state.camera_height * zoom_factor).clamp(64.0, 1800.0);
            }
        }
    }

    let follow_player = local_input
        .as_ref()
        .is_none_or(|li| li.camera_follow_player);

    let mut pan_dir = Vec2::ZERO;
    if !follow_player && !egui_wants_kb {
        if state.held_keys.contains(&KeyCode::A) || state.held_keys.contains(&KeyCode::Left) {
            pan_dir.x -= 1.0;
        }
        if state.held_keys.contains(&KeyCode::D) || state.held_keys.contains(&KeyCode::Right) {
            pan_dir.x += 1.0;
        }
        if state.held_keys.contains(&KeyCode::S) || state.held_keys.contains(&KeyCode::Down) {
            pan_dir.y -= 1.0;
        }
        if state.held_keys.contains(&KeyCode::W) || state.held_keys.contains(&KeyCode::Up) {
            pan_dir.y += 1.0;
        }
    }

    let boost = if state.held_keys.contains(&KeyCode::ShiftLeft)
        || state.held_keys.contains(&KeyCode::ShiftRight)
    {
        2.5
    } else {
        1.0
    };
    let zoom_scale = (state.camera_height / 360.0).clamp(0.35, 4.0);
    let velocity = pan_dir.normalize_or_zero() * state.camera_speed * boost * zoom_scale;

    let current_mouse_world = mouse_world.and_then(|mw| mw.0);

    for (_ent, (camera, transform)) in entities.iter_with((&mut cameras, &mut transforms)) {
        if state.reset_camera_requested {
            transform.translation.x = local_view.camera_pos.x;
            transform.translation.y = local_view.camera_pos.y;
            if let Some(ref mut li) = local_input {
                li.camera_follow_player = true;
            }
            state.reset_camera_requested = false;
        }
        camera.size = CameraSize::FixedHeight(state.camera_height);

        if !follow_player {
            transform.translation.x += velocity.x * dt;
            transform.translation.y += velocity.y * dt;
        }

        // Middle-mouse drag camera panning detaches camera follow automatically
        if !egui_wants_pointer && state.held_mouse.contains(&MouseButton::Middle) {
            if let Some(ref mut li) = local_input {
                li.camera_follow_player = false;
            }
            if let Some(curr_world) = current_mouse_world {
                if let Some(prev_world) = state.prev_middle_drag_world {
                    let delta = prev_world - curr_world;
                    transform.translation.x += delta.x;
                    transform.translation.y += delta.y;
                } else {
                    state.prev_middle_drag_world = Some(curr_world);
                }
            }
        } else {
            state.prev_middle_drag_world = None;
        }

        if !follow_player {
            sim_settings.focus_world_pos = transform.translation.truncate();
        }
    }
}

/// Applies brush painting/erasing, prefab stamping, or particle emission at the cursor position.
pub fn world_debug_tool_system(
    egui_ctx: Option<Res<EguiCtx>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    materials: Res<Materials>,
    prefab_configs: Res<PrefabConfigs>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
    mut state: ResMut<DebugToolState>,
) {
    let egui_wants_pointer = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_pointer_input() || ctx.is_pointer_over_area());

    let Some(cursor_world) = mouse_world.and_then(|mw| mw.0) else {
        state.prev_cursor_global = None;
        return;
    };

    if egui_wants_pointer {
        state.prev_cursor_global = None;
        return;
    }

    let cursor_global = world_to_global_atom(cursor_world);

    let left_down = state.held_mouse.contains(&MouseButton::Left);
    let right_down = state.held_mouse.contains(&MouseButton::Right);
    let left_just_pressed = state.just_pressed_mouse.contains(&MouseButton::Left);
    let right_just_pressed = state.just_pressed_mouse.contains(&MouseButton::Right);

    match state.tool_mode {
        DebugToolMode::PlayerTool => {
            // Handled deterministically in `player_world_action_system` via `MatchInputs`.
            state.prev_cursor_global = None;
        }
        DebugToolMode::Brush => {
            if left_down || right_down {
                let mat_id = if right_down {
                    0 // Right-click always erases to Void
                } else {
                    state.selected_material_id
                };
                let only_air = if right_down { false } else { state.only_air };
                let start = state.prev_cursor_global.unwrap_or(cursor_global);
                let radius = state.brush_radius.clamp(0, 32);
                let shape = state.brush_shape;

                for center in Line::points_inclusive(start, cursor_global) {
                    let offsets = match shape {
                        BrushShape::Circle => circle_points(center, radius),
                        BrushShape::Square => square_points(center, radius),
                    };
                    for pt in offsets {
                        chunk_world.paint_at_global(
                            pt,
                            mat_id,
                            only_air,
                            &materials,
                            &mut dirty_rects,
                        );
                    }
                }
                state.prev_cursor_global = Some(cursor_global);
            } else {
                state.prev_cursor_global = None;
            }
        }
        DebugToolMode::PrefabStamp => {
            if left_just_pressed
                && let Some(config) = prefab_configs.0.get(state.selected_prefab_idx)
            {
                stamp_prefab_manual(
                    &mut chunk_world,
                    &mut dirty_rects,
                    &materials,
                    config,
                    cursor_global,
                    state.snap_prefab_to_surface,
                    state.overwrite_prefab_terrain,
                );
            }
            if right_down {
                let start = state.prev_cursor_global.unwrap_or(cursor_global);
                let radius = state.brush_radius.clamp(0, 32);
                for center in Line::points_inclusive(start, cursor_global) {
                    for pt in circle_points(center, radius) {
                        chunk_world.paint_at_global(pt, 0, false, &materials, &mut dirty_rects);
                    }
                }
                state.prev_cursor_global = Some(cursor_global);
            } else {
                state.prev_cursor_global = None;
            }
        }
        DebugToolMode::ParticleEmitter => {
            state.prev_cursor_global = None;
            let Some(ref mut pool) = particle_pool else {
                return;
            };

            let should_emit = left_just_pressed || (state.particle_continuous_fountain && left_down);
            if should_emit {
                match state.particle_spawn_type {
                    ParticleSpawnType::Fx => {
                        let spread = (fastrand::f32() * 2.0 - 1.0) * state.particle_spread_deg;
                        let rad = (state.particle_angle_deg + spread).to_radians();
                        let speed = state.particle_speed * (0.85 + fastrand::f32() * 0.3);
                        let vel = [rad.cos() * speed, rad.sin() * speed];
                        let spawn_pos = [cursor_global.x as f32 + 0.5, cursor_global.y as f32 + 0.5];
                        pool.spawn_fx(
                            spawn_pos,
                            vel,
                            state.particle_fx_color,
                            state.particle_fx_lifetime,
                            state.particle_fx_bounces,
                        );
                    }
                    ParticleSpawnType::Matter => {
                        let spread = (fastrand::f32() * 2.0 - 1.0) * state.particle_spread_deg;
                        let rad = (state.particle_angle_deg + spread).to_radians();
                        let speed = state.particle_speed * (0.85 + fastrand::f32() * 0.3);
                        let vel = [rad.cos() * speed, rad.sin() * speed];
                        if !pool.eject_from_grid_dirty(
                            &mut chunk_world,
                            &mut dirty_rects,
                            cursor_global.x,
                            cursor_global.y,
                            vel,
                        ) {
                            let mut atom = Atom::new_with_rng(
                                state.selected_material_id,
                                &materials,
                                &mut fastrand::Rng::new(),
                            );
                            atom.set_free_falling(true);
                            atom.set_life_5bit(0);
                            atom.vx = 0;
                            atom.vy = 0;
                            let spawn_pos = [cursor_global.x as f32 + 0.5, cursor_global.y as f32 + 0.5];
                            pool.push_particle(Particle {
                                pos: spawn_pos,
                                vel,
                                atom,
                            });
                        }
                    }
                    ParticleSpawnType::Explosion => {
                        if left_just_pressed {
                            let r = state.particle_burst_radius.clamp(1, 16);
                            for pt in circle_points(cursor_global, r) {
                                let dx = pt.x as f32 - cursor_global.x as f32;
                                let dy = pt.y as f32 - cursor_global.y as f32;
                                let dist = (dx * dx + dy * dy).sqrt().max(0.2);
                                let speed = (state.particle_speed * 0.5)
                                    + fastrand::f32() * (state.particle_speed * 0.8);
                                let vel = [dx / dist * speed, dy / dist * speed];
                                pool.eject_from_grid_dirty(
                                    &mut chunk_world,
                                    &mut dirty_rects,
                                    pt.x,
                                    pt.y,
                                    vel,
                                );
                            }
                            // Spawn explosive sparks
                            for _ in 0..12 {
                                let angle = fastrand::f32() * std::f32::consts::TAU;
                                let speed = fastrand::f32() * state.particle_speed;
                                let vel = [angle.cos() * speed, angle.sin() * speed];
                                let spark_pos = [cursor_global.x as f32 + 0.5, cursor_global.y as f32 + 0.5];
                                let color = if fastrand::bool() {
                                    [255, 200, 50, 255]
                                } else {
                                    [255, 100, 20, 255]
                                };
                                pool.spawn_fx(spark_pos, vel, color, fastrand::u8(10..25), true);
                            }
                        }
                    }
                }
            }

            if right_just_pressed {
                // Right click in particle emitter mode triggers radial explosion blast
                let r = state.particle_burst_radius.clamp(1, 16);
                for pt in circle_points(cursor_global, r) {
                    let dx = pt.x as f32 - cursor_global.x as f32;
                    let dy = pt.y as f32 - cursor_global.y as f32;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.2);
                    let speed = (state.particle_speed * 0.5)
                        + fastrand::f32() * (state.particle_speed * 0.8);
                    let vel = [dx / dist * speed, dy / dist * speed];
                    pool.eject_from_grid_dirty(&mut chunk_world, &mut dirty_rects, pt.x, pt.y, vel);
                }
            }
        }
    }
}

fn append_rect_path(points: &mut Vec<Vec2>, line_breaks: &mut Vec<usize>, tl: Vec2, br: Vec2) {
    let tr = Vec2::new(br.x, tl.y);
    let bl = Vec2::new(tl.x, br.y);
    points.extend_from_slice(&[tl, tr, br, bl, tl]);
    line_breaks.push(points.len());
}

fn ensure_overlay_entity(
    slot: &mut Option<Entity>,
    entities: &mut Entities,
    transforms: &mut CompMut<Transform>,
    paths: &mut CompMut<Path2d>,
    z: f32,
) -> Entity {
    if let Some(ent) = *slot
        && entities.is_alive(ent)
    {
        return ent;
    }
    let ent = entities.create();
    transforms.insert(ent, Transform::from_translation(Vec3::new(0.0, 0.0, z)));
    paths.insert(ent, Path2d::default());
    *slot = Some(ent);
    ent
}

/// Renders wireframe debug overlays (dirty rects, chunk grid borders, brush cursor, and
/// `KinematicActor` hitbox / step-height / aim / ground normal) using `Path2d`.
pub fn debug_overlays_system(
    mouse_world: Option<Res<MouseWorldPosition>>,
    egui_ctx: Option<Res<EguiCtx>>,
    chunk_world: Res<ChunkWorld>,
    dirty_rects: Res<DirtyRects>,
    match_inputs: Res<MatchInputs>,
    local_view: Res<LocalView>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut paths: CompMut<Path2d>,
    mut state: ResMut<DebugToolState>,
) {
    // 1. Update Dirty Rects overlay
    let dirty_ent = ensure_overlay_entity(
        &mut state.dirty_rect_path_entity,
        &mut entities,
        &mut transforms,
        &mut paths,
        10.0,
    );
    if let Some(path) = paths.get_mut(dirty_ent) {
        path.points.clear();
        path.line_breaks.clear();
        path.color = Color::rgba(0.2, 1.0, 0.35, 0.85);
        path.thickness = 1.0;

        if state.show_dirty_rects {
            for (chunk_pos, rect) in &dirty_rects.current {
                let min_x = chunk_pos.x as f32 * CHUNK_SIZE as f32 + rect.min.x as f32;
                let max_x = chunk_pos.x as f32 * CHUNK_SIZE as f32 + rect.max.x as f32 + 1.0;
                let top_y = -(chunk_pos.y as f32 * CHUNK_SIZE as f32) - rect.min.y as f32;
                let bot_y = -(chunk_pos.y as f32 * CHUNK_SIZE as f32) - rect.max.y as f32 - 1.0;
                append_rect_path(
                    &mut path.points,
                    &mut path.line_breaks,
                    Vec2::new(min_x, top_y),
                    Vec2::new(max_x, bot_y),
                );
            }
        }
    }

    // 2. Chunk Grid Borders overlay
    let grid_ent = ensure_overlay_entity(
        &mut state.chunk_grid_path_entity,
        &mut entities,
        &mut transforms,
        &mut paths,
        8.0,
    );
    if let Some(path) = paths.get_mut(grid_ent) {
        path.points.clear();
        path.line_breaks.clear();
        path.color = Color::rgba(0.35, 0.75, 1.0, 0.35);
        path.thickness = 1.0;

        if state.show_chunk_grid {
            let cs = CHUNK_SIZE as f32;
            for &chunk_pos in chunk_world.chunks.keys() {
                let min_x = chunk_pos.x as f32 * cs;
                let max_x = min_x + cs;
                let top_y = -(chunk_pos.y as f32 * cs);
                let bot_y = top_y - cs;
                append_rect_path(
                    &mut path.points,
                    &mut path.line_breaks,
                    Vec2::new(min_x, top_y),
                    Vec2::new(max_x, bot_y),
                );
            }
        }
    }

    // 3. Brush / Tool Cursor preview overlay
    let cursor_ent = ensure_overlay_entity(
        &mut state.brush_cursor_path_entity,
        &mut entities,
        &mut transforms,
        &mut paths,
        15.0,
    );
    let egui_wants_pointer = egui_ctx
        .as_ref()
        .is_some_and(|ctx| ctx.wants_pointer_input() || ctx.is_pointer_over_area());
    if let Some(path) = paths.get_mut(cursor_ent) {
        path.points.clear();
        path.line_breaks.clear();
        let preview_rgba = match state.tool_mode {
            DebugToolMode::ParticleEmitter => match state.particle_spawn_type {
                ParticleSpawnType::Fx => state.particle_fx_color,
                ParticleSpawnType::Matter => {
                    Atom::material_preview_color(state.selected_material_id)
                }
                ParticleSpawnType::Explosion => [255, 140, 20, 255],
            },
            DebugToolMode::PrefabStamp => [180, 220, 255, 255],
            _ => Atom::material_preview_color(state.selected_material_id),
        };
        path.color = Color::rgba(
            preview_rgba[0] as f32 / 255.0,
            preview_rgba[1] as f32 / 255.0,
            preview_rgba[2] as f32 / 255.0,
            (preview_rgba[3] as f32 / 255.0).max(0.6),
        );
        path.thickness = 1.0;

        if state.show_brush_cursor
            && !egui_wants_pointer
            && !matches!(state.tool_mode, DebugToolMode::PlayerTool)
            && let Some(cursor_world) = mouse_world.and_then(|mw| mw.0)
        {
            let global = world_to_global_atom(cursor_world);
            let r = state.brush_radius.clamp(0, 32) as f32;
            let cx = global.x as f32 + 0.5;
            let cy = -(global.y as f32) - 0.5;

            match state.tool_mode {
                DebugToolMode::Brush => match state.brush_shape {
                    BrushShape::Circle if r >= 1.5 => {
                        const SEGMENTS: usize = 24;
                        for i in 0..=SEGMENTS {
                            let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
                            path.points.push(Vec2::new(
                                cx + angle.cos() * (r + 0.5),
                                cy + angle.sin() * (r + 0.5),
                            ));
                        }
                        path.line_breaks.push(path.points.len());
                    }
                    _ => {
                        let half = r + 0.5;
                        append_rect_path(
                            &mut path.points,
                            &mut path.line_breaks,
                            Vec2::new(cx - half, cy + half),
                            Vec2::new(cx + half, cy - half),
                        );
                    }
                },
                DebugToolMode::PrefabStamp => {
                    let half = r + 0.5;
                    append_rect_path(
                        &mut path.points,
                        &mut path.line_breaks,
                        Vec2::new(cx - half, cy + half),
                        Vec2::new(cx + half, cy - half),
                    );
                }
                DebugToolMode::ParticleEmitter => match state.particle_spawn_type {
                    ParticleSpawnType::Explosion => {
                        let br = state.particle_burst_radius.clamp(1, 16) as f32;
                        const SEGMENTS: usize = 20;
                        for i in 0..=SEGMENTS {
                            let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
                            path.points.push(Vec2::new(
                                cx + angle.cos() * (br + 0.5),
                                cy + angle.sin() * (br + 0.5),
                            ));
                        }
                        path.line_breaks.push(path.points.len());
                    }
                    ParticleSpawnType::Fx | ParticleSpawnType::Matter => {
                        let rad = state.particle_angle_deg.to_radians();
                        let dir = Vec2::new(rad.cos(), -rad.sin());
                        path.points.push(Vec2::new(cx, cy));
                        path.points.push(Vec2::new(cx, cy) + dir * 18.0);
                        path.line_breaks.push(path.points.len());
                    }
                },
                DebugToolMode::PlayerTool => {}
            }
        }
    }

    // 4. KinematicActor hitbox, step-height line, ground normal, and aim direction overlay
    let actor_ent = ensure_overlay_entity(
        &mut state.actor_debug_path_entity,
        &mut entities,
        &mut transforms,
        &mut paths,
        12.0,
    );
    if let Some(path) = paths.get_mut(actor_ent) {
        path.points.clear();
        path.line_breaks.clear();
        path.color = Color::rgba(0.25, 0.95, 1.0, 0.9);
        path.thickness = 1.0;

        if state.show_actor_debug {
            let alpha = local_view.render_alpha.clamp(0.0, 1.0);
            for (ent, actor) in entities.iter_with(&actors) {
                let p = actor.prev_pos.lerp(actor.pos, alpha);
                let tl = Vec2::new(p.x - actor.half_size.x, p.y + actor.half_size.y);
                let br = Vec2::new(p.x + actor.half_size.x, p.y - actor.half_size.y);
                append_rect_path(&mut path.points, &mut path.line_breaks, tl, br);

                // Foot step-height line
                let step_y = (p.y - actor.half_size.y) + actor.step_height as f32;
                path.points.push(Vec2::new(tl.x, step_y));
                path.points.push(Vec2::new(br.x, step_y));
                path.line_breaks.push(path.points.len());

                // Ground normal vector from feet center
                let foot_center = Vec2::new(p.x, p.y - actor.half_size.y);
                path.points.push(foot_center);
                path.points.push(foot_center + actor.ground_normal * 6.0);
                path.line_breaks.push(path.points.len());

                // Aim vector if player-controlled
                if let Some(ctrl) = player_controlled.get(ent)
                    && let Some(input) = match_inputs.players.get(ctrl.player_id as usize)
                {
                    let aim = input.aim_vec2();
                    if aim != Vec2::ZERO {
                        path.points.push(p);
                        path.points.push(p + aim * 12.0);
                        path.line_breaks.push(path.points.len());
                    }
                }
            }
        }
    }

    // 5. Active particles are rendered on the GPU by `astratomic_render::particles::render_particles_system`.
}

/// Renders the interactive `egui` World & Player Debug Window with a tabbed layout.
pub fn debug_ui_system(
    egui_ctx: Option<Res<EguiCtx>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    materials: Res<Materials>,
    entities: Res<Entities>,
    player_controlled: Comp<PlayerControlled>,
    actors: Comp<KinematicActor>,
    actor_states: Comp<ActorState>,
    match_inputs: Res<MatchInputs>,
    local_view: Res<LocalView>,
    mut local_input: Option<ResMut<LocalInputState>>,
    mut generator: ResMut<Generator>,
    mut generator_hook: Option<ResMut<ChunkGeneratorHook>>,
    mut sim_settings: ResMut<SimulationSettings>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut particle_pool: Option<ResMut<ParticlePool>>,
    mut particle_render_config: Option<ResMut<ParticleRenderConfig>>,
    mut prefab_configs: ResMut<PrefabConfigs>,
    mut prefab_placed: ResMut<PrefabPlaced>,
    mut pending_stamps: ResMut<PendingPrefabStamps>,
    io_worker: Res<WorldIoWorker>,
    mut state: ResMut<DebugToolState>,
) {
    let Some(ctx) = egui_ctx else {
        return;
    };

    if let Some(ref mut prc) = particle_render_config {
        prc.enabled = state.show_particles_overlay;
    }

    if !state.gen_params_initialized && generator.scale > 0.0 {
        state.gen_draft = *generator;
        state.gen_params_initialized = true;
    }

    let cursor_world = mouse_world.and_then(|mw| mw.0);
    let cursor_global = cursor_world.map(world_to_global_atom);
    let cursor_chunk_pos = cursor_global.map(global_to_chunk);
    let hovered_atom = cursor_chunk_pos
        .as_ref()
        .and_then(|cp| chunk_world.get_atom(cp))
        .copied();

    egui::Window::new("Astratomic World Debug")
        .default_pos(egui::pos2(12.0, 12.0))
        .default_width(340.0)
        .resizable(true)
        .show(&ctx, |ui| {
            // --- Persistent Telemetry Header Banner ---
            ui.horizontal(|ui| {
                let frame_ms = 1000.0 / state.smoothed_fps.max(1.0);
                ui.label(
                    egui::RichText::new(format!(
                        "{:.0} FPS ({:.1} ms)",
                        state.smoothed_fps, frame_ms
                    ))
                    .strong(),
                );
                ui.separator();
                ui.label(format!("Chunks: {}", chunk_world.chunks.len()));
                if let Some(ref pool) = particle_pool {
                    ui.separator();
                    ui.label(format!("Parts: {}", pool.len()));
                }
                ui.separator();
                let pause_label = if sim_settings.paused {
                    "▶ Resume (P)"
                } else {
                    "⏸ Pause (P)"
                };
                if ui.small_button(pause_label).clicked() {
                    sim_settings.paused = !sim_settings.paused;
                }
                if sim_settings.paused && ui.small_button("⏭").clicked() {
                    sim_settings.step_once = true;
                }
            });

            ui.add_space(2.0);

            // --- Tab Selection Bar ---
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut state.selected_tab, DebugTab::Tools, "🛠 Tools");
                ui.selectable_value(&mut state.selected_tab, DebugTab::Particles, "✨ Particles");
                ui.selectable_value(&mut state.selected_tab, DebugTab::Simulation, "⏱ Sim & Cam");
                ui.selectable_value(&mut state.selected_tab, DebugTab::Player, "🏃 Player");
                ui.selectable_value(&mut state.selected_tab, DebugTab::WorldGen, "🌍 World Gen");
            });

            ui.separator();

            // --- Tab Body ---
            match state.selected_tab {
                DebugTab::Tools => {
                    ui.heading("World Tool");
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(
                            &mut state.tool_mode,
                            DebugToolMode::PlayerTool,
                            "⛏ Player Tool",
                        );
                        ui.selectable_value(
                            &mut state.tool_mode,
                            DebugToolMode::Brush,
                            "🖌 Material Brush",
                        );
                        ui.selectable_value(
                            &mut state.tool_mode,
                            DebugToolMode::PrefabStamp,
                            "🌲 Stamp Prefab",
                        );
                        ui.selectable_value(
                            &mut state.tool_mode,
                            DebugToolMode::ParticleEmitter,
                            "✨ Particle Tool",
                        );
                    });

                    match state.tool_mode {
                        DebugToolMode::PlayerTool => {
                            ui.small("Left-Click: Excavate along Aim | Right-Click: Emit Material (Keys 0-9)");
                        }
                        DebugToolMode::Brush => {
                            ui.small("Left-Click: Paint Material | Right-Click: Erase (Void) | Keys 0-9");
                            ui.horizontal(|ui| {
                                ui.label("Radius:");
                                ui.add(egui::Slider::new(&mut state.brush_radius, 0..=32).suffix(" px"));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Shape:");
                                ui.selectable_value(&mut state.brush_shape, BrushShape::Circle, "Circle");
                                ui.selectable_value(&mut state.brush_shape, BrushShape::Square, "Square");
                                ui.checkbox(&mut state.only_air, "Only Fill Air");
                            });

                            ui.label("Materials:");
                            egui::Grid::new("material_palette_grid")
                                .num_columns(3)
                                .spacing([6.0, 4.0])
                                .show(ui, |ui| {
                                    for (idx, &mat_id) in Atom::PAINTABLE_MATERIAL_IDS.iter().enumerate() {
                                        let [r, g, b, _] = Atom::material_preview_color(mat_id);
                                        let selected = state.selected_material_id == mat_id;
                                        let label = Atom::material_name(mat_id);

                                        ui.horizontal(|ui| {
                                            let (swatch_rect, _) = ui.allocate_exact_size(
                                                egui::vec2(12.0, 12.0),
                                                egui::Sense::hover(),
                                            );
                                            ui.painter().rect_filled(
                                                swatch_rect,
                                                2.0,
                                                egui::Color32::from_rgb(r, g, b),
                                            );
                                            if ui.selectable_label(selected, label).clicked() {
                                                state.selected_material_id = mat_id;
                                            }
                                        });

                                        if (idx + 1) % 3 == 0 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        }
                        DebugToolMode::PrefabStamp => {
                            ui.small("Left-Click: Stamp Selected Prefab | Right-Click: Erase");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut state.snap_prefab_to_surface, "Snap to Surface");
                                ui.checkbox(&mut state.overwrite_prefab_terrain, "Overwrite Terrain");
                            });

                            if prefab_configs.0.is_empty() {
                                ui.label("No prefabs found in assets/prefabs/");
                            } else {
                                ui.horizontal_wrapped(|ui| {
                                    for (idx, config) in prefab_configs.0.iter().enumerate() {
                                        let label = format!(
                                            "{} ({}x{})",
                                            config.display_name,
                                            config.image.width(),
                                            config.image.height()
                                        );
                                        if ui
                                            .selectable_label(state.selected_prefab_idx == idx, label)
                                            .clicked()
                                        {
                                            state.selected_prefab_idx = idx;
                                        }
                                    }
                                });
                            }
                        }
                        DebugToolMode::ParticleEmitter => {
                            ui.small("Left-Click: Emit Particles | Right-Click: Explosion Burst");
                            ui.horizontal(|ui| {
                                if ui.button("Open Full Particle Settings ➡").clicked() {
                                    state.selected_tab = DebugTab::Particles;
                                }
                            });
                        }
                    }

                    ui.separator();
                    ui.heading("Cursor Inspector");
                    if let (Some(global), Some(cp)) = (cursor_global, cursor_chunk_pos) {
                        ui.label(format!(
                            "Atom: ({}, {}) | Chunk: ({}, {}) [{}x{}]",
                            global.x, global.y, cp.chunk.x, cp.chunk.y, cp.atom.x, cp.atom.y
                        ));
                        if let Some(atom) = hovered_atom {
                            ui.horizontal(|ui| {
                                let [r, g, b, a] = if atom.is_void() {
                                    [30, 32, 42, 255]
                                } else {
                                    atom.color
                                };
                                let (rect, _) =
                                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                                ui.painter().rect_filled(
                                    rect,
                                    2.0,
                                    egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                                );
                                ui.label(format!(
                                    "#{} {} ({:?}) vel=({},{}) moving={}",
                                    atom.material_id,
                                    Atom::material_name(atom.material_id),
                                    atom.state(),
                                    atom.vx,
                                    atom.vy,
                                    atom.moving()
                                ));
                            });
                        } else {
                            ui.label("Atom: <unloaded chunk>");
                        }
                    } else {
                        ui.label("Cursor: <outside window>");
                    }
                }

                DebugTab::Particles => {
                    ui.heading("Particle Pool Status");
                    let active_count = particle_pool.as_ref().map(|p| p.len()).unwrap_or(0);
                    let matter_count = particle_pool
                        .as_ref()
                        .map(|p| p.particles().iter().filter(|pt| pt.atom.life_5bit() == 0).count())
                        .unwrap_or(0);
                    let fx_count = active_count.saturating_sub(matter_count);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Active: {} / {}",
                                active_count, MAX_PARTICLES
                            ))
                            .strong(),
                        );
                        ui.separator();
                        ui.label(format!("Matter: {}", matter_count));
                        ui.separator();
                        ui.label(format!("FX: {}", fx_count));
                    });

                    let fill_ratio = active_count as f32 / MAX_PARTICLES as f32;
                    ui.add(
                        egui::ProgressBar::new(fill_ratio)
                            .text(format!("{:.1}% Capacity", fill_ratio * 100.0)),
                    );

                    ui.horizontal(|ui| {
                        if ui.button("🧹 Clear All Particles").clicked() {
                            if let Some(ref mut pool) = particle_pool {
                                pool.clear();
                            }
                        }
                        ui.checkbox(&mut state.show_particles_overlay, "Show Particles Overlay");
                    });

                    ui.separator();
                    ui.heading("Particle Emitter Tool");
                    ui.horizontal(|ui| {
                        let is_active = matches!(state.tool_mode, DebugToolMode::ParticleEmitter);
                        let label = if is_active {
                            "✅ Emitter Active (Click World)"
                        } else {
                            "🎯 Activate World Click Emitter"
                        };
                        if ui.selectable_label(is_active, label).clicked() {
                            state.tool_mode = if is_active {
                                DebugToolMode::PlayerTool
                            } else {
                                DebugToolMode::ParticleEmitter
                            };
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Mode:");
                        ui.selectable_value(&mut state.particle_spawn_type, ParticleSpawnType::Fx, "✨ Ephemeral FX");
                        ui.selectable_value(&mut state.particle_spawn_type, ParticleSpawnType::Matter, "⛏ Real Matter");
                        ui.selectable_value(&mut state.particle_spawn_type, ParticleSpawnType::Explosion, "💣 Explosion");
                    });

                    ui.horizontal(|ui| {
                        ui.label("Speed:");
                        ui.add(egui::Slider::new(&mut state.particle_speed, 0.5..=24.0).suffix(" px/t"));
                        ui.label("Spread:");
                        ui.add(egui::Slider::new(&mut state.particle_spread_deg, 0.0..=90.0).suffix("°"));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Angle:");
                        ui.add(egui::Slider::new(&mut state.particle_angle_deg, -180.0..=180.0).suffix("°"));
                    });

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Quick Aim:");
                        if ui.small_button("⬆ Up").clicked() { state.particle_angle_deg = -90.0; }
                        if ui.small_button("↗ Up-Right").clicked() { state.particle_angle_deg = -45.0; }
                        if ui.small_button("➡ Right").clicked() { state.particle_angle_deg = 0.0; }
                        if ui.small_button("↘ Down-Right").clicked() { state.particle_angle_deg = 45.0; }
                        if ui.small_button("⬇ Down").clicked() { state.particle_angle_deg = 90.0; }
                        if ui.small_button("⬅ Left").clicked() { state.particle_angle_deg = 180.0; }
                    });

                    ui.checkbox(
                        &mut state.particle_continuous_fountain,
                        "⛲ Continuous stream while holding Left-Click",
                    );

                    match state.particle_spawn_type {
                        ParticleSpawnType::Fx => {
                            ui.separator();
                            ui.label("Ephemeral FX Settings:");
                            ui.horizontal(|ui| {
                                ui.label("Lifetime:");
                                ui.add(egui::Slider::new(&mut state.particle_fx_lifetime, 1..=31).suffix(" ticks"));
                                ui.checkbox(&mut state.particle_fx_bounces, "Bouncing (FLAG_PARTICLE_BOUNCE)");
                            });

                            ui.horizontal_wrapped(|ui| {
                                ui.label("Presets:");
                                if ui.small_button("🔥 Spark").clicked() {
                                    state.particle_fx_color = [255, 180, 40, 255];
                                    state.particle_fx_bounces = true;
                                }
                                if ui.small_button("💧 Foam").clicked() {
                                    state.particle_fx_color = [80, 200, 255, 230];
                                    state.particle_fx_bounces = false;
                                }
                                if ui.small_button("💨 Smoke").clicked() {
                                    state.particle_fx_color = [180, 180, 190, 200];
                                    state.particle_fx_bounces = false;
                                }
                                if ui.small_button("🟢 Acid").clicked() {
                                    state.particle_fx_color = [50, 240, 60, 220];
                                    state.particle_fx_bounces = true;
                                }
                                if ui.small_button("🟣 Plasma").clicked() {
                                    state.particle_fx_color = [220, 80, 255, 255];
                                    state.particle_fx_bounces = true;
                                }
                            });

                            let mut c = egui::Color32::from_rgba_unmultiplied(
                                state.particle_fx_color[0],
                                state.particle_fx_color[1],
                                state.particle_fx_color[2],
                                state.particle_fx_color[3],
                            );
                            ui.horizontal(|ui| {
                                ui.label("FX Color:");
                                let (swatch_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(14.0, 14.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(swatch_rect, 2.0, c);
                                if egui::color_picker::color_edit_button_srgba(
                                    ui,
                                    &mut c,
                                    egui::color_picker::Alpha::Opaque,
                                )
                                .changed()
                                {
                                    state.particle_fx_color = [c.r(), c.g(), c.b(), c.a()];
                                }
                            });
                        }
                        ParticleSpawnType::Matter => {
                            ui.separator();
                            ui.small("Click on solid/powder/liquid cells in the world to eject them, or click air to emit the selected material below:");
                            ui.horizontal(|ui| {
                                ui.label("Material in Air:");
                                let [mr, mg, mb, _] =
                                    Atom::material_preview_color(state.selected_material_id);
                                let (swatch_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(14.0, 14.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(
                                    swatch_rect,
                                    2.0,
                                    egui::Color32::from_rgb(mr, mg, mb),
                                );
                                let curr_name = Atom::material_name(state.selected_material_id);
                                egui::ComboBox::from_id_salt("particle_matter_select")
                                    .selected_text(curr_name)
                                    .show_ui(ui, |ui| {
                                        for &id in &Atom::PAINTABLE_MATERIAL_IDS {
                                            if id != 0 {
                                                let [r, g, b, _] = Atom::material_preview_color(id);
                                                ui.horizontal(|ui| {
                                                    let (rect, _) = ui.allocate_exact_size(
                                                        egui::vec2(10.0, 10.0),
                                                        egui::Sense::hover(),
                                                    );
                                                    ui.painter().rect_filled(
                                                        rect,
                                                        2.0,
                                                        egui::Color32::from_rgb(r, g, b),
                                                    );
                                                    ui.selectable_value(
                                                        &mut state.selected_material_id,
                                                        id,
                                                        Atom::material_name(id),
                                                    );
                                                });
                                            }
                                        }
                                    });
                            });
                        }
                        ParticleSpawnType::Explosion => {
                            ui.separator();
                            ui.label("Explosion Settings:");
                            ui.horizontal(|ui| {
                                ui.label("Blast Radius:");
                                ui.add(egui::Slider::new(&mut state.particle_burst_radius, 1..=16).suffix(" px"));
                            });
                            ui.small("Clicking the world detonate-ejects all terrain atoms within radius radially outwards!");
                        }
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        if let Some(cursor) = cursor_global {
                            if ui.button("💥 Trigger at Cursor").clicked() {
                                if let Some(ref mut pool) = particle_pool {
                                    let rad = state.particle_angle_deg.to_radians();
                                    let vel = [rad.cos() * state.particle_speed, rad.sin() * state.particle_speed];
                                    let spawn_pos = [cursor.x as f32 + 0.5, cursor.y as f32 + 0.5];
                                    match state.particle_spawn_type {
                                        ParticleSpawnType::Fx => {
                                            pool.spawn_fx(
                                                spawn_pos,
                                                vel,
                                                state.particle_fx_color,
                                                state.particle_fx_lifetime,
                                                state.particle_fx_bounces,
                                            );
                                        }
                                        ParticleSpawnType::Matter => {
                                            if !pool.eject_from_grid_dirty(
                                                &mut chunk_world,
                                                &mut dirty_rects,
                                                cursor.x,
                                                cursor.y,
                                                vel,
                                            ) {
                                                let mut atom = Atom::new_with_rng(
                                                    state.selected_material_id,
                                                    &materials,
                                                    &mut fastrand::Rng::new(),
                                                );
                                                atom.set_free_falling(true);
                                                atom.set_life_5bit(0);
                                                pool.push_particle(Particle {
                                                    pos: spawn_pos,
                                                    vel,
                                                    atom,
                                                });
                                            }
                                        }
                                        ParticleSpawnType::Explosion => {
                                            let r = state.particle_burst_radius.clamp(1, 16);
                                            for pt in circle_points(cursor, r) {
                                                let dx = pt.x as f32 - cursor.x as f32;
                                                let dy = pt.y as f32 - cursor.y as f32;
                                                let dist = (dx * dx + dy * dy).sqrt().max(0.2);
                                                let speed = (state.particle_speed * 0.5)
                                                    + fastrand::f32() * (state.particle_speed * 0.8);
                                                let bvel = [dx / dist * speed, dy / dist * speed];
                                                pool.eject_from_grid_dirty(
                                                    &mut chunk_world,
                                                    &mut dirty_rects,
                                                    pt.x,
                                                    pt.y,
                                                    bvel,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    if active_count > 0 {
                        ui.separator();
                        egui::CollapsingHeader::new(format!(
                            "Active Particle Telemetry Sample ({})",
                            active_count.min(8)
                        ))
                        .default_open(false)
                        .show(ui, |ui| {
                            if let Some(ref pool) = particle_pool {
                                for (i, p) in pool.particles().iter().take(8).enumerate() {
                                    let life = p.atom.life_5bit();
                                    let (rgba, desc) = if p.atom.material_id != 0 {
                                        (
                                            Atom::material_preview_color(p.atom.material_id),
                                            format!("Matter ({})", Atom::material_name(p.atom.material_id)),
                                        )
                                    } else {
                                        let c = if p.atom.color[3] != 0 {
                                            p.atom.color
                                        } else {
                                            [255, 255, 255, 255]
                                        };
                                        (c, format!("FX (life={})", life))
                                    };
                                    ui.horizontal(|ui| {
                                        let (rect, _) = ui.allocate_exact_size(
                                            egui::vec2(10.0, 10.0),
                                            egui::Sense::hover(),
                                        );
                                        ui.painter().rect_filled(
                                            rect,
                                            2.0,
                                            egui::Color32::from_rgb(rgba[0], rgba[1], rgba[2]),
                                        );
                                        ui.small(format!(
                                            "#{}: pos=({:.1}, {:.1}) | vel=({:.2}, {:.2}) | {}",
                                            i, p.pos[0], p.pos[1], p.vel[0], p.vel[1], desc
                                        ));
                                    });
                                }
                            }
                        });
                    }
                }

                DebugTab::Simulation => {
                    ui.heading("Simulation Controls");
                    ui.horizontal(|ui| {
                        let pause_label = if sim_settings.paused {
                            "▶ Resume (P)"
                        } else {
                            "⏸ Pause (P)"
                        };
                        if ui.button(pause_label).clicked() {
                            sim_settings.paused = !sim_settings.paused;
                        }
                        if ui
                            .add_enabled(sim_settings.paused, egui::Button::new("⏭ Step 1 Frame"))
                            .clicked()
                        {
                            sim_settings.step_once = true;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Steps / Frame:");
                        ui.add(egui::Slider::new(&mut sim_settings.steps_per_frame, 1..=4));
                        ui.checkbox(&mut sim_settings.stream_chunks, "Stream Chunks");
                    });

                    ui.separator();
                    ui.heading("Camera Controls");
                    ui.horizontal(|ui| {
                        ui.label("Zoom (View Height):");
                        ui.add(
                            egui::Slider::new(&mut state.camera_height, 64.0..=1800.0)
                                .logarithmic(true)
                                .suffix(" px"),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Presets:");
                        for (label, h) in [
                            ("Close (180)", 180.0),
                            ("Normal (240)", 240.0),
                            ("Wide (420)", 420.0),
                            ("Full (800)", 800.0),
                        ] {
                            if ui.small_button(label).clicked() {
                                state.camera_height = h;
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        if let Some(ref mut li) = local_input {
                            ui.checkbox(&mut li.camera_follow_player, "Camera Follow Player");
                        }
                        if ui.button("🎯 Snap Camera to Player").clicked() {
                            state.reset_camera_requested = true;
                        }
                    });
                    ui.small("Move: A/D | Jump: Space/W | Jetpack: Shift/Space | Zoom: Scroll | Free Pan: Middle-Drag");

                    ui.separator();
                    ui.heading("Debug Overlays");
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut state.show_actor_debug, "Actor Hitbox & Aim");
                        ui.checkbox(&mut state.show_dirty_rects, "Dirty Rects");
                        ui.checkbox(&mut state.show_chunk_grid, "Chunk Grid");
                        ui.checkbox(&mut state.show_brush_cursor, "Brush Cursor");
                        ui.checkbox(&mut state.show_particles_overlay, "Particles Overlay");
                    });
                }

                DebugTab::Player => {
                    ui.heading("Local Player & KinematicActor");
                    let mut found_player = false;
                    for (ent, (ctrl, actor)) in entities.iter_with((&player_controlled, &actors)) {
                        if ctrl.player_id == local_view.local_player_id {
                            found_player = true;
                            let input = match_inputs.players[ctrl.player_id as usize];
                            let st = actor_states.get(ent).copied().unwrap_or_default();
                            ui.small(format!(
                                "Player #{} | pos=({:.2}, {:.2}) | vel=({:.2}, {:.2})",
                                ctrl.player_id, actor.pos.x, actor.pos.y, actor.vel.x, actor.vel.y
                            ));
                            ui.small(format!(
                                "grounded={} | coyote={}t | normal=({:.2}, {:.2}) | buried={}t",
                                actor.grounded,
                                st.coyote_ticks,
                                actor.ground_normal.x,
                                actor.ground_normal.y,
                                st.buried_ticks
                            ));
                            ui.small(format!(
                                "fuel={:.1}/{:.0} | slot={} | aim=({:.2}, {:.2}) | cd={}t",
                                st.jetpack_fuel,
                                st.max_jetpack_fuel,
                                input.selected_slot,
                                input.aim_dir[0],
                                input.aim_dir[1],
                                st.fire_cooldown_ticks
                            ));
                            break;
                        }
                    }
                    if !found_player {
                        ui.small("No local player entity found.");
                    }

                    ui.horizontal(|ui| {
                        if let Some(ref mut li) = local_input {
                            ui.checkbox(&mut li.camera_follow_player, "Camera Follow Player");
                        }
                        if ui.button("Respawn at Surface").clicked() {
                            state.respawn_player_requested = true;
                        }
                    });
                }

                DebugTab::WorldGen => {
                    ui.heading("Persistence & Storage");
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save All Chunks (redb)").clicked() {
                            chunk_world.save_all_chunks(&io_worker);
                        }
                        ui.label(format!("I/O Queue: {}", io_worker.pending_loads.len()));
                    });

                    ui.separator();
                    ui.heading("Procedural Generator Settings");
                    let draft = &mut state.gen_draft;
                    ui.horizontal(|ui| {
                        ui.label("Seed:");
                        ui.add(egui::DragValue::new(&mut draft.seed).speed(1.0));
                        if ui.small_button("Random").clicked() {
                            draft.seed = fastrand::u32(..);
                        }
                    });
                    ui.add(
                        egui::Slider::new(&mut draft.scale, 128.0..=40960.0)
                            .text("Feature Scale")
                            .suffix(" px"),
                    );
                    ui.add(
                        egui::Slider::new(&mut draft.surface_level, -1024..=0)
                            .text("Surface Level (Y)"),
                    );
                    ui.add(egui::Slider::new(&mut draft.hill_amplitude, 0.0..=1600.0).text("Hills"));
                    ui.add(
                        egui::Slider::new(&mut draft.mountain_amplitude, 0.0..=2400.0)
                            .text("Mountains"),
                    );
                    ui.add(
                        egui::Slider::new(&mut draft.cave_density, 0.0..=20.0).text("Cave Density"),
                    );
                    ui.add(
                        egui::Slider::new(&mut draft.cave_fade_depth, 1.0..=2000.0)
                            .text("Cave Surface Fade"),
                    );
                    ui.add(
                        egui::Slider::new(&mut draft.ore_richness, 0.0..=5.0)
                            .text("Ore Richness (Quantity)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut draft.ore_blob_size, 0.2..=5.0)
                            .text("Ore Blob Size"),
                    );

                    ui.horizontal(|ui| {
                        if ui.button("Reset Defaults").clicked() {
                            state.gen_draft = Generator::default_config();
                        }
                        if ui.button("🔄 Regenerate Active Window").clicked() {
                            *generator = state.gen_draft;
                            let hook = generator.create_hook();
                            if let Some(ref mut gh) = generator_hook {
                                **gh = hook.clone();
                            }
                            chunk_world.regenerate_window(
                                &materials,
                                &hook,
                                &mut dirty_rects,
                            );
                            stamp_initial_window_prefabs(
                                &mut chunk_world,
                                &mut dirty_rects,
                                &mut prefab_configs,
                                &mut prefab_placed,
                                &mut pending_stamps,
                                &materials,
                                generator.seed,
                            );
                        }
                    });
                }
            }
        });
}
