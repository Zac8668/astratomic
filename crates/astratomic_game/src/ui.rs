//! Egui world debug panel, material brush, prefab stamper, camera controls, and debug overlays.

use std::collections::HashSet;

use astratomic_sim::{
    atom::{Atom, Materials},
    chunk::CHUNK_SIZE,
    geom_tools::{Line, circle_points, square_points},
    prefabs::{PendingPrefabStamps, PrefabConfigs, PrefabPlaced, stamp_prefab_manual},
    world::{
        ChunkWorld, DirtyRects, Generator, SimulationSettings, global_to_chunk,
        world_to_global_atom,
    },
    world_store::WorldIoWorker,
};
use bones_framework::prelude::*;

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
    Brush = 0,
    PrefabStamp = 1,
}

/// Session resource holding the state of the world debug UI, camera, and brush tools.
#[derive(HasSchema, Clone, Debug)]
#[schema(opaque)]
pub struct DebugToolState {
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
    pub camera_height: f32,
    pub camera_speed: f32,
    pub reset_camera_requested: bool,
    pub gen_scale: f64,
    pub gen_seed: u32,
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
}

impl Default for DebugToolState {
    fn default() -> Self {
        Self {
            tool_mode: DebugToolMode::Brush,
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
            camera_height: 420.0,
            camera_speed: 320.0,
            reset_camera_requested: false,
            gen_scale: 512.0,
            gen_seed: 8668,
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
        }
    }
}

pub fn register_debug_ui_schemas() {
    BrushShape::register_schema();
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
    mut state: ResMut<DebugToolState>,
) {
    state.just_pressed_keys.clear();
    state.just_pressed_mouse.clear();

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

/// Controls the 2D camera pan (WASD / Arrow keys / Middle-Mouse drag) and zoom (scroll wheel),
/// and updates `SimulationSettings::focus_world_pos` for chunk streaming.
pub fn camera_control_system(
    time: Res<Time>,
    mouse: Option<Res<MouseInputs>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    egui_ctx: Option<Res<EguiCtx>>,
    entities: Res<Entities>,
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
        if state.just_pressed_keys.contains(&KeyCode::Space) {
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
                state.tool_mode = DebugToolMode::Brush;
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

    let mut pan_dir = Vec2::ZERO;
    if !egui_wants_kb {
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
            transform.translation.x = 0.0;
            transform.translation.y = 200.0;
            state.reset_camera_requested = false;
        }
        camera.size = CameraSize::FixedHeight(state.camera_height);
        transform.translation.x += velocity.x * dt;
        transform.translation.y += velocity.y * dt;

        // Middle-mouse drag camera panning
        if !egui_wants_pointer && state.held_mouse.contains(&MouseButton::Middle) {
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

        sim_settings.focus_world_pos = transform.translation.truncate();
    }
}

/// Applies brush painting/erasing or prefab stamping at the cursor position in world space.
pub fn world_debug_tool_system(
    egui_ctx: Option<Res<EguiCtx>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    materials: Res<Materials>,
    prefab_configs: Res<PrefabConfigs>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
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

    match state.tool_mode {
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

/// Renders wireframe debug overlays (dirty rects, chunk grid borders, and brush cursor) using `Path2d`.
pub fn debug_overlays_system(
    mouse_world: Option<Res<MouseWorldPosition>>,
    egui_ctx: Option<Res<EguiCtx>>,
    chunk_world: Res<ChunkWorld>,
    dirty_rects: Res<DirtyRects>,
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

    // 3. Brush Cursor preview overlay
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
        let preview_rgba = Atom::material_preview_color(state.selected_material_id);
        path.color = Color::rgba(
            (preview_rgba[0] as f32 / 255.0).max(0.35),
            (preview_rgba[1] as f32 / 255.0).max(0.35),
            (preview_rgba[2] as f32 / 255.0).max(0.35),
            0.9,
        );
        path.thickness = 1.0;

        if state.show_brush_cursor
            && !egui_wants_pointer
            && let Some(cursor_world) = mouse_world.and_then(|mw| mw.0)
        {
            let global = world_to_global_atom(cursor_world);
            let r = state.brush_radius.clamp(0, 32) as f32;
            let cx = global.x as f32 + 0.5;
            let cy = -(global.y as f32) - 0.5;

            match (state.tool_mode, state.brush_shape) {
                (DebugToolMode::Brush, BrushShape::Circle) if r >= 1.5 => {
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
            }
        }
    }
}

/// Renders the interactive `egui` World Debug Window.
pub fn debug_ui_system(
    egui_ctx: Option<Res<EguiCtx>>,
    mouse_world: Option<Res<MouseWorldPosition>>,
    materials: Res<Materials>,
    mut generator: ResMut<Generator>,
    mut sim_settings: ResMut<SimulationSettings>,
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut prefab_configs: ResMut<PrefabConfigs>,
    mut prefab_placed: ResMut<PrefabPlaced>,
    mut pending_stamps: ResMut<PendingPrefabStamps>,
    io_worker: Res<WorldIoWorker>,
    mut state: ResMut<DebugToolState>,
) {
    let Some(ctx) = egui_ctx else {
        return;
    };

    if !state.gen_params_initialized && generator.1 > 0.0 {
        state.gen_scale = generator.1;
        state.gen_seed = generator.2;
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
        .default_width(310.0)
        .resizable(true)
        .show(&ctx, |ui| {
            // --- Telemetry & Cursor Inspector ---
            ui.horizontal(|ui| {
                let frame_ms = 1000.0 / state.smoothed_fps.max(1.0);
                ui.label(
                    egui::RichText::new(format!(
                        "FPS: {:.0} ({:.1} ms)",
                        state.smoothed_fps, frame_ms
                    ))
                    .strong(),
                );
                ui.separator();
                ui.label(format!(
                    "Chunks: {} | Dirty: {}",
                    chunk_world.chunks.len(),
                    dirty_rects.current.len()
                ));
            });

            ui.small(format!(
                "Window Origin: ({}, {}) | Camera: ({:.0}, {:.0}) | I/O Queue: {}",
                chunk_world.pos.x,
                chunk_world.pos.y,
                sim_settings.focus_world_pos.x,
                sim_settings.focus_world_pos.y,
                io_worker.pending_loads.len()
            ));

            ui.separator();

            // Cursor Inspector
            if let (Some(global), Some(cp)) = (cursor_global, cursor_chunk_pos) {
                ui.label(format!(
                    "Cursor Atom: ({}, {}) | Chunk: ({}, {}) [{}x{}]",
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
                            "#{} {} ({:?}) vel=({},{}) moving={} idle={}",
                            atom.material_id,
                            Atom::material_name(atom.material_id),
                            atom.state(),
                            atom.vx,
                            atom.vy,
                            atom.moving(),
                            atom.f_idle()
                        ));
                    });
                } else {
                    ui.label("Atom: <unloaded chunk>");
                }
            } else {
                ui.label("Cursor: <outside window>");
            }

            ui.separator();

            // --- Simulation & Camera Controls ---
            ui.heading("Simulation & Camera");
            ui.horizontal(|ui| {
                let pause_label = if sim_settings.paused {
                    "▶ Resume (Space)"
                } else {
                    "⏸ Pause (Space)"
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
                    ("Normal (420)", 420.0),
                    ("Wide (800)", 800.0),
                    ("Full Window (1200)", 1200.0),
                ] {
                    if ui.small_button(label).clicked() {
                        state.camera_height = h;
                    }
                }
            });
            ui.small("Pan: WASD / Arrows (Shift = Fast) or Middle-Mouse Drag | Zoom: Scroll Wheel");

            ui.separator();

            // --- World Interaction Tool ---
            ui.heading("World Tool");
            ui.horizontal(|ui| {
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
            });

            match state.tool_mode {
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
            }

            ui.separator();

            // --- Debug Overlays ---
            ui.heading("Debug Overlays");
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut state.show_dirty_rects, "Dirty Rects");
                ui.checkbox(&mut state.show_chunk_grid, "Chunk Grid");
                ui.checkbox(&mut state.show_brush_cursor, "Brush Cursor");
            });

            ui.separator();

            // --- World Persistence & Procedural Generation ---
            ui.heading("Storage & World Generation");
            ui.horizontal(|ui| {
                if ui.button("💾 Save All Chunks (redb)").clicked() {
                    chunk_world.save_all_chunks(&io_worker);
                }
                if ui.button("🎯 Reset Camera to Surface").clicked() {
                    state.reset_camera_requested = true;
                }
            });

            egui::CollapsingHeader::new("Procedural Generator Settings")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Seed:");
                        ui.add(egui::DragValue::new(&mut state.gen_seed).speed(1.0));
                        if ui.small_button("Random").clicked() {
                            state.gen_seed = fastrand::u32(..);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Scale:");
                        ui.add(
                            egui::Slider::new(&mut state.gen_scale, 128.0..=1024.0).suffix(" px"),
                        );
                    });

                    if ui.button("🔄 Regenerate Active Window").clicked() {
                        *generator = Generator::from_params(
                            state.gen_scale,
                            8,
                            2.0,
                            0.8,
                            2.0,
                            state.gen_seed,
                        );
                        chunk_world.regenerate_window(
                            &materials,
                            &generator,
                            &mut prefab_configs,
                            &mut prefab_placed,
                            &mut pending_stamps,
                            &mut dirty_rects,
                        );
                    }
                });
        });
}
