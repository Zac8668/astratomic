//! Prefab definitions, image stamping, surface detection, and deterministic placement.
//!
//! Prefabs are multi-atom structures (trees, surface rocks, grass clumps, ruins) loaded from
//! `<assets_dir>/prefabs/*`. Each prefab directory contains:
//! - `spawn_config.ron`: [`PrefabSpawnConfig`] controlling slot width and placement chance.
//! - `image.png`: RGBA visual appearance of each atom.
//! - `mask.png`: Red channel gives the material ID (see `assets/atoms.ron`), alpha > 0 means solid atom.

use std::{
    collections::{HashMap, HashSet},
    fs,
    num::Wrapping,
    path::Path,
};

use bones_framework::prelude::*;
use image::RgbaImage;
use serde::{Deserialize, Serialize};

use astratomic_physics::rigidbody::{
    PhysicsBodyHandle, PixelRigidBody, RapierPhysics, spawn_pixel_rigidbody_with_components,
};
use astratomic_sim::{
    atom::{Atom, Materials},
    consts::{CHUNK_LENGHT, LOAD_HEIGHT, LOAD_WIDTH},
    world::{
        ChunkPos, ChunkWorld, DirtyRects, global_to_chunk, resolve_assets_dir, update_dirty_rects,
        update_dirty_rects_3x3,
    },
};

// Material IDs matching `assets/atoms.ron`
pub const MAT_VOID: u8 = 0;
pub const MAT_SAND: u8 = 2;
pub const MAT_GRAVEL: u8 = 4;
pub const MAT_GRASS: u8 = 6;
pub const MAT_DIRT: u8 = 7;
pub const MAT_ROCK: u8 = 8;
pub const MAT_COMPACTED_DIRT: u8 = 10;
pub const MAT_LEAVES: u8 = 11;
pub const MAT_WOOD: u8 = 12;

/// Checks if a material ID is natural ground suitable for surface prefabs.
#[inline]
pub fn is_natural_surface_ground(mat_id: u8, allow_rock: bool) -> bool {
    match mat_id {
        MAT_GRASS | MAT_DIRT | MAT_COMPACTED_DIRT | MAT_SAND | MAT_GRAVEL => true,
        MAT_ROCK => allow_rock,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, HasSchema)]
#[repr(C)]
pub struct PrefabSpawnConfig {
    pub slot_width: i32,
    pub slot_chance: f32,
}

#[derive(Clone, Debug)]
pub struct PrefabConfig {
    /// Canonical relative path used for deterministic seed salting (e.g. `"prefabs/tree"`).
    pub name: String,
    /// Short human-friendly folder name (e.g. `"tree"`).
    pub display_name: String,
    pub spawn_config: PrefabSpawnConfig,
    pub positions: Vec<i32>,
    pub image: RgbaImage,
    pub mask: RgbaImage,
    pub mask_bounds: Option<(i32, i32, i32)>,
}

#[derive(HasSchema, Clone, Default, Debug)]
#[schema(opaque)]
pub struct PrefabConfigs(pub Vec<PrefabConfig>);

impl PrefabConfigs {
    /// Load all prefab definitions from `<assets_dir>/prefabs/*`.
    pub fn load_from_assets_dir(assets_dir: impl AsRef<Path>) -> Self {
        let prefabs_dir = assets_dir.as_ref().join("prefabs");
        let mut configs = Vec::new();

        if let Ok(entries) = fs::read_dir(&prefabs_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };

                let config_path = path.join("spawn_config.ron");
                let image_path = path.join("image.png");
                let mask_path = path.join("mask.png");

                let (Ok(config_bytes), Ok(image_bytes), Ok(mask_bytes)) = (
                    fs::read(&config_path),
                    fs::read(&image_path),
                    fs::read(&mask_path),
                ) else {
                    continue;
                };

                let Ok(spawn_config) = ron::de::from_bytes::<PrefabSpawnConfig>(&config_bytes)
                else {
                    continue;
                };
                let (Ok(image_dyn), Ok(mask_dyn)) = (
                    image::load_from_memory(&image_bytes),
                    image::load_from_memory(&mask_bytes),
                ) else {
                    continue;
                };

                let image = image_dyn.into_rgba8();
                let mask = mask_dyn.into_rgba8();
                let mask_bounds = prefab_mask_bounds(&mask);

                configs.push(PrefabConfig {
                    name: format!("prefabs/{dir_name}"),
                    display_name: dir_name.to_string(),
                    spawn_config,
                    positions: Vec::new(),
                    image,
                    mask,
                    mask_bounds,
                });
            }
        }

        configs.sort_by(|a, b| a.name.cmp(&b.name));
        Self(configs)
    }
}

/// Tracks occupied horizontal intervals and chunk columns that have had prefabs generated.
#[derive(HasSchema, Default, Clone, Debug)]
#[schema(opaque)]
pub struct PrefabPlaced {
    pub x_ranges: Vec<(i32, i32)>,
    pub columns_generated: HashSet<i32>,
}

impl PrefabPlaced {
    pub fn is_blocked(&self, start: i32, end: i32, padding: i32) -> bool {
        let new_start = start - padding;
        let new_end = end + padding;
        for &(s, e) in &self.x_ranges {
            if new_start <= e && new_end >= s {
                return true;
            }
        }
        false
    }

    pub fn insert_range_merged(&mut self, start: i32, end: i32, padding: i32) {
        let (s_new, e_new) = if start <= end {
            (start - padding, end + padding)
        } else {
            (end - padding, start + padding)
        };

        self.x_ranges.push((s_new, e_new));
        self.x_ranges.sort_unstable_by_key(|&(s, _)| s);

        let mut merged: Vec<(i32, i32)> = Vec::with_capacity(self.x_ranges.len());
        for (s, e) in self.x_ranges.drain(..) {
            if let Some((ms, me)) = merged.last_mut() {
                if s <= *me {
                    *me = (*me).max(e);
                    *ms = (*ms).min(s);
                } else {
                    merged.push((s, e));
                }
            } else {
                merged.push((s, e));
            }
        }

        self.x_ranges = merged;
    }

    pub fn clear(&mut self) {
        self.x_ranges.clear();
        self.columns_generated.clear();
    }
}

#[derive(HasSchema, Clone, Copy, Default, Debug)]
#[schema(opaque)]
#[repr(C)]
pub struct PendingStampCell {
    pub local: UVec2,
    pub id: u8,
    pub color: [u8; 4],
}

#[derive(HasSchema, Default, Clone, Debug)]
pub struct PendingPrefabStamps(pub HashMap<IVec2, Vec<PendingStampCell>>);

pub fn get_prefabs_in_chunk(
    slot_width: i32,
    slot_chance: f32,
    chunk_start_x: i32,
    seed: u32,
) -> Vec<i32> {
    if slot_width <= 0 {
        return Vec::new();
    }

    let chunk_end_x = chunk_start_x + CHUNK_LENGHT as i32;
    let mut positions = Vec::new();

    let start_slot = (chunk_start_x as f32 / slot_width as f32).floor() as i32;
    let end_slot = (((chunk_end_x - 1) as f32) / slot_width as f32).floor() as i32;

    for slot_idx in start_slot..=end_slot {
        if get_random_at(seed, slot_idx) > slot_chance {
            continue;
        }

        let offset_rng = get_random_at(seed.wrapping_add(1), slot_idx);
        let offset = (offset_rng * slot_width as f32) as i32;
        let prefab_x = (slot_idx * slot_width) + offset;

        positions.push(prefab_x);
    }
    positions
}

/// Returns deterministic prefab X positions that fall within the given chunk.
pub fn prefab_xs_for_chunk(config: &PrefabConfig, chunk_x: i32, seed: u32) -> Vec<i32> {
    let chunk_start_x = chunk_x * CHUNK_LENGHT as i32;
    let chunk_end_x = chunk_start_x + CHUNK_LENGHT as i32 - 1;
    let salted_seed = seed ^ prefab_seed_salt(&config.name);

    let mut xs = get_prefabs_in_chunk(
        config.spawn_config.slot_width,
        config.spawn_config.slot_chance,
        chunk_start_x,
        salted_seed,
    );
    xs.retain(|&x| (chunk_start_x..=chunk_end_x).contains(&x));
    xs.sort_unstable();
    xs.dedup();
    xs
}

/// Deterministic 32-bit FNV-1a hash used to salt prefab RNG.
pub fn prefab_seed_salt(name: &str) -> u32 {
    let mut hash: u32 = 0x811C_9DC5;
    for &b in name.as_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

pub fn prefab_mask_bounds(mask: &RgbaImage) -> Option<(i32, i32, i32)> {
    let w = mask.width() as usize;
    let h = mask.height() as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let raw = mask.as_raw();
    if raw.len() < w * h * 4 {
        return None;
    }

    let mut min_x: i32 = i32::MAX;
    let mut max_x: i32 = i32::MIN;
    let mut bottom_y: i32 = i32::MIN;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let id = raw[idx];
            let alpha = raw[idx + 3];
            if alpha == 0 || id == 0 {
                continue;
            }

            let x_i = x as i32;
            let y_i = y as i32;
            min_x = min_x.min(x_i);
            max_x = max_x.max(x_i);
            bottom_y = bottom_y.max(y_i);
        }
    }

    (bottom_y != i32::MIN).then_some((min_x, max_x, bottom_y))
}

/// Find the true sky-exposed surface ground level at global coordinate `x`.
///
/// Raycasts strictly downwards from the top of the loaded window (`min_y`) towards `max_y`.
/// Returns `Some(surface_air_y)` only if:
/// 1. The top of the window at `(x, min_y)` is open sky (void). If `min_y` is solid, the column
///    is entirely underground (deep in caves), so no surface prefab can spawn.
/// 2. The first non-void atom hit from the sky is natural surface ground (`MAT_GRASS`, `MAT_DIRT`, etc.).
///    If it hits leaves or wood (an existing tree) or a cave ceiling/liquid, it is rejected.
/// 3. The ground directly beneath `surface_air_y + 1` has solid foundation (not a 1-pixel hollow crust).
pub fn find_surface_air_y(
    chunk_world: &ChunkWorld,
    x: i32,
    min_y: i32,
    max_y: i32,
    allow_rock: bool,
) -> Option<i32> {
    if min_y >= max_y {
        return None;
    }

    // 1. Must start in open sky (void). If min_y is already solid, we are underground in caves!
    let top_atom = chunk_world.get_atom(&global_to_chunk(IVec2::new(x, min_y)))?;
    if !top_atom.is_void() {
        return None;
    }

    // 2. Scan downwards from the sky until we hit the first solid atom.
    let mut y = min_y;
    while y <= max_y {
        let chunk_pos = global_to_chunk(IVec2::new(x, y));
        let atom = chunk_world.get_atom(&chunk_pos)?;

        if !atom.is_void() {
            let surface_solid_y = y;
            let surface_air_y = y - 1;

            // Must be natural surface ground (NOT leaves, wood, ores, liquids)
            if !is_natural_surface_ground(atom.material_id, allow_rock) {
                return None;
            }

            // 3. Foundation check: the ground beneath must not be a 1-pixel hollow crust over a cave.
            let under_pos = global_to_chunk(IVec2::new(x, surface_solid_y + 1));
            if let Some(under_atom) = chunk_world.get_atom(&under_pos)
                && under_atom.is_void()
            {
                return None;
            }

            return Some(surface_air_y);
        }

        y += 1;
    }

    None
}

fn queue_pending_stamp(
    pending: &mut PendingPrefabStamps,
    world_pos: IVec2,
    id: u8,
    color: [u8; 4],
) {
    let chunk_pos = global_to_chunk(world_pos);
    pending
        .0
        .entry(chunk_pos.chunk)
        .or_default()
        .push(PendingStampCell {
            local: chunk_pos.atom,
            id,
            color,
        });
}

pub fn apply_pending_stamps_for_chunk(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    materials: &Materials,
    pending: &mut PendingPrefabStamps,
    chunk: IVec2,
) {
    let Some(cells) = pending.0.remove(&chunk) else {
        return;
    };

    for cell in cells {
        if !(2..=12).contains(&cell.id) {
            continue;
        }

        let chunk_pos = ChunkPos::new(cell.local, chunk);
        let Some(atom) = chunk_world.get_mut_atom(chunk_pos) else {
            continue;
        };

        if !atom.is_void() {
            continue;
        }

        let mut new_atom = Atom::new(cell.id, materials);
        new_atom.color = cell.color;
        *atom = new_atom;

        update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
        update_dirty_rects(&mut dirty_rects.render, chunk_pos);
    }
}

pub fn stamp_prefab_at_x(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    (prefab_placed, pending): (&mut PrefabPlaced, &mut PendingPrefabStamps),
    materials: &Materials,
    config: &PrefabConfig,
    x: i32,
    bounds_y: (i32, i32),
) -> bool {
    let (min_y, max_y) = bounds_y;
    let Some((mask_min_x, mask_max_x, bottom)) = config.mask_bounds else {
        return false;
    };

    let footprint_start_x = x + mask_min_x;
    let footprint_end_x = x + mask_max_x;

    let is_tree = config.name.contains("tree");
    let is_rock = config.name.contains("rock");
    let padding = if is_tree { 8 } else { 3 };

    if prefab_placed.is_blocked(footprint_start_x, footprint_end_x, padding) {
        return false;
    }

    let allow_rock_ground = is_rock;
    let Some(surface_air_y) = find_surface_air_y(chunk_world, x, min_y, max_y, allow_rock_ground)
    else {
        return false;
    };

    let w = config.mask.width() as usize;
    let h = config.mask.height() as usize;
    if w == 0 || h == 0 {
        return false;
    }

    let mask_raw = config.mask.as_raw();
    let image_raw = config.image.as_raw();
    if mask_raw.len() < w * h * 4 || image_raw.len() < w * h * 4 {
        return false;
    }

    let origin = IVec2::new(x, surface_air_y - bottom);

    // 1. Verify ground foundation across the base footprint:
    // Check that beneath the prefab base, ground is solid (not hollow or a cliff edge).
    let base_width = (footprint_end_x - footprint_start_x + 1).max(1);
    let mut solid_base_count = 0;
    for check_x in footprint_start_x..=footprint_end_x {
        let is_solid = (0..=2).any(|dy| {
            let p = global_to_chunk(IVec2::new(check_x, surface_air_y + 1 + dy));
            chunk_world
                .get_atom(&p)
                .is_some_and(|a| is_natural_surface_ground(a.material_id, allow_rock_ground))
        });
        if is_solid {
            solid_base_count += 1;
        }
    }
    let min_ratio = if is_tree { 6 } else { 5 };
    if solid_base_count * 10 < base_width * min_ratio {
        return false;
    }

    // 2. Verify clearance in the stamp volume:
    // The volume where solid prefab atoms will sit must NOT overlap existing
    // solid atoms (like leaves or wood from an existing tree, rocks, or terrain).
    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 4;
            let alpha = mask_raw[idx + 3];
            let id = mask_raw[idx];
            if alpha == 0 || id == 0 {
                continue;
            }

            let cell_world_pos = origin + IVec2::new(px as i32, py as i32);
            let chunk_pos = global_to_chunk(cell_world_pos);
            if let Some(atom) = chunk_world.get_atom(&chunk_pos)
                && !atom.is_void()
            {
                // Space is already occupied! Do not stamp on top of existing structures.
                return false;
            }
        }
    }

    // 3. Stamp atoms
    let mut placed_any = false;
    let mut queued_any = false;

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 4;
            let id = mask_raw[idx];
            let alpha = mask_raw[idx + 3];
            if alpha == 0 || id == 0 || !(2..=12).contains(&id) {
                continue;
            }

            let color = [
                image_raw[idx],
                image_raw[idx + 1],
                image_raw[idx + 2],
                image_raw[idx + 3],
            ];

            let world_pos = origin + IVec2::new(px as i32, py as i32);
            let chunk_pos = global_to_chunk(world_pos);
            let Some(atom) = chunk_world.get_mut_atom(chunk_pos) else {
                queue_pending_stamp(pending, world_pos, id, color);
                queued_any = true;
                continue;
            };

            let mut new_atom = Atom::new(id, materials);
            new_atom.color = color;
            *atom = new_atom;
            placed_any = true;

            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
        }
    }

    if placed_any || queued_any {
        prefab_placed.insert_range_merged(footprint_start_x, footprint_end_x, padding);
    }

    placed_any
}

/// Stamp a prefab directly at the given global atom coordinate `(anchor_x, anchor_y)`
/// (used by the debug UI prefab stamp tool), optionally snapping to the surface.
pub fn stamp_prefab_manual(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    materials: &Materials,
    config: &PrefabConfig,
    world_pos: IVec2,
    snap_to_surface: bool,
    overwrite_existing: bool,
) -> bool {
    let Some((mask_min_x, mask_max_x, bottom)) = config.mask_bounds else {
        return false;
    };

    let w = config.mask.width() as usize;
    let h = config.mask.height() as usize;
    if w == 0 || h == 0 {
        return false;
    }

    let mask_raw = config.mask.as_raw();
    let image_raw = config.image.as_raw();
    if mask_raw.len() < w * h * 4 || image_raw.len() < w * h * 4 {
        return false;
    }

    let center_offset_x = (mask_min_x + mask_max_x) / 2;
    let target_x = world_pos.x - center_offset_x;

    let target_bottom_y = if snap_to_surface {
        let min_y = chunk_world.pos.y * CHUNK_LENGHT as i32;
        let max_y = (chunk_world.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;
        let is_rock = config.name.contains("rock");
        find_surface_air_y(chunk_world, world_pos.x, min_y, max_y, is_rock).unwrap_or(world_pos.y)
    } else {
        world_pos.y
    };

    let origin = IVec2::new(target_x, target_bottom_y - bottom);
    let mut placed_any = false;

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 4;
            let id = mask_raw[idx];
            let alpha = mask_raw[idx + 3];
            if alpha == 0 || id == 0 || !(2..=12).contains(&id) {
                continue;
            }

            let color = [
                image_raw[idx],
                image_raw[idx + 1],
                image_raw[idx + 2],
                image_raw[idx + 3],
            ];

            let cell_world_pos = origin + IVec2::new(px as i32, py as i32);
            let chunk_pos = global_to_chunk(cell_world_pos);
            let Some(atom) = chunk_world.get_mut_atom(chunk_pos) else {
                continue;
            };

            if !overwrite_existing && !atom.is_void() {
                continue;
            }

            let mut new_atom = Atom::new(id, materials);
            new_atom.color = color;
            *atom = new_atom;
            placed_any = true;

            update_dirty_rects_3x3(&mut dirty_rects.current, chunk_pos);
            update_dirty_rects(&mut dirty_rects.render, chunk_pos);
        }
    }

    placed_any
}

/// Stamp prefabs across all columns in the initial window.
pub fn stamp_initial_window_prefabs(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    prefab_configs: &mut PrefabConfigs,
    prefab_placed: &mut PrefabPlaced,
    pending_prefab_stamps: &mut PendingPrefabStamps,
    materials: &Materials,
    seed: u32,
) {
    prefab_placed.clear();
    pending_prefab_stamps.0.clear();

    // Prioritize large structures (trees) first, then rocks, then grass clumps
    let mut sorted_configs = prefab_configs.0.clone();
    sorted_configs.sort_by_key(|c| {
        if c.name.contains("tree") {
            0
        } else if c.name.contains("rock") {
            1
        } else {
            2
        }
    });

    for config in &mut sorted_configs {
        config.positions.clear();
        for x in chunk_world.pos.x..(chunk_world.pos.x + LOAD_WIDTH) {
            config
                .positions
                .extend(prefab_xs_for_chunk(config, x, seed));
        }
        config.positions.sort_unstable();
        config.positions.dedup();
    }

    let min_y = chunk_world.pos.y * CHUNK_LENGHT as i32;
    let max_y = (chunk_world.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;

    for config in &sorted_configs {
        for &x in &config.positions {
            let _ = stamp_prefab_at_x(
                chunk_world,
                dirty_rects,
                (prefab_placed, pending_prefab_stamps),
                materials,
                config,
                x,
                (min_y, max_y),
            );
        }
    }

    for x in chunk_world.pos.x..(chunk_world.pos.x + LOAD_WIDTH) {
        prefab_placed.columns_generated.insert(x);
    }
}

/// Stamp prefabs for a newly loaded or generated chunk at `chunk_pos`.
pub fn stamp_prefabs_for_new_chunk(
    chunk_world: &mut ChunkWorld,
    dirty_rects: &mut DirtyRects,
    prefab_placed: &mut PrefabPlaced,
    pending_prefab_stamps: &mut PendingPrefabStamps,
    prefab_configs: &PrefabConfigs,
    materials: &Materials,
    seed: u32,
    chunk_pos: IVec2,
) {
    apply_pending_stamps_for_chunk(
        chunk_world,
        dirty_rects,
        materials,
        pending_prefab_stamps,
        chunk_pos,
    );

    // Only generate prefabs once per X column:
    if !prefab_placed.columns_generated.insert(chunk_pos.x) {
        return;
    }

    let min_y = chunk_world.pos.y * CHUNK_LENGHT as i32;
    let max_y = (chunk_world.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;

    let mut sorted_configs = prefab_configs.0.clone();
    sorted_configs.sort_by_key(|c| {
        if c.name.contains("tree") {
            0
        } else if c.name.contains("rock") {
            1
        } else {
            2
        }
    });

    for config in &sorted_configs {
        let xs = prefab_xs_for_chunk(config, chunk_pos.x, seed);
        for x in xs {
            let _ = stamp_prefab_at_x(
                chunk_world,
                dirty_rects,
                (prefab_placed, pending_prefab_stamps),
                materials,
                config,
                x,
                (min_y, max_y),
            );
        }
    }
}

fn get_random_at(seed: u32, x: i32) -> f32 {
    let mut n = Wrapping(x as u32);
    n *= Wrapping(0xB529_7A4D);
    n += Wrapping(seed);
    n ^= n >> 8;
    n += Wrapping(0x68E3_1DA4);
    n ^= n << 8;
    n *= Wrapping(0x1B56_C4E9);
    n ^= n >> 8;

    (n.0 as f32) / (u32::MAX as f32)
}

/// Registers all prefab schemas with the global `bones_schema` registry.
pub fn register_prefab_schemas() {
    PrefabSpawnConfig::register_schema();
    PrefabConfigs::register_schema();
    PrefabPlaced::register_schema();
    PendingStampCell::register_schema();
    PendingPrefabStamps::register_schema();
}

/// Startup system that stamps initial window prefabs after chunk generation.
pub fn setup_prefabs_system(
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut prefab_configs: ResMut<PrefabConfigs>,
    mut prefab_placed: ResMut<PrefabPlaced>,
    mut pending_stamps: ResMut<PendingPrefabStamps>,
    materials: Res<Materials>,
    generator: Option<Res<crate::worldgen::Generator>>,
) {
    let seed = generator.as_ref().map(|g| g.seed).unwrap_or(8668);
    stamp_initial_window_prefabs(
        &mut chunk_world,
        &mut dirty_rects,
        &mut prefab_configs,
        &mut prefab_placed,
        &mut pending_stamps,
        &materials,
        seed,
    );
}

/// PreUpdate system in `astratomic_game` that stamps prefabs and applies pending stamps
/// whenever chunks are loaded or streamed into `ChunkWorld`.
pub fn prefab_streaming_system(
    mut chunk_world: ResMut<ChunkWorld>,
    mut dirty_rects: ResMut<DirtyRects>,
    mut prefab_placed: ResMut<PrefabPlaced>,
    mut pending_stamps: ResMut<PendingPrefabStamps>,
    prefab_configs: Res<PrefabConfigs>,
    materials: Res<Materials>,
    generator: Option<Res<crate::worldgen::Generator>>,
) {
    let seed = generator.as_ref().map(|g| g.seed).unwrap_or(8668);

    // 1. Apply any pending stamps for chunks that are in `chunk_world`
    let pending_chunks: Vec<IVec2> = pending_stamps.0.keys().copied().collect();
    for chunk in pending_chunks {
        if chunk_world.chunks.contains_key(&chunk) {
            apply_pending_stamps_for_chunk(
                &mut chunk_world,
                &mut dirty_rects,
                &materials,
                &mut pending_stamps,
                chunk,
            );
        }
    }

    // 2. For every loaded chunk, ensure its column has had prefabs generated
    let loaded_coords: Vec<IVec2> = chunk_world.chunks.keys().copied().collect();
    for pos in loaded_coords {
        if !prefab_placed.columns_generated.contains(&pos.x) {
            stamp_prefabs_for_new_chunk(
                &mut chunk_world,
                &mut dirty_rects,
                &mut prefab_placed,
                &mut pending_stamps,
                &prefab_configs,
                &materials,
                seed,
                pos,
            );
        }
    }
}

/// Bones plugin for prefab management and placement.
#[derive(Default, Clone, Copy)]
pub struct AstratomicPrefabsPlugin;

impl GamePlugin for AstratomicPrefabsPlugin {
    fn install(self, _game: &mut Game) {
        register_prefab_schemas();
    }
}

impl SessionPlugin for AstratomicPrefabsPlugin {
    fn install(self, session: &mut Session) {
        register_prefab_schemas();

        let assets_dir = resolve_assets_dir();
        let configs = PrefabConfigs::load_from_assets_dir(&assets_dir);

        session.world.insert_resource(configs);
        session.world.init_resource::<PrefabPlaced>();
        session.world.init_resource::<PendingPrefabStamps>();

        session
            .add_system_to_stage(CoreStage::Startup, setup_prefabs_system)
            .add_system_to_stage(CoreStage::PreUpdate, prefab_streaming_system);
    }
}

/// Creates a dynamic [`PixelRigidBody`] from a [`PrefabConfig`].
///
/// Reads the material ID from the `mask.png` red channel (where alpha > 0),
/// extracts colors from `image.png`, and initializes atom states from `materials`.
pub fn pixel_body_from_prefab(
    config: &PrefabConfig,
    materials: Option<&Materials>,
    spawn_pos: Vec2,
) -> PixelRigidBody {
    let w = config.mask.width() as usize;
    let h = config.mask.height() as usize;
    let mut atoms = vec![Atom::AIR; w * h];

    let mask_raw = config.mask.as_raw();
    let image_raw = config.image.as_raw();

    for py in 0..h {
        let body_y = h - 1 - py;
        for px in 0..w {
            let idx = (py * w + px) * 4;
            let id = if idx < mask_raw.len() { mask_raw[idx] } else { 0 };
            let alpha = if idx + 3 < mask_raw.len() { mask_raw[idx + 3] } else { 0 };

            if alpha != 0 && id != 0 {
                let color = if idx + 3 < image_raw.len() {
                    [
                        image_raw[idx],
                        image_raw[idx + 1],
                        image_raw[idx + 2],
                        image_raw[idx + 3].max(255),
                    ]
                } else {
                    [200, 200, 200, 255]
                };

                let state = materials
                    .and_then(|m| m.0.get(id as usize))
                    .map(|m| m.default_state)
                    .unwrap_or_else(|| astratomic_sim::atom::AtomState::default_for_material_id(id));

                atoms[body_y * w + px] = Atom::with_state(id, state, color);
            }
        }
    }

    PixelRigidBody {
        width: w as u16,
        height: h as u16,
        atoms: std::sync::Arc::new(atoms),
        prev_pos: spawn_pos,
        pos: spawn_pos,
        prev_angle: 0.0,
        angle: 0.0,
        image_handle: None,
        render_cid: None,
        last_linvel: Vec2::ZERO,
        prev_submerged_liquid_count: 0,
    }
}

/// Spawns a prefab as a dynamic [`PixelRigidBody`] into Rapier physics and ECS components.
///
/// The dynamic collider is generated using Marching Squares + RDP polyline simplification.
pub fn spawn_prefab_as_rigidbody(
    config: &PrefabConfig,
    materials: Option<&Materials>,
    spawn_pos: Vec2,
    physics: &mut RapierPhysics,
    entities: &mut Entities,
    handles: &mut CompMut<PhysicsBodyHandle>,
    pixel_bodies: &mut CompMut<PixelRigidBody>,
    transforms: &mut CompMut<Transform>,
) -> Entity {
    let body = pixel_body_from_prefab(config, materials, spawn_pos);
    let ent = entities.create();
    spawn_pixel_rigidbody_with_components(
        ent,
        physics,
        materials,
        body,
        handles,
        pixel_bodies,
        transforms,
    );
    ent
}

#[cfg(test)]
mod tests {
    use super::*;
    use astratomic_sim::chunk::Chunk;

    #[test]
    fn prefabs_load_from_assets() {
        let prefabs = PrefabConfigs::load_from_assets_dir(resolve_assets_dir());
        assert!(
            !prefabs.0.is_empty(),
            "expected prefabs to load from assets/prefabs"
        );
        assert!(prefabs.0.iter().any(|p| p.name.contains("tree")));
    }

    #[test]
    fn surface_finder_rejects_underground_caves() {
        let mut world = ChunkWorld::default();
        let chunk_pos = ivec2(0, 0);
        let mut chunk = Chunk::default();

        // Simulate an underground chunk where the top is solid rock, with a cave in the middle
        // (y = 0..10 is rock, y = 11..20 is cave air, y = 21..63 is rock)
        for y in 0..64 {
            for x in 0..64 {
                let id = if (11..=20).contains(&y) {
                    MAT_VOID
                } else {
                    MAT_ROCK
                };
                chunk.atoms_mut()[y * 64 + x] = Atom::with_state(
                    id,
                    astratomic_sim::atom::AtomState::Solid,
                    [100, 100, 100, 255],
                );
            }
        }
        world.add_chunk(chunk_pos, chunk);

        // find_surface_air_y starting from min_y = 0:
        // Top of window is solid rock (underground cave environment).
        // It MUST return None because there is no open sky above!
        let surface = find_surface_air_y(&world, 10, 0, 63, true);
        assert_eq!(
            surface, None,
            "should reject cave air when min_y is underground"
        );
    }

    #[test]
    fn surface_finder_detects_real_surface_and_rejects_leaves() {
        let mut world = ChunkWorld::default();
        let chunk_pos = ivec2(0, 0);
        let mut chunk = Chunk::default();

        // y = 0..19 is void (sky)
        // y = 20..63 is grass / dirt
        for y in 20..64 {
            for x in 0..64 {
                let id = if y == 20 { MAT_GRASS } else { MAT_DIRT };
                chunk.atoms_mut()[y * 64 + x] = Atom::with_state(
                    id,
                    astratomic_sim::atom::AtomState::Solid,
                    [50, 150, 50, 255],
                );
            }
        }
        world.add_chunk(chunk_pos, chunk);

        // From sky (min_y = 0), should hit ground at y = 20, returning surface_air_y = 19
        let surface = find_surface_air_y(&world, 10, 0, 63, false);
        assert_eq!(surface, Some(19));

        // Now place tree leaves at (10, 10) in the sky
        let atom = world
            .get_mut_atom(ChunkPos::new(uvec2(10, 10), chunk_pos))
            .unwrap();
        *atom = Atom::with_state(
            MAT_LEAVES,
            astratomic_sim::atom::AtomState::Solid,
            [0, 255, 0, 255],
        );

        // find_surface_air_y should reject leaves because leaves are NOT natural ground!
        let surface_on_tree = find_surface_air_y(&world, 10, 0, 63, false);
        assert_eq!(
            surface_on_tree, None,
            "should reject spawning on top of tree leaves"
        );
    }

    #[test]
    fn stamping_prevents_overlap() {
        let mut world = ChunkWorld::default();
        let chunk_pos = ivec2(0, 0);
        let mut chunk = Chunk::default();

        // Terrain with grass at y = 20, dirt below
        for y in 20..64 {
            for x in 0..64 {
                let id = if y == 20 { MAT_GRASS } else { MAT_DIRT };
                chunk.atoms_mut()[y * 64 + x] = Atom::with_state(
                    id,
                    astratomic_sim::atom::AtomState::Solid,
                    [50, 150, 50, 255],
                );
            }
        }
        world.add_chunk(chunk_pos, chunk);

        let materials = Materials::default_builtin();
        let configs = PrefabConfigs::load_from_assets_dir(resolve_assets_dir());
        let tree_config = configs.0.iter().find(|c| c.name.contains("tree")).unwrap();

        let mut dirty_rects = DirtyRects::default();
        let mut placed = PrefabPlaced::default();
        let mut pending = PendingPrefabStamps::default();

        // First tree stamp should succeed
        let placed_first = stamp_prefab_at_x(
            &mut world,
            &mut dirty_rects,
            (&mut placed, &mut pending),
            &materials,
            tree_config,
            25,
            (0, 63),
        );
        assert!(placed_first, "first tree should be successfully placed");

        // Second stamp within the tree's footprint should be rejected!
        let placed_overlapping = stamp_prefab_at_x(
            &mut world,
            &mut dirty_rects,
            (&mut placed, &mut pending),
            &materials,
            tree_config,
            28,
            (0, 63),
        );
        assert!(
            !placed_overlapping,
            "overlapping tree stamp should be rejected"
        );
    }

    #[test]
    fn test_spawn_prefab_as_rigidbody() {
        let configs = PrefabConfigs::load_from_assets_dir(resolve_assets_dir());
        let tree_config = configs.0.iter().find(|c| c.name.contains("tree")).unwrap();
        let materials = Materials::default_builtin();

        let world = World::default();
        world.init_resource::<Entities>();
        world.init_resource::<RapierPhysics>();

        let body = pixel_body_from_prefab(tree_config, Some(&materials), vec2(0.0, 50.0));
        assert_eq!(body.width, tree_config.mask.width() as u16);
        assert_eq!(body.height, tree_config.mask.height() as u16);
        assert!(!body.atoms.is_empty());
        assert!(body.atoms.iter().any(|a| !a.is_air()));

        let ent = {
            let mut physics = world.resource_mut::<RapierPhysics>();
            let mut entities = world.resource_mut::<Entities>();
            let mut handles = world.component_mut::<PhysicsBodyHandle>();
            let mut pixel_bodies = world.component_mut::<PixelRigidBody>();
            let mut transforms = world.component_mut::<Transform>();

            spawn_prefab_as_rigidbody(
                tree_config,
                Some(&materials),
                vec2(0.0, 50.0),
                &mut physics,
                &mut entities,
                &mut handles,
                &mut pixel_bodies,
                &mut transforms,
            )
        };

        let physics = world.resource::<RapierPhysics>();
        let handles = world.component::<PhysicsBodyHandle>();
        assert!(!physics.bodies.is_empty());
        assert!(!physics.colliders.is_empty());
        let h = handles.get(ent).expect("Entity must have PhysicsBodyHandle");
        let rb = physics.bodies.get(h.handle()).expect("RigidBody must exist in Rapier");
        assert!(rb.is_dynamic());
    }
}
