use std::{collections::HashMap, fs, num::Wrapping, path::Path};

use bones_framework::prelude::{HasSchema, IVec2, UVec2};
use image::RgbaImage;
use serde::{Deserialize, Serialize};

use crate::{
    atom::{Atom, Materials},
    consts::{CHUNK_LENGHT, LOAD_HEIGHT},
    world::{
        ChunkPos, ChunkWorld, DirtyRects, global_to_chunk, update_dirty_rects,
        update_dirty_rects_3x3,
    },
};

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

#[derive(HasSchema, Default, Clone, Debug)]
pub struct PrefabPlaced {
    pub x_ranges: Vec<(i32, i32)>,
}

impl PrefabPlaced {
    pub fn insert_range_merged(&mut self, start: i32, end: i32) {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };

        self.x_ranges.push((start, end));
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

pub fn find_surface_air_y(
    chunk_world: &ChunkWorld,
    x: i32,
    start_y: i32,
    min_y: i32,
    max_y: i32,
) -> Option<i32> {
    if min_y > max_y {
        return None;
    }

    let mut y = start_y.clamp(min_y, max_y);
    let start_atom = chunk_world.get_atom(&global_to_chunk(IVec2::new(x, y)))?;

    const MAX_STEPS: i32 = 4096;
    let mut steps = 0;

    if !start_atom.is_void() {
        // Inside ground; go up (decreasing Y) to find the first air cell.
        while y >= min_y && steps < MAX_STEPS {
            let atom = chunk_world.get_atom(&global_to_chunk(IVec2::new(x, y)))?;
            if atom.is_void() {
                return Some(y);
            }
            y -= 1;
            steps += 1;
        }
    } else {
        // In air; go down (increasing Y) until we hit ground, then surface air is one above.
        while y <= max_y && steps < MAX_STEPS {
            let atom = chunk_world.get_atom(&global_to_chunk(IVec2::new(x, y)))?;
            if !atom.is_void() {
                let surface_air = y - 1;
                return (surface_air >= min_y).then_some(surface_air);
            }
            y += 1;
            steps += 1;
        }
    }

    None
}

const PREFAB_EXCLUSION_RADIUS_X: i32 = 2;

fn prefab_padded_x_range(start_x: i32, end_x: i32) -> (i32, i32) {
    (
        start_x - PREFAB_EXCLUSION_RADIUS_X,
        end_x + PREFAB_EXCLUSION_RADIUS_X,
    )
}

fn prefab_x_blocked(prefab_placed: &PrefabPlaced, start_x: i32, end_x: i32) -> bool {
    let (new_start, new_end) = prefab_padded_x_range(start_x, end_x);

    for &(start, end) in &prefab_placed.x_ranges {
        if new_start <= end && new_end >= start {
            return true;
        }
    }

    false
}

fn record_prefab_attempt(prefab_placed: &mut PrefabPlaced, start_x: i32, end_x: i32) {
    let (start, end) = prefab_padded_x_range(start_x, end_x);
    prefab_placed.insert_range_merged(start, end);
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
    (x, y_seed): (i32, i32),
    bounds_y: (i32, i32),
) -> bool {
    let (min_y, max_y) = bounds_y;
    let Some((mask_min_x, mask_max_x, bottom)) = config.mask_bounds else {
        return false;
    };

    let Some(surface_air_y) = find_surface_air_y(chunk_world, x, y_seed, min_y, max_y) else {
        return false;
    };

    let footprint_start_x = x + mask_min_x;
    let footprint_end_x = x + mask_max_x;
    if prefab_x_blocked(prefab_placed, footprint_start_x, footprint_end_x) {
        return false;
    }

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

            if !atom.is_void() {
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

    if placed_any || queued_any {
        record_prefab_attempt(prefab_placed, footprint_start_x, footprint_end_x);
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
        find_surface_air_y(chunk_world, world_pos.x, world_pos.y, min_y, max_y)
            .unwrap_or(world_pos.y)
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

/// Stamp prefabs for a newly generated chunk at `chunk_pos`.
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

    let min_y = chunk_world.pos.y * CHUNK_LENGHT as i32;
    let max_y = (chunk_world.pos.y + LOAD_HEIGHT) * CHUNK_LENGHT as i32 - 1;
    let y_seed = (chunk_world.pos.y + LOAD_HEIGHT / 2) * CHUNK_LENGHT as i32;

    for config in &prefab_configs.0 {
        let xs = prefab_xs_for_chunk(config, chunk_pos.x, seed);
        for x in xs {
            let _ = stamp_prefab_at_x(
                chunk_world,
                dirty_rects,
                (prefab_placed, pending_prefab_stamps),
                materials,
                config,
                (x, y_seed),
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
