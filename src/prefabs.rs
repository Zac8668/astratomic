use crate::prelude::*;
use bevy::asset::Handle;
use bevy::prelude::Image;
use std::fs;

pub fn load_prefabs(mut prefab_configs: ResMut<PrefabConfigs>, asset_server: Res<AssetServer>) {
    //Get prefabs to spawn
    let prefabs_dir = "assets/prefabs";
    if let Ok(entries) = fs::read_dir(prefabs_dir) {
        for entry in entries.flatten() {
            let entry: std::fs::DirEntry = entry;
            let path = entry.path();
            if path.is_dir() {
                let config_path = path.join("spawn_config.ron");
                if config_path.exists() {
                    if let Ok(bytes) = fs::read(&config_path) {
                        if let Ok(spawn_config) = ron::de::from_bytes::<PrefabSpawnConfig>(&bytes) {
                            // Remove "assets/" prefix for asset_server.load
                            let rel_path = path.strip_prefix("assets/").unwrap_or(&path);
                            let name = rel_path.to_string_lossy().to_string();
                            let image_path = rel_path.join("image.png");
                            let mask_path = rel_path.join("mask.png");
                            let image_handle = asset_server.load(image_path.to_str().unwrap());
                            let mask_handle = asset_server.load(mask_path.to_str().unwrap());
                            prefab_configs.0.push(PrefabConfig {
                                name,
                                spawn_config,
                                positions: Vec::<i32>::new(),
                                image: image_handle,
                                mask: mask_handle,
                            });
                        }
                    }
                }
            }
        }
    }

    prefab_configs.0.sort_by(|a, b| a.name.cmp(&b.name));
}

pub fn get_prefabs_in_chunk(
    slot_width: i32,
    slot_chance: f32,
    chunk_start_x: i32,
    seed: u32,
) -> Vec<i32> {
    let chunk_end_x = chunk_start_x + CHUNK_LENGHT as i32;
    let mut positions = Vec::new();

    let start_slot = (chunk_start_x as f32 / slot_width as f32).floor() as i32;
    let end_slot = (((chunk_end_x - 1) as f32) / slot_width as f32).floor() as i32;

    for slot_idx in start_slot..=end_slot {
        if get_random_at(seed, slot_idx) > slot_chance {
            continue;
        }

        let offset_rng = get_random_at(seed + 1, slot_idx);
        let offset = (offset_rng * slot_width as f32) as i32;
        let prefab_x = (slot_idx * slot_width) + offset;

        positions.push(prefab_x);
    }
    positions
}

/// Returns deterministic prefab X positions that fall within the given chunk.
///
/// This centralizes the salting + sorting + in-bounds filtering so both
/// `manager_setup` and `move_manager` stay consistent.
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

/// Deterministic, stable hash used to salt prefab RNG.
///
/// We intentionally avoid `DefaultHasher` because it is randomized per-process.
pub fn prefab_seed_salt(name: &str) -> u32 {
    // 32-bit FNV-1a
    let mut hash: u32 = 0x811C9DC5;
    for &b in name.as_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

// -----------------------------------------------------------------------------
// Prefab placement & stamping
// -----------------------------------------------------------------------------

#[derive(Resource, Default, Clone)]
pub struct PrefabPlaced {
    pub x_ranges: Vec<(i32, i32)>,
}

impl PrefabPlaced {
    fn insert_range_merged(&mut self, start: i32, end: i32) {
        let (start, end) = if start <= end { (start, end) } else { (end, start) };

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

#[derive(Clone, Copy)]
pub struct PendingStampCell {
    pub local: UVec2,
    pub id: u8,
    pub color: [u8; 4],
}

#[derive(Resource, Default)]
pub struct PendingPrefabStamps(pub HashMap<IVec2, Vec<PendingStampCell>>);

pub fn prefab_images<'a>(
    images: &'a Assets<Image>,
    config: &PrefabConfig,
) -> Option<(&'a Image, &'a Image)> {
    Some((images.get(&config.image)?, images.get(&config.mask)?))
}

fn prefab_mask_bounds(mask: &Image) -> Option<(i32, i32, i32)> {
    // Returns (min_x, max_x, bottom_y) for non-empty pixels in the mask.
    let w = mask.width() as usize;
    let h = mask.height() as usize;
    if w == 0 || h == 0 {
        return None;
    }
    if mask.data.len() < w * h * 4 {
        return None;
    }

    let mut min_x: i32 = i32::MAX;
    let mut max_x: i32 = i32::MIN;
    let mut bottom_y: i32 = i32::MIN;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let id = mask.data[idx];
            let alpha = mask.data[idx + 3];
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

fn find_surface_air_y(
    chunk_manager: &ChunkManager,
    x: i32,
    start_y: i32,
    min_y: i32,
    max_y: i32,
) -> Option<i32> {
    if min_y > max_y {
        return None;
    }

    let mut y = start_y.clamp(min_y, max_y);
    let start_atom = chunk_manager.get_atom(&global_to_chunk(IVec2::new(x, y)))?;

    const MAX_STEPS: i32 = 4096;
    let mut steps = 0;

    if !start_atom.is_void() {
        // We are inside ground; go up to find the first air cell.
        while y >= min_y && steps < MAX_STEPS {
            let atom = chunk_manager.get_atom(&global_to_chunk(IVec2::new(x, y)))?;
            if atom.is_void() {
                return Some(y);
            }
            y -= 1;
            steps += 1;
        }
    } else {
        // We are in air; go down until we hit ground, then the surface air cell is one above.
        while y <= max_y && steps < MAX_STEPS {
            let atom = chunk_manager.get_atom(&global_to_chunk(IVec2::new(x, y)))?;
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
        // inclusive ranges overlap
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
    chunk_manager: &mut ChunkManager,
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
        let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) else {
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
    chunk_manager: &mut ChunkManager,
    dirty_rects: &mut DirtyRects,
    prefab_placed: &mut PrefabPlaced,
    pending: &mut PendingPrefabStamps,
    materials: &Materials,
    prefab_image: &Image,
    prefab_mask: &Image,
    x: i32,
    y_seed: i32,
    bounds_y: (i32, i32),
) -> bool {
    let (min_y, max_y) = bounds_y;
    let Some((mask_min_x, mask_max_x, bottom)) = prefab_mask_bounds(prefab_mask) else {
        return false;
    };

    let Some(surface_air_y) = find_surface_air_y(chunk_manager, x, y_seed, min_y, max_y) else {
        return false;
    };

    // Block by the actual prefab footprint (mask content) + 2px padding.
    let footprint_start_x = x + mask_min_x;
    let footprint_end_x = x + mask_max_x;
    if prefab_x_blocked(prefab_placed, footprint_start_x, footprint_end_x) {
        return false;
    }

    let w = prefab_mask.width() as usize;
    let h = prefab_mask.height() as usize;
    if w == 0 || h == 0 {
        return false;
    }
    if prefab_mask.data.len() < w * h * 4 || prefab_image.data.len() < w * h * 4 {
        return false;
    }

    // Align the lowest non-empty mask pixel row with the surface air cell.
    let origin = IVec2::new(x, surface_air_y - bottom);

    let mut placed_any = false;
    let mut queued_any = false;

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 4;

            let id = prefab_mask.data[idx];
            let alpha = prefab_mask.data[idx + 3];
            if alpha == 0 || id == 0 {
                continue;
            }
            // Keep within known atom IDs (Atom::new panics outside this range in this codebase).
            if !(2..=12).contains(&id) {
                continue;
            }

            let color = [
                prefab_image.data[idx],
                prefab_image.data[idx + 1],
                prefab_image.data[idx + 2],
                prefab_image.data[idx + 3],
            ];

            let world_pos = origin + IVec2::new(px as i32, py as i32);
            let chunk_pos = global_to_chunk(world_pos);
            let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) else {
                queue_pending_stamp(pending, world_pos, id, color);
                queued_any = true;
                continue;
            };

            // Avoid overwriting terrain / existing content.
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

    // If we placed anything (or queued pixels for future chunks), reserve this X range.
    if placed_any || queued_any {
        record_prefab_attempt(prefab_placed, footprint_start_x, footprint_end_x);
    }

    placed_any
}

fn get_random_at(seed: u32, x: i32) -> f32 {
    use std::num::Wrapping;

    let mut n = Wrapping(x as u32);
    n = n * Wrapping(0xB5297A4D);
    n = n + Wrapping(seed);
    n = n ^ (n >> 8);
    n = n + Wrapping(0x68E31DA4);
    n = n ^ (n << 8);
    n = n * Wrapping(0x1B56C4E9);
    n = n ^ (n >> 8);

    // Convert huge integer to float 0.0 - 1.0
    (n.0 as f32) / (u32::MAX as f32)
}

#[derive(Resource, Default)]
pub struct PrefabConfigs(pub Vec<PrefabConfig>);

pub struct PrefabConfig {
    pub name: String,
    pub spawn_config: PrefabSpawnConfig,
    pub positions: Vec<i32>,
    pub image: Handle<Image>,
    pub mask: Handle<Image>,
}

#[derive(Debug, serde::Deserialize)]
pub struct PrefabSpawnConfig {
    pub slot_width: i32,
    pub slot_chance: f32,
}

pub struct PrefabsPlugin;
impl Plugin for PrefabsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PrefabConfigs>()
            .add_systems(OnEnter(GameState::Loading), load_prefabs);
    }
}
