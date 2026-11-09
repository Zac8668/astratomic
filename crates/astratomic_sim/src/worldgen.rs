//! Stateless, per-chunk procedural world generation built on `fastnoise-lite` + `fastrand`.
//!
//! Every atom is a pure function of `(seed, global_x, global_y)` (plus a per-chunk seeded
//! [`fastrand::Rng`] for color noise and vein-edge dithering), so chunks can be generated in any
//! order, on any thread, with no cross-chunk caches.
//!
//! Coordinates are `ChunkWorld` global atom coordinates (`+Y` down). "Depth" is the number of atoms
//! below the local terrain surface.
//!
//! Layers, top to bottom:
//! - **Surface**: rolling fBm hills + ridged mountain ranges, with a thin grass cap.
//! - **Soil**: dirt with gravel pockets, then a dithered band of compacted dirt.
//! - **Rock**: with coal / iron / gold ore blobs that only appear past their minimum depths.
//! - **Caves**: domain-warped "noodle" tunnels (`|noise| < width`) plus large chambers. Both fade
//!   out near the surface, except at sparse surface entrances.

use std::sync::Arc;

use bones_framework::prelude::*;
use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use serde::{Deserialize, Serialize};

use crate::{
    atom::{Atom, Materials},
    chunk::{CHUNK_SIZE, ChunkGrid},
};

// Material IDs used by the generator (must match `assets/atoms.ron`).
const MAT_GRAVEL: u8 = 4;
const MAT_GRASS: u8 = 6;
const MAT_DIRT: u8 = 7;
const MAT_ROCK: u8 = 8;
const MAT_COMPACTED_DIRT: u8 = 10;
const MAT_COAL: u8 = 13;
const MAT_IRON: u8 = 14;
const MAT_GOLD: u8 = 15;

/// Reference feature size: `scale == 512.0` means "default sized" terrain and caves.
const REFERENCE_SCALE: f64 = 512.0;

/// Stateless procedural world generator configuration (loaded from `assets/gen.ron`).
///
/// Only plain parameters are stored here; noise samplers are rebuilt per chunk (which is trivially
/// cheap), so the resource stays `Send + Sync`, rollback-friendly, and order-independent.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[schema(opaque)]
#[serde(default)]
pub struct Generator {
    /// World seed.
    pub seed: u32,
    /// Feature size in atoms (`512` = default). Larger values stretch hills, caves, and veins.
    pub scale: f64,
    /// Average global atom Y of the terrain surface (`+Y` down, so negative is "up").
    pub surface_level: i32,
    /// Height in atoms of rolling hills above/below `surface_level`.
    pub hill_amplitude: f32,
    /// Extra height in atoms of ridged mountain ranges.
    pub mountain_amplitude: f32,
    /// Multiplier for tunnel width and chamber frequency (`0` = no caves, `1` = default).
    pub cave_density: f32,
    /// Depth in atoms over which caves fade in below the surface.
    pub cave_fade_depth: f32,
    /// Multiplier for ore vein thickness (`0` = no ores, `1` = default).
    pub ore_richness: f32,
}

impl Default for Generator {
    fn default() -> Self {
        Self::default_config()
    }
}

impl Generator {
    pub const fn default_config() -> Self {
        Self {
            seed: 8668,
            scale: REFERENCE_SCALE,
            surface_level: -256,
            hill_amplitude: 48.0,
            mountain_amplitude: 70.0,
            cave_density: 1.0,
            cave_fade_depth: 56.0,
            ore_richness: 1.0,
        }
    }

    /// Default configuration with a custom `seed` and feature `scale`.
    pub fn new(seed: u32, scale: f64) -> Self {
        Self {
            seed,
            scale,
            ..Self::default_config()
        }
    }

    /// Parse `gen.ron`. Accepts the new named-struct format, or the legacy
    /// `[scale, octaves, lacunarity, persistence, attenuation, seed]` array (only scale and
    /// seed are used from the legacy format).
    pub fn from_ron_bytes(bytes: &[u8]) -> Result<Self, ron::error::SpannedError> {
        match ron::de::from_bytes::<Self>(bytes) {
            Ok(generator) => Ok(generator),
            Err(err) => match ron::de::from_bytes::<[f64; 6]>(bytes) {
                Ok(legacy) => Ok(Self::new(legacy[5] as u32, legacy[0])),
                Err(_) => Err(err),
            },
        }
    }

    /// Generate the [`ChunkGrid`] for chunk `index`. Pure function of `(self, index)`.
    pub fn generate_grid(&self, index: IVec2, materials: &Materials) -> Arc<ChunkGrid> {
        let mut grid = ChunkGrid::new_zeroed_box();
        let noise = WorldNoise::new(self);
        let mut rng = fastrand::Rng::with_seed(chunk_seed(self.seed, index));

        let origin_x = index.x * CHUNK_SIZE as i32;
        let origin_y = index.y * CHUNK_SIZE as i32;

        // Per-column values (surface height, soil thickness, cave entrance strength).
        let mut columns = [ColumnInfo::default(); CHUNK_SIZE];
        for (lx, column) in columns.iter_mut().enumerate() {
            *column = noise.column((origin_x + lx as i32) as f32, self);
        }

        // Fast path: whole chunk above the highest surface in it -> pure air.
        let min_surface = columns
            .iter()
            .map(|c| c.surface_y)
            .fold(f32::INFINITY, f32::min);
        if ((origin_y + CHUNK_SIZE as i32) as f32) < min_surface {
            return Arc::from(grid);
        }

        for ly in 0..CHUNK_SIZE {
            let gy = (origin_y + ly as i32) as f32;
            for (lx, column) in columns.iter().enumerate() {
                let gx = (origin_x + lx as i32) as f32;
                let depth = gy - column.surface_y;
                if depth < 0.0 {
                    continue; // Sky
                }

                if noise.is_cave(gx, gy, depth, column, self) {
                    continue;
                }

                let id = noise.material(gx, gy, depth, column, self, &mut rng);
                grid.atoms[ly * CHUNK_SIZE + lx] = Atom::new_with_rng(id, materials, &mut rng);
            }
        }

        Arc::from(grid)
    }
}

#[derive(Clone, Copy, Default)]
struct ColumnInfo {
    /// Global atom Y of the first solid atom in this column.
    surface_y: f32,
    /// Thickness of the dirt layer.
    dirt_depth: f32,
    /// Thickness of the compacted dirt layer below the dirt.
    compact_depth: f32,
    /// Extra "virtual depth" granted to caves so tunnels can break through the surface here.
    entrance_bonus: f32,
}

/// Noise samplers for a single chunk generation pass.
struct WorldNoise {
    hills: FastNoiseLite,
    mountains: FastNoiseLite,
    mountain_mask: FastNoiseLite,
    surface_detail: FastNoiseLite,
    soil: FastNoiseLite,
    entrances: FastNoiseLite,
    warp: FastNoiseLite,
    noodle: FastNoiseLite,
    noodle_width: FastNoiseLite,
    chamber: FastNoiseLite,
    pockets: FastNoiseLite,
    coal_blob: FastNoiseLite,
    coal_cluster: FastNoiseLite,
    iron_blob: FastNoiseLite,
    iron_cluster: FastNoiseLite,
    gold_blob: FastNoiseLite,
    gold_cluster: FastNoiseLite,
}

fn make_noise(
    seed: u32,
    layer: i32,
    freq: f32,
    fractal: FractalType,
    octaves: i32,
) -> FastNoiseLite {
    let mut n = FastNoiseLite::with_seed((seed as i32).wrapping_add(layer.wrapping_mul(7919)));
    n.set_noise_type(Some(NoiseType::OpenSimplex2));
    n.set_frequency(Some(freq));
    n.set_fractal_type(Some(fractal));
    if fractal != FractalType::None {
        n.set_fractal_octaves(Some(octaves));
        n.set_fractal_lacunarity(Some(2.0));
        n.set_fractal_gain(Some(0.5));
    }
    n
}

impl WorldNoise {
    fn new(cfg: &Generator) -> Self {
        let scale = if cfg.scale > 1.0 {
            cfg.scale
        } else {
            REFERENCE_SCALE
        };
        // Frequency multiplier: bigger scale -> lower frequency -> larger features.
        let f = (REFERENCE_SCALE / scale) as f32;
        let s = cfg.seed;

        let mut warp = make_noise(s, 6, 0.0045 * f, FractalType::None, 1);
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_domain_warp_amp(Some(38.0 / f));

        Self {
            hills: make_noise(s, 0, 0.0022 * f, FractalType::FBm, 4),
            mountains: make_noise(s, 1, 0.0016 * f, FractalType::Ridged, 3),
            mountain_mask: make_noise(s, 2, 0.0006 * f, FractalType::None, 1),
            surface_detail: make_noise(s, 3, 0.03 * f, FractalType::None, 1),
            soil: make_noise(s, 4, 0.012 * f, FractalType::FBm, 2),
            entrances: make_noise(s, 5, 0.0035 * f, FractalType::None, 1),
            warp,
            noodle: make_noise(s, 7, 0.0055 * f, FractalType::FBm, 2),
            noodle_width: make_noise(s, 8, 0.008 * f, FractalType::None, 1),
            chamber: make_noise(s, 9, 0.0042 * f, FractalType::FBm, 3),
            pockets: make_noise(s, 10, 0.02 * f, FractalType::FBm, 2),
            coal_blob: make_noise(s, 11, 0.038 * f, FractalType::FBm, 2),
            coal_cluster: make_noise(s, 12, 0.009 * f, FractalType::None, 1),
            iron_blob: make_noise(s, 13, 0.048 * f, FractalType::FBm, 2),
            iron_cluster: make_noise(s, 14, 0.011 * f, FractalType::None, 1),
            gold_blob: make_noise(s, 15, 0.060 * f, FractalType::FBm, 2),
            gold_cluster: make_noise(s, 16, 0.013 * f, FractalType::None, 1),
        }
    }

    fn column(&self, x: f32, cfg: &Generator) -> ColumnInfo {
        let hills = self.hills.get_noise_2d(x, 0.0) * cfg.hill_amplitude;
        // Mountains only rise where the very-low-frequency mask is positive.
        let mask = smoothstep(0.05, 0.45, self.mountain_mask.get_noise_2d(x, 0.0));
        let mountains = (self.mountains.get_noise_2d(x, 0.0) * 0.5 + 0.5) * cfg.mountain_amplitude * mask;
        let detail = self.surface_detail.get_noise_2d(x, 0.0) * 2.0;
        let height = hills + mountains + detail;

        let soil = self.soil.get_noise_2d(x, 500.0);
        let entrance = self.entrances.get_noise_2d(x, 0.0);

        ColumnInfo {
            // `+Y` is down, so terrain height subtracts from the surface level.
            surface_y: (cfg.surface_level as f32 - height).floor(),
            // Mountains have thinner soil.
            dirt_depth: (9.0 + soil * 4.0) * (1.0 - 0.6 * mask),
            compact_depth: (20.0 + soil * 6.0) * (1.0 - 0.5 * mask),
            entrance_bonus: smoothstep(0.62, 0.9, entrance) * cfg.cave_fade_depth,
        }
    }

    fn is_cave(&self, x: f32, y: f32, depth: f32, col: &ColumnInfo, cfg: &Generator) -> bool {
        if cfg.cave_density <= 0.0 {
            return false;
        }

        // Surface depth-fading: 0 at the surface, 1 once `cave_fade_depth` deep.
        let fade = smoothstep(0.0, cfg.cave_fade_depth.max(1.0), depth + col.entrance_bonus);
        if fade <= 0.0 {
            return false;
        }

        let (wx, wy) = self.warp.domain_warp_2d(x, y);

        // Noodle tunnels: the zero-isoline of warped noise forms long, connected, winding paths.
        let width = (0.04 + 0.03 * self.noodle_width.get_noise_2d(x, y)).max(0.0)
            * fade
            * cfg.cave_density;
        if self.noodle.get_noise_2d(wx, wy).abs() < width {
            return true;
        }

        // Large chambers: high plateaus of low-frequency noise, more common with depth.
        let depth01 = (depth / 900.0).clamp(0.0, 1.0);
        let threshold = lerp(0.6, 0.4, depth01) + (1.0 - fade) * 0.5
            + (1.0 - cfg.cave_density.min(2.0)) * 0.25;
        self.chamber.get_noise_2d(wx * 0.9, wy * 1.25) > threshold
    }

    fn material(
        &self,
        x: f32,
        y: f32,
        depth: f32,
        col: &ColumnInfo,
        cfg: &Generator,
        rng: &mut fastrand::Rng,
    ) -> u8 {
        if depth < 2.0 {
            return MAT_GRASS;
        }

        if depth < col.dirt_depth {
            return if depth > 4.0 && self.pockets.get_noise_2d(x, y) > 0.62 {
                MAT_GRAVEL
            } else {
                MAT_DIRT
            };
        }

        // Dithered transition from compacted dirt into rock.
        let rock_start = col.dirt_depth + col.compact_depth;
        let transition = 6.0;
        if depth < rock_start + transition {
            let t = ((depth - rock_start) / transition).clamp(0.0, 1.0);
            if depth < rock_start || rng.f32() > t {
                return MAT_COMPACTED_DIRT;
            }
        }

        // Rock layer with rare gravel pockets.
        if depth > 60.0 && self.pockets.get_noise_2d(x * 0.7, y * 0.7 + 3000.0) > 0.72 {
            return MAT_GRAVEL;
        }

        if cfg.ore_richness > 0.0 {
            // Ore blobs: 2D smooth noise peaks modulated by cluster noise form isolated, rounded deposits.
            let richness_offset = (cfg.ore_richness - 1.0) * 0.05;

            let is_in_blob = |blob_noise: &FastNoiseLite,
                              cluster_noise: &FastNoiseLite,
                              min_depth: f32,
                              base_threshold: f32,
                              rng: &mut fastrand::Rng|
             -> bool {
                if depth < min_depth {
                    return false;
                }
                let cluster = cluster_noise.get_noise_2d(x, y);
                let depth_boost = smoothstep(min_depth, min_depth + 90.0, depth) * 0.04;
                // Cluster centers lower the threshold, letting the 2D noise peak expand into a blob
                let threshold = base_threshold - (cluster * 0.22) - depth_boost - richness_offset;

                let shape = blob_noise.get_noise_2d(x, y);
                let margin = shape - threshold;
                if margin > 0.035 {
                    true
                } else if margin > 0.0 {
                    rng.f32() < (margin / 0.035)
                } else {
                    false
                }
            };

            // Check rarest to most common
            if is_in_blob(&self.gold_blob, &self.gold_cluster, 240.0, 0.67, rng) {
                return MAT_GOLD;
            }
            if is_in_blob(&self.iron_blob, &self.iron_cluster, 80.0, 0.60, rng) {
                return MAT_IRON;
            }
            if is_in_blob(&self.coal_blob, &self.coal_cluster, 24.0, 0.55, rng) {
                return MAT_COAL;
            }
        }

        MAT_ROCK
    }
}

/// Deterministic per-chunk RNG seed (SplitMix64 finalizer over seed + chunk coords).
fn chunk_seed(seed: u32, index: IVec2) -> u64 {
    let mut z = (seed as u64)
        ^ ((index.x as u32 as u64) << 32)
        ^ (index.y as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_and_layered() {
        let materials = Materials::default_builtin();
        let generator = Generator::default_config();

        // Same input -> identical output (color noise included).
        let a = generator.generate_grid(ivec2(3, -4), &materials);
        let b = generator.generate_grid(ivec2(3, -4), &materials);
        assert_eq!(*a, *b);

        // High in the sky is empty.
        let sky = generator.generate_grid(ivec2(0, -20), &materials);
        assert!(sky.atoms.iter().all(|a| a.is_void()));

        // Deep underground is mostly solid (caves are a minority) and contains rock.
        let deep = generator.generate_grid(ivec2(0, 8), &materials);
        let solid = deep.atoms.iter().filter(|a| !a.is_void()).count();
        assert!(solid > CHUNK_SIZE * CHUNK_SIZE / 3, "solid = {solid}");
        assert!(deep.atoms.iter().any(|a| a.material_id == MAT_ROCK));
    }

    #[test]
    fn legacy_gen_ron_still_parses() {
        let legacy = b"(512.0, 8., 2.0, 0.8, 2.0, 1234.)";
        let generator = Generator::from_ron_bytes(legacy).unwrap();
        assert_eq!(generator.seed, 1234);
        assert_eq!(generator.scale, 512.0);
    }

    #[test]
    fn ores_generate_in_blobs() {
        let materials = Materials::default_builtin();
        let generator = Generator::default_config();

        // Search underground chunks for an ore deposit
        let mut found_blob = false;
        'search: for cy in 0..12 {
            for cx in -4..4 {
                let grid = generator.generate_grid(ivec2(cx, cy), &materials);
                // Check for coal, iron, or gold
                let mut visited = [false; CHUNK_SIZE * CHUNK_SIZE];
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let idx = y * CHUNK_SIZE + x;
                        let mat = grid.atoms[idx].material_id;
                        if (mat == MAT_COAL || mat == MAT_IRON || mat == MAT_GOLD) && !visited[idx] {
                            // Flood-fill connected ore component
                            let mut queue = vec![(x, y)];
                            visited[idx] = true;
                            let mut min_x = x;
                            let mut max_x = x;
                            let mut min_y = y;
                            let mut max_y = y;
                            let mut count = 0;

                            while let Some((qx, qy)) = queue.pop() {
                                count += 1;
                                min_x = min_x.min(qx);
                                max_x = max_x.max(qx);
                                min_y = min_y.min(qy);
                                max_y = max_y.max(qy);

                                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                                    let nx = qx as i32 + dx;
                                    let ny = qy as i32 + dy;
                                    if nx >= 0 && nx < CHUNK_SIZE as i32 && ny >= 0 && ny < CHUNK_SIZE as i32 {
                                        let nidx = ny as usize * CHUNK_SIZE + nx as usize;
                                        if !visited[nidx] && grid.atoms[nidx].material_id == mat {
                                            visited[nidx] = true;
                                            queue.push((nx as usize, ny as usize));
                                        }
                                    }
                                }
                            }

                            let width = max_x - min_x + 1;
                            let height = max_y - min_y + 1;
                            // A blob has 2D substance: both width and height >= 3, and reasonable compactness (not a 1px line)
                            if count >= 8 && width >= 3 && height >= 3 {
                                found_blob = true;
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        assert!(found_blob, "expected to find ore blobs with 2D width and height >= 3");
    }
}
