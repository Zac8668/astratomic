use bones_schema::HasSchema;
use serde::{Deserialize, Serialize};

/// Compact 8-byte `Pod` atom representation storing a baked RGBA8 color and a `u8` material ID.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Default,
    bytemuck::Pod,
    bytemuck::Zeroable,
    HasSchema,
    Serialize,
    Deserialize,
)]
#[schema(opaque)]
#[repr(C)]
pub struct Atom {
    pub color: [u8; 4],
    pub material_id: u8,
    pub flags: u8,
    pub vx: i8,
    pub vy: i8,
}

const _: [(); 8] = [(); std::mem::size_of::<Atom>()];

#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Debug, HasSchema)]
#[repr(u8)]
pub enum AtomState {
    #[default]
    Void = 0,
    Solid = 1,
    Object = 2,
    Powder = 3,
    Liquid = 4,
    Gas = 5,
}

impl AtomState {
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & Atom::STATE_MASK {
            1 => Self::Solid,
            2 => Self::Object,
            3 => Self::Powder,
            4 => Self::Liquid,
            5 => Self::Gas,
            _ => Self::Void,
        }
    }

    #[inline]
    pub const fn default_for_material_id(material_id: u8) -> Self {
        match material_id {
            0 => Self::Void,
            1 => Self::Object,
            2 | 4 => Self::Powder,
            3 | 5 | 9 => Self::Liquid,
            6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15 => Self::Solid,
            _ => Self::Solid,
        }
    }

    #[inline]
    pub const fn is_fluid(&self) -> bool {
        matches!(self, Self::Liquid | Self::Gas)
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone, Copy, HasSchema)]
#[repr(C)]
pub struct Material {
    #[serde(default)]
    pub inertial_resistance: f32,
    #[serde(default)]
    pub flow: u8,
    #[serde(default)]
    pub damage: f32,
    #[serde(default)]
    pub default_state: AtomState,
    #[serde(default)]
    pub weight: f32,
}

/// Custom [`bones_asset::AssetLoader`] for loading [`Materials`] from `.ron` / `atoms.ron` asset files.
pub struct MaterialsAssetLoader;

impl bones_asset::AssetLoader for MaterialsAssetLoader {
    fn load(
        &self,
        _ctx: bones_asset::AssetLoadCtx,
        bytes: &[u8],
    ) -> bones_asset::BoxedFuture<bones_asset::anyhow::Result<bones_asset::SchemaBox>> {
        let bytes = bytes.to_vec();
        Box::pin(async move {
            let materials = Materials::from_ron_bytes(&bytes)?;
            Ok(bones_asset::SchemaBox::new(materials))
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, HasSchema)]
#[type_data(bones_asset::asset_loader(["atoms.ron", "ron"], MaterialsAssetLoader))]
pub struct Materials(pub Vec<Material>);

impl Materials {
    pub fn from_ron_bytes(bytes: &[u8]) -> Result<Self, ron::error::SpannedError> {
        ron::de::from_bytes(bytes)
    }

    pub fn from_ron_str(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::de::from_str(s)
    }

    pub fn get_from_atom(&self, atom: &Atom) -> &Material {
        &self.0[atom.material_id as usize]
    }

    pub fn get_from_id(&self, id: u8) -> &Material {
        &self.0[id as usize]
    }

    pub fn default_builtin() -> Self {
        Self(vec![
            // 0: Void
            Material {
                default_state: AtomState::Void,
                weight: 0.0,
                ..Default::default()
            },
            // 1: Object
            Material {
                default_state: AtomState::Object,
                weight: 1.0,
                ..Default::default()
            },
            // 2: Sand
            Material {
                inertial_resistance: 0.1,
                default_state: AtomState::Powder,
                weight: 1.5,
                ..Default::default()
            },
            // 3: Water
            Material {
                flow: 5,
                damage: 0.0,
                default_state: AtomState::Liquid,
                weight: 1.0,
                ..Default::default()
            },
            // 4: Gravel
            Material {
                inertial_resistance: 0.60,
                default_state: AtomState::Powder,
                weight: 2.0,
                ..Default::default()
            },
            // 5: Lava
            Material {
                flow: 1,
                damage: 15.0,
                default_state: AtomState::Liquid,
                weight: 2.2,
                ..Default::default()
            },
            // 6: Grass
            Material {
                default_state: AtomState::Solid,
                weight: 0.5,
                ..Default::default()
            },
            // 7: Dirt
            Material {
                default_state: AtomState::Solid,
                weight: 1.3,
                ..Default::default()
            },
            // 8: Rock
            Material {
                default_state: AtomState::Solid,
                weight: 2.5,
                ..Default::default()
            },
            // 9: Acid
            Material {
                flow: 5,
                damage: 3.0,
                default_state: AtomState::Liquid,
                weight: 1.1,
                ..Default::default()
            },
            // 10: Compacted Dirt
            Material {
                default_state: AtomState::Solid,
                weight: 1.6,
                ..Default::default()
            },
            // 11: Leaves
            Material {
                default_state: AtomState::Solid,
                weight: 0.3,
                ..Default::default()
            },
            // 12: Wood
            Material {
                default_state: AtomState::Solid,
                weight: 0.5,
                ..Default::default()
            },
            // 13: Coal
            Material {
                default_state: AtomState::Solid,
                weight: 1.3,
                ..Default::default()
            },
            // 14: Iron Ore
            Material {
                default_state: AtomState::Solid,
                weight: 4.0,
                ..Default::default()
            },
            // 15: Gold Ore
            Material {
                default_state: AtomState::Solid,
                weight: 10.0,
                ..Default::default()
            },
        ])
    }
}

impl std::ops::Index<&Atom> for Materials {
    type Output = Material;
    #[track_caller]
    fn index(&self, atom: &Atom) -> &Self::Output {
        self.get_from_atom(atom)
    }
}

impl std::ops::Index<u8> for Materials {
    type Output = Material;
    #[track_caller]
    fn index(&self, id: u8) -> &Self::Output {
        self.get_from_id(id)
    }
}

impl Atom {
    /// Low 3 bits of `flags` encode [`AtomState`].
    pub const STATE_MASK: u8 = 0b0000_0111;
    /// Bit 3 of `flags` indicates whether a powder atom is actively free-falling.
    pub const FLAG_FREE_FALLING: u8 = 0b0000_1000;
    /// Alias for backwards compatibility with liquid/powder movement checks.
    pub const FLAG_MOVING: u8 = Self::FLAG_FREE_FALLING;
    /// Bits 4..6 of `flags` store `f_idle` (0..=3).
    pub const F_IDLE_SHIFT: u8 = 4;
    pub const F_IDLE_MASK: u8 = 0b0011_0000;
    /// Bits 6..8 of `flags` store `updated_at` parity (`dt & 0b11`).
    pub const UPDATED_AT_SHIFT: u8 = 6;
    pub const UPDATED_AT_MASK: u8 = 0b1100_0000;
    /// Bits 3..=7 of `flags` store 5-bit particle lifetime (0 = real matter, >0 = ephemeral FX).
    pub const LIFE_5BIT_SHIFT: u8 = 3;
    pub const LIFE_5BIT_MASK: u8 = 0b1111_1000;

    pub const VOID: Self = Self {
        color: [0, 0, 0, 0],
        material_id: 0,
        flags: 0,
        vx: 0,
        vy: 0,
    };
    pub const AIR: Self = Self::VOID;

    #[inline]
    pub const fn object() -> Self {
        Self {
            color: [0, 0, 0, 0],
            material_id: 1,
            flags: AtomState::Object as u8,
            vx: 0,
            vy: 0,
        }
    }

    #[inline]
    pub const fn with_state(material_id: u8, state: AtomState, color: [u8; 4]) -> Self {
        let flags = if (state as u8) == (AtomState::Powder as u8) {
            (state as u8) | Self::FLAG_FREE_FALLING
        } else {
            state as u8
        };
        Self {
            color,
            material_id,
            flags,
            vx: 0,
            vy: 0,
        }
    }

    /// Create a new atom of material `id` with randomized color noise from the global RNG.
    pub fn new(id: u8, materials: &Materials) -> Self {
        let mut rng = fastrand::Rng::with_seed(fastrand::u64(..));
        Self::new_with_rng(id, materials, &mut rng)
    }

    /// Create a new atom of material `id`, drawing color noise from `rng` so procedural
    /// generation with a seeded [`fastrand::Rng`] is fully deterministic.
    pub fn new_with_rng(id: u8, materials: &Materials, rng: &mut fastrand::Rng) -> Self {
        if id == 0 {
            return Self::VOID;
        }
        if id == 1 {
            return Self::object();
        }

        let state = materials
            .0
            .get(id as usize)
            .map(|m| m.default_state)
            .unwrap_or_else(|| AtomState::default_for_material_id(id));

        let color = match id {
            2 => [
                // Sand
                (230 + rng.i16(-20..20)) as u8,
                (197 + rng.i16(-20..20)) as u8,
                (92 + rng.i16(-20..20)) as u8,
                255,
            ],
            3 => [
                // Water
                (20 + rng.i16(-20..20)) as u8,
                (125 + rng.i16(-20..20)) as u8,
                (204 + rng.i16(-20..20)) as u8,
                150,
            ],
            4 => [
                // Gravel
                (110 + rng.i16(-12..12)) as u8,
                (110 + rng.i16(-12..12)) as u8,
                (110 + rng.i16(-12..12)) as u8,
                255,
            ],
            5 => [
                // Lava
                (245 + rng.i16(-10..10)) as u8,
                (140 + rng.i16(-20..20)) as u8,
                (10 + rng.i16(-10..10)) as u8,
                255,
            ],
            6 => [
                // Grass
                (40 + rng.i16(-10..10)) as u8,
                (165 + rng.i16(-10..10)) as u8,
                (49 + rng.i16(-5..5)) as u8,
                255,
            ],
            7 => [
                // Dirt
                (120 + rng.i16(-10..10)) as u8,
                (70 + rng.i16(-10..10)) as u8,
                (40 + rng.i16(-5..5)) as u8,
                255,
            ],
            8 => [
                // Rock
                (80 + rng.i16(-10..10)) as u8,
                (80 + rng.i16(-10..10)) as u8,
                (80 + rng.i16(-5..5)) as u8,
                255,
            ],
            9 => [
                // Acid
                (10 + rng.i16(-10..10)) as u8,
                (230 + rng.i16(-10..10)) as u8,
                (40 + rng.i16(-5..5)) as u8,
                255,
            ],
            10 => [
                // Compacted Dirt
                (100 + rng.i16(-10..10)) as u8,
                (50 + rng.i16(-10..10)) as u8,
                (30 + rng.i16(-5..5)) as u8,
                255,
            ],
            11 => [
                // Leaves
                (60 + rng.i16(-15..15)) as u8,
                (180 + rng.i16(-20..20)) as u8,
                (60 + rng.i16(-10..10)) as u8,
                220,
            ],
            12 => [
                // Wood
                (120 + rng.i16(-10..10)) as u8,
                (80 + rng.i16(-10..10)) as u8,
                (40 + rng.i16(-5..5)) as u8,
                255,
            ],
            13 => [
                // Coal
                (34 + rng.i16(-6..6)) as u8,
                (32 + rng.i16(-6..6)) as u8,
                (36 + rng.i16(-6..6)) as u8,
                255,
            ],
            14 => [
                // Iron Ore
                (168 + rng.i16(-14..14)) as u8,
                (104 + rng.i16(-12..12)) as u8,
                (78 + rng.i16(-10..10)) as u8,
                255,
            ],
            15 => [
                // Gold Ore
                (238 + rng.i16(-12..12)) as u8,
                (190 + rng.i16(-16..16)) as u8,
                (52 + rng.i16(-12..12)) as u8,
                255,
            ],
            _ => panic!("Atom not found, invalid ID. {id}"),
        };

        Self::with_state(id, state, color)
    }

    pub const PAINTABLE_MATERIAL_IDS: [u8; 15] =
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    #[inline]
    pub const fn material_name(id: u8) -> &'static str {
        match id {
            0 => "Void (Erase)",
            1 => "Object",
            2 => "Sand",
            3 => "Water",
            4 => "Gravel",
            5 => "Lava",
            6 => "Grass",
            7 => "Dirt",
            8 => "Rock",
            9 => "Acid",
            10 => "Compacted Dirt",
            11 => "Leaves",
            12 => "Wood",
            13 => "Coal",
            14 => "Iron Ore",
            15 => "Gold Ore",
            _ => "Unknown",
        }
    }

    #[inline]
    pub const fn material_preview_color(id: u8) -> [u8; 4] {
        match id {
            0 => [30, 32, 42, 255],
            1 => [255, 255, 255, 255],
            2 => [230, 197, 92, 255],
            3 => [20, 125, 204, 200],
            4 => [110, 110, 110, 255],
            5 => [245, 140, 10, 255],
            6 => [40, 165, 49, 255],
            7 => [120, 70, 40, 255],
            8 => [80, 80, 80, 255],
            9 => [10, 230, 40, 255],
            10 => [100, 50, 30, 255],
            11 => [60, 180, 60, 220],
            12 => [120, 80, 40, 255],
            13 => [34, 32, 36, 255],
            14 => [168, 104, 78, 255],
            15 => [238, 190, 52, 255],
            _ => [255, 0, 255, 255],
        }
    }

    #[inline]
    pub const fn color(&self) -> [u8; 4] {
        self.color
    }

    #[inline]
    pub const fn state(&self) -> AtomState {
        let bits = self.flags & Self::STATE_MASK;
        if bits != 0 {
            AtomState::from_bits(bits)
        } else {
            AtomState::default_for_material_id(self.material_id)
        }
    }

    #[inline]
    pub fn set_state(&mut self, state: AtomState) {
        let base_flags = (self.flags & !Self::STATE_MASK) | (state as u8);
        self.flags = if matches!(state, AtomState::Powder) {
            base_flags | Self::FLAG_FREE_FALLING
        } else {
            base_flags
        };
    }

    #[inline]
    pub const fn is_free_falling(&self) -> bool {
        (self.flags & Self::FLAG_FREE_FALLING) != 0
    }

    #[inline]
    pub const fn free_falling(&self) -> bool {
        self.is_free_falling()
    }

    #[inline]
    pub fn set_free_falling(&mut self, free_falling: bool) {
        if free_falling {
            self.flags |= Self::FLAG_FREE_FALLING;
        } else {
            self.flags &= !Self::FLAG_FREE_FALLING;
        }
    }

    #[inline]
    pub const fn moving(&self) -> bool {
        self.is_free_falling()
    }

    #[inline]
    pub fn set_moving(&mut self, moving: bool) {
        self.set_free_falling(moving);
    }

    #[inline]
    pub const fn f_idle(&self) -> u8 {
        (self.flags & Self::F_IDLE_MASK) >> Self::F_IDLE_SHIFT
    }

    #[inline]
    pub fn set_f_idle(&mut self, f_idle: u8) {
        let clamped = f_idle & 0b11;
        self.flags = (self.flags & !Self::F_IDLE_MASK) | (clamped << Self::F_IDLE_SHIFT);
    }

    #[inline]
    pub const fn updated_at(&self) -> u8 {
        (self.flags & Self::UPDATED_AT_MASK) >> Self::UPDATED_AT_SHIFT
    }

    #[inline]
    pub fn set_updated_at(&mut self, updated_at: u8) {
        let masked = updated_at & 0b11;
        self.flags = (self.flags & !Self::UPDATED_AT_MASK) | (masked << Self::UPDATED_AT_SHIFT);
    }

    #[inline]
    pub const fn is_liquid(&self) -> bool {
        matches!(self.state(), AtomState::Liquid)
    }

    #[inline]
    pub const fn is_void(&self) -> bool {
        self.material_id == 0
    }

    #[inline]
    pub const fn is_air(&self) -> bool {
        self.is_void()
    }

    #[inline]
    pub const fn life_5bit(&self) -> u8 {
        (self.flags & Self::LIFE_5BIT_MASK) >> Self::LIFE_5BIT_SHIFT
    }

    #[inline]
    pub fn set_life_5bit(&mut self, life: u8) {
        let clamped = life.min(31);
        self.flags = (self.flags & !Self::LIFE_5BIT_MASK) | (clamped << Self::LIFE_5BIT_SHIFT);
    }

    #[inline]
    pub fn mark_updated(&mut self, current_tick: u64) {
        self.set_updated_at((current_tick & 0b11) as u8);
    }

    #[inline]
    pub const fn is_object(&self) -> bool {
        self.material_id == 1 || matches!(self.state(), AtomState::Object)
    }

    #[inline]
    pub const fn is_powder(&self) -> bool {
        matches!(self.state(), AtomState::Powder)
    }

    #[inline]
    pub const fn is_solid(&self) -> bool {
        matches!(self.state(), AtomState::Solid)
    }

    #[inline]
    pub const fn is_gas(&self) -> bool {
        matches!(self.state(), AtomState::Gas)
    }

    #[inline]
    pub const fn is_fluid(&self) -> bool {
        self.state().is_fluid()
    }
}
