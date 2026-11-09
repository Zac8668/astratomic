use bones_schema::HasSchema;
use rand::Rng;
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
            6 | 7 | 8 | 10 | 11 | 12 => Self::Solid,
            _ => Self::Solid,
        }
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
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, HasSchema)]
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
    /// Bit 3 of `flags` indicates whether a powder/liquid atom is actively moving.
    pub const FLAG_MOVING: u8 = 0b0000_1000;
    /// Bits 4..6 of `flags` store `f_idle` (0..=3).
    pub const F_IDLE_SHIFT: u8 = 4;
    pub const F_IDLE_MASK: u8 = 0b0011_0000;
    /// Bits 6..8 of `flags` store `updated_at` parity (`dt & 0b11`).
    pub const UPDATED_AT_SHIFT: u8 = 6;
    pub const UPDATED_AT_MASK: u8 = 0b1100_0000;

    pub const VOID: Self = Self {
        color: [0, 0, 0, 0],
        material_id: 0,
        flags: 0,
        vx: 0,
        vy: 0,
    };

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
        Self {
            color,
            material_id,
            flags: state as u8,
            vx: 0,
            vy: 0,
        }
    }

    pub fn new(id: u8, materials: &Materials) -> Self {
        let state = materials
            .0
            .get(id as usize)
            .map(|m| m.default_state)
            .unwrap_or_else(|| AtomState::default_for_material_id(id));

        let mut rng = rand::rng();
        let color = match id {
            0 | 1 => [0, 0, 0, 0],
            2 => [
                // Sand
                (230 + rng.random_range(-20_i16..20_i16)) as u8,
                (197 + rng.random_range(-20_i16..20_i16)) as u8,
                (92 + rng.random_range(-20_i16..20_i16)) as u8,
                255,
            ],
            3 => [
                // Water
                (20 + rng.random_range(-20_i16..20_i16)) as u8,
                (125 + rng.random_range(-20_i16..20_i16)) as u8,
                (204 + rng.random_range(-20_i16..20_i16)) as u8,
                150,
            ],
            4 => [
                // Gravel
                (110 + rng.random_range(-12_i16..12_i16)) as u8,
                (110 + rng.random_range(-12_i16..12_i16)) as u8,
                (110 + rng.random_range(-12_i16..12_i16)) as u8,
                255,
            ],
            5 => [
                // Lava
                (245 + rng.random_range(-10_i16..10_i16)) as u8,
                (140 + rng.random_range(-20_i16..20_i16)) as u8,
                (10 + rng.random_range(-10_i16..10_i16)) as u8,
                255,
            ],
            6 => [
                // Grass
                (40 + rng.random_range(-10_i16..10_i16)) as u8,
                (165 + rng.random_range(-10_i16..10_i16)) as u8,
                (49 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            7 => [
                // Dirt
                (120 + rng.random_range(-10_i16..10_i16)) as u8,
                (70 + rng.random_range(-10_i16..10_i16)) as u8,
                (40 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            8 => [
                // Rock
                (80 + rng.random_range(-10_i16..10_i16)) as u8,
                (80 + rng.random_range(-10_i16..10_i16)) as u8,
                (80 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            9 => [
                // Acid
                (10 + rng.random_range(-10_i16..10_i16)) as u8,
                (230 + rng.random_range(-10_i16..10_i16)) as u8,
                (40 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            10 => [
                // Compacted Dirt
                (100 + rng.random_range(-10_i16..10_i16)) as u8,
                (50 + rng.random_range(-10_i16..10_i16)) as u8,
                (30 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            11 => [
                // Leaves
                (60 + rng.random_range(-15_i16..15_i16)) as u8,
                (180 + rng.random_range(-20_i16..20_i16)) as u8,
                (60 + rng.random_range(-10_i16..10_i16)) as u8,
                220,
            ],
            12 => [
                // Wood
                (120 + rng.random_range(-10_i16..10_i16)) as u8,
                (80 + rng.random_range(-10_i16..10_i16)) as u8,
                (40 + rng.random_range(-5_i16..5_i16)) as u8,
                255,
            ],
            _ => panic!("Atom not found, invalid ID. {id}"),
        };

        Self::with_state(id, state, color)
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
        self.flags = (self.flags & !Self::STATE_MASK) | (state as u8);
    }

    #[inline]
    pub const fn moving(&self) -> bool {
        (self.flags & Self::FLAG_MOVING) != 0
    }

    #[inline]
    pub fn set_moving(&mut self, moving: bool) {
        if moving {
            self.flags |= Self::FLAG_MOVING;
        } else {
            self.flags &= !Self::FLAG_MOVING;
        }
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
}
