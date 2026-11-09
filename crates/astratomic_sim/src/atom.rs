use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, PartialEq, Debug, Serialize, Deserialize, Eq)]
pub struct Atom {
    pub color: [u8; 4],
    pub state: AtomState,
    pub id: u8,

    #[serde(skip)]
    pub moving: bool,
    #[serde(skip)]
    pub speed: (i8, i8),
    #[serde(skip)]
    pub f_idle: u8,
    #[serde(skip)]
    pub updated_at: u8,
}

#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Debug)]
pub enum AtomState {
    Solid,
    Object,
    Powder,
    Liquid,
    Gas,
    #[default]
    Void,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
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

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Materials(pub Vec<Material>);

impl Materials {
    pub fn get_from_atom(&self, atom: &Atom) -> &Material {
        &self.0[atom.id as usize]
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
    pub fn object() -> Self {
        Atom {
            id: 1,
            state: AtomState::Object,
            ..Default::default()
        }
    }

    pub fn new(id: u8, materials: &Materials) -> Atom {
        let mut atom = Atom {
            id,
            state: materials[id].default_state,
            ..Default::default()
        };

        let mut rng = rand::rng();
        match id {
            2 => {
                // Sand
                atom.color = [
                    (230 + rng.random_range(-20_i16..20_i16)) as u8,
                    (197 + rng.random_range(-20_i16..20_i16)) as u8,
                    (92 + rng.random_range(-20_i16..20_i16)) as u8,
                    255,
                ];
            }
            3 => {
                // Water
                atom.color = [
                    (20 + rng.random_range(-20_i16..20_i16)) as u8,
                    (125 + rng.random_range(-20_i16..20_i16)) as u8,
                    (204 + rng.random_range(-20_i16..20_i16)) as u8,
                    150,
                ];
            }
            4 => {
                // Gravel
                atom.color = [
                    (110 + rng.random_range(-12_i16..12_i16)) as u8,
                    (110 + rng.random_range(-12_i16..12_i16)) as u8,
                    (110 + rng.random_range(-12_i16..12_i16)) as u8,
                    255,
                ];
            }
            5 => {
                // Lava
                atom.color = [
                    (245 + rng.random_range(-10_i16..10_i16)) as u8,
                    (140 + rng.random_range(-20_i16..20_i16)) as u8,
                    (10 + rng.random_range(-10_i16..10_i16)) as u8,
                    255,
                ];
            }
            6 => {
                // Grass
                atom.color = [
                    (40 + rng.random_range(-10_i16..10_i16)) as u8,
                    (165 + rng.random_range(-10_i16..10_i16)) as u8,
                    (49 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            7 => {
                // Dirt
                atom.color = [
                    (120 + rng.random_range(-10_i16..10_i16)) as u8,
                    (70 + rng.random_range(-10_i16..10_i16)) as u8,
                    (40 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            8 => {
                // Rock
                atom.color = [
                    (80 + rng.random_range(-10_i16..10_i16)) as u8,
                    (80 + rng.random_range(-10_i16..10_i16)) as u8,
                    (80 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            9 => {
                // Acid
                atom.color = [
                    (10 + rng.random_range(-10_i16..10_i16)) as u8,
                    (230 + rng.random_range(-10_i16..10_i16)) as u8,
                    (40 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            10 => {
                // Compacted Dirt
                atom.color = [
                    (100 + rng.random_range(-10_i16..10_i16)) as u8,
                    (50 + rng.random_range(-10_i16..10_i16)) as u8,
                    (30 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            11 => {
                // Leaves
                atom.color = [
                    (60 + rng.random_range(-15_i16..15_i16)) as u8,
                    (180 + rng.random_range(-20_i16..20_i16)) as u8,
                    (60 + rng.random_range(-10_i16..10_i16)) as u8,
                    220,
                ];
            }
            12 => {
                // Wood
                atom.color = [
                    (120 + rng.random_range(-10_i16..10_i16)) as u8,
                    (80 + rng.random_range(-10_i16..10_i16)) as u8,
                    (40 + rng.random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            _ => panic!("Atom not found, invalid ID. {id}"),
        }

        atom
    }

    pub fn is_liquid(&self) -> bool {
        matches!(self.state, AtomState::Liquid)
    }

    pub fn is_void(&self) -> bool {
        matches!(self.state, AtomState::Void)
    }

    pub fn is_object(&self) -> bool {
        matches!(self.state, AtomState::Object)
    }

    pub fn is_powder(&self) -> bool {
        matches!(self.state, AtomState::Powder)
    }

    pub fn is_solid(&self) -> bool {
        matches!(self.state, AtomState::Solid)
    }
}
