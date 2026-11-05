use rand::Rng;
use std::collections::HashSet;
use std::f32::consts::PI;

use serde::Deserialize;

use crate::prelude::*;

#[derive(Clone, Copy, Default, PartialEq, Debug, Serialize, Deserialize, Eq)]
pub struct Atom {
    pub color: [u8; 4],
    pub state: AtomState,
    pub id: u8,

    #[serde(skip)]
    pub moving: bool,
    #[serde(skip)]
    pub speed: (i8, i8),
    // Frames idle
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

impl Atom {
    pub fn object() -> Self {
        Atom {
            id: 1,
            //color: [255, 255, 255, 255],
            ..Default::default()
        }
    }

    pub fn new(id: u8, materials: &Materials) -> Atom {
        let mut atom = Atom {
            id,
            state: materials[id].default_state,
            ..Default::default()
        };

        //Change color and material, etc
        match id {
            2 => {
                //Sand
                atom.color = [
                    (230 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (197 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (92 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    255,
                ];
            }
            3 => {
                //Water
                atom.color = [
                    (20 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (125 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (204 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    150,
                ];
            }
            4 => {
                //Gravel
                atom.color = [
                    (110 + rand::rng().random_range(-12_i16..12_i16)) as u8,
                    (110 + rand::rng().random_range(-12_i16..12_i16)) as u8,
                    (110 + rand::rng().random_range(-12_i16..12_i16)) as u8,
                    255,
                ];
            }
            5 => {
                //Lava
                atom.color = [
                    (245 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (140 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (10 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    255,
                ];
            }
            6 => {
                //Grass
                atom.color = [
                    (40 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (165 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (49 + rand::rng().random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            7 => {
                //Dirt
                atom.color = [
                    (120 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (70 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (40 + rand::rng().random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            8 => {
                //Rock
                atom.color = [
                    (80 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (80 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (80 + rand::rng().random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            9 => {
                //Acid
                atom.color = [
                    (10 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (230 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (40 + rand::rng().random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            10 => {
                //Compacted Dirt
                atom.color = [
                    (100 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (50 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (30 + rand::rng().random_range(-5_i16..5_i16)) as u8,
                    255,
                ];
            }
            11 => {
                // Leaves
                atom.color = [
                    (60 + rand::rng().random_range(-15_i16..15_i16)) as u8,
                    (180 + rand::rng().random_range(-20_i16..20_i16)) as u8,
                    (60 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    220,
                ];
            }
            12 => {
                // Wood
                atom.color = [
                    (120 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (80 + rand::rng().random_range(-10_i16..10_i16)) as u8,
                    (40 + rand::rng().random_range(-5_i16..5_i16)) as u8,
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

// Update different types of atoms

/// Updates powder and returns atoms awakened
pub fn update_powder(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    dt: u8,
    inertial_resistance: f32,
) -> HashSet<IVec2> {
    let mut awakened = HashSet::new();

    let mut cur_pos = pos;

    // Get atom speed
    let mut speed = get_speed(chunks, cur_pos);
    if speed < ATOM_TERM_VEL {
        speed += GRAVITY;
        set_speed(chunks, cur_pos, speed);
    }

    for _ in 0..speed {
        let neigh = down_neigh(chunks, cur_pos, &[(3, 0.2)], dt);
        let mut swapped = false;
        if get_moving(chunks, cur_pos) {
            set_moving(chunks, cur_pos, inertial_resistance);
        }
        for neigh in neigh {
            if neigh.0 {
                set_moving(chunks, cur_pos, inertial_resistance);
                swap(chunks, cur_pos, cur_pos + neigh.1, dt);
                awakened.insert(cur_pos);
                cur_pos += neigh.1;
                awakened.insert(cur_pos);
                swapped = true;
                break;
            }
        }

        if !swapped {
            let vel = Vec2::new(0.0, -(speed as f32));

            set_vel(
                chunks,
                cur_pos,
                Vec2::from_angle(rand::rng().random_range(-PI / 2.0..PI / 2.))
                    .rotate(vel * 0.3 * 1. * (1. - inertial_resistance))
                    .as_ivec2(),
            );

            break;
        }
    }

    awakened
}

/// Updates liquid and returns atoms awakened
pub fn update_liquid(
    chunks: &mut UpdateChunksType,
    pos: IVec2,
    flow: u8,
    dt: u8,
) -> HashSet<IVec2> {
    let mut awakened = HashSet::new();
    let mut cur_pos = pos;

    // Get fall speed
    let mut speed = get_speed(chunks, pos);
    if speed < ATOM_TERM_VEL {
        speed += GRAVITY;
        set_speed(chunks, pos, speed);
    }

    let mut swapped = false;
    for _ in 0..speed {
        let neigh = down_neigh(chunks, cur_pos, &[], dt);
        for neigh in neigh {
            if neigh.0 {
                swap(chunks, cur_pos, cur_pos + neigh.1, dt);
                awakened.insert(cur_pos);
                cur_pos += neigh.1;
                awakened.insert(cur_pos);
                swapped = true;

                break;
            }
        }
    }

    if !swapped {
        set_speed(chunks, cur_pos, 0);
        let neigh = side_neigh(chunks, cur_pos, &[], dt);
        let side = if neigh[0].0 {
            Some(neigh[0].1.x)
        } else if neigh[1].0 {
            Some(neigh[1].1.x)
        } else {
            None
        };

        if let Some(side) = side {
            for _ in 0..flow {
                let atom = get_atom(chunks, cur_pos);
                if !swapable(chunks, cur_pos + IVec2::new(side, 0), &[], atom, dt) {
                    break;
                }

                swap(chunks, cur_pos, cur_pos + IVec2::new(side, 0), dt);
                awakened.insert(cur_pos);
                cur_pos += IVec2::new(side, 0);
                awakened.insert(cur_pos);
            }
        }
    }

    awakened
}

/// This updates the atom with a vector based velocity, not a automata like one
pub fn update_atom(chunks: &mut UpdateChunksType, pos: IVec2, dt: u8) -> HashSet<IVec2> {
    let mut awakened = HashSet::new();
    let mut cur_pos = pos;

    // Add gravity
    let mut vel = get_vel(chunks, cur_pos);
    if vel.y < ATOM_TERM_VEL as i32 {
        vel += GRAVITY as i32 * IVec2::Y;
        set_vel(chunks, cur_pos, vel);
    }

    // Move
    for pos in Line::new(cur_pos, vel) {
        awakened.insert(cur_pos);
        let atom = get_atom(chunks, cur_pos);
        if swapable(chunks, pos, &[], atom, dt) {
            swap(chunks, cur_pos, pos, dt);
            cur_pos = pos;
            awakened.insert(cur_pos);
        } else if get_atom(chunks, pos).is_liquid() && get_atom(chunks, cur_pos).is_liquid() {
            awakened.insert(pos);
            set_vel(chunks, pos, vel * 4 / 5);
            set_vel(chunks, cur_pos, vel / 5);
            break;
        } else {
            if vel.abs().x > 4 && vel.abs().y > 4 {
                set_vel(
                    chunks,
                    cur_pos,
                    (Vec2::from_angle(PI).rotate(vel.as_vec2()) * 0.5).as_ivec2(),
                );
            } else if !swapable(chunks, cur_pos + IVec2::Y, &[], atom, dt) {
                set_vel(chunks, cur_pos, IVec2::ZERO);
            }
            break;
        }
    }

    awakened
}
