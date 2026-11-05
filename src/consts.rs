use bevy::prelude::Color;

// Chunk Length consts
// Chunk length MUST be divisible by 4
pub const CHUNK_LENGHT: usize = 64;
pub const HALF_CHUNK_LENGHT: usize = CHUNK_LENGHT / 2;

pub const CHUNK_LEN: usize = CHUNK_LENGHT * CHUNK_LENGHT;
pub const HALF_CHUNK_LEN: usize = CHUNK_LEN / 2;
pub const QUARTER_CHUNK_LEN: usize = CHUNK_LEN / 4;

// Actor consts

pub const UP_WALK_HEIGHT: usize = 5;

// Player consts
pub const FUEL_MAX: f32 = 50.;
pub const FUEL_REGEN: f32 = 1.;
pub const FUEL_COMSUMPTON: f32 = 0.48;
pub const JETPACK_FORCE: f32 = 1.5;
pub const JETPACK_MAX: f32 = 3.;

pub const JUMP_MAG: f32 = 5.;
pub const PRESSED_JUMP_MAG: f32 = 0.2;
pub const TIME_JUMP_PRESSED: f32 = 0.8;
pub const RUN_SPEED: f32 = 2.5;

pub const TOOL_DISTANCE: f32 = 32.;
pub const TOOL_RANGE: f32 = 16.;

pub const ZOOM_LOWER_BOUND: f32 = 0.15;
pub const ZOOM_UPPER_BOUND: f32 = 4.0;

pub const PLAYER_TERM_VEL: u8 = 20;

pub const DAMAGE_FEED_TIME: f32 = 0.2;

// Engine consts

pub const GRAVITY: u8 = 1;
pub const ATOM_TERM_VEL: u8 = 20;
pub const FRAMES_SLEEP: u8 = 1;
//Has to be even
pub const LOAD_WIDTH: i32 = 32;
pub const LOAD_HEIGHT: i32 = 18;

pub const _CAMERA_SPEED: f32 = 10.;

//Layers
pub const PLAYER_LAYER: f32 = 1.;
pub const PARTICLE_LAYER: f32 = 10.;
pub const AUTOMATA_LAYER: f32 = 100.;
pub const RIGIDBODY_LAYER: f32 = 1000.;

//Buttons
pub const NORMAL_BUTTON: Color = Color::srgb(0.15, 0.15, 0.15);
pub const HOVERED_BUTTON: Color = Color::srgb(0.25, 0.25, 0.25);
pub const PRESSED_BUTTON: Color = Color::srgb(0.35, 0.75, 0.35);

//How much the rigidbody needs to be submerged to totally cut of gravity
//pub const STOP_RATE: f32 = 0.05;
