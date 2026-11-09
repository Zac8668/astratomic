// Chunk length MUST be divisible by 4
pub const CHUNK_SIZE: usize = 64;
pub const HALF_CHUNK_SIZE: usize = CHUNK_SIZE / 2;

pub const CHUNK_LENGHT: usize = CHUNK_SIZE;
pub const HALF_CHUNK_LENGHT: usize = CHUNK_SIZE / 2;

pub const CHUNK_LEN: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const HALF_CHUNK_LEN: usize = CHUNK_LEN / 2;
pub const QUARTER_CHUNK_LEN: usize = CHUNK_LEN / 4;

pub const GRAVITY: u8 = 1;
pub const ATOM_TERM_VEL: u8 = 20;
pub const FRAMES_SLEEP: u8 = 1;

// Must be even
pub const LOAD_WIDTH: i32 = 32;
pub const LOAD_HEIGHT: i32 = 18;

// Actor & Player physics constants
/// Maximum vertical height in atoms that the player/actor can automatically step up (e.g. 2, 3, 4).
pub const PLAYER_STEP_HEIGHT: i32 = 5;
pub const UP_WALK_HEIGHT: i32 = PLAYER_STEP_HEIGHT;

pub const FUEL_MAX: f32 = 50.0;
pub const FUEL_REGEN: f32 = 1.0;
pub const FUEL_COMSUMPTON: f32 = 0.48;
pub const JETPACK_FORCE: f32 = 1.5;
pub const JETPACK_MAX: f32 = 3.0;

pub const JUMP_MAG: f32 = 5.0;
pub const PRESSED_JUMP_MAG: f32 = 0.2;
pub const TIME_JUMP_PRESSED: f32 = 0.8;
pub const RUN_SPEED: f32 = 2.5;

pub const TOOL_DISTANCE: f32 = 32.0;
pub const TOOL_RANGE: f32 = 16.0;

pub const PLAYER_TERM_VEL: u8 = 20;
pub const PLAYER_LAYER: f32 = 2.0;
