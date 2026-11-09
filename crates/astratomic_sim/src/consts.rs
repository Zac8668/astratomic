// Chunk length MUST be divisible by 4
pub const CHUNK_LENGHT: usize = 64;
pub const HALF_CHUNK_LENGHT: usize = CHUNK_LENGHT / 2;

pub const CHUNK_LEN: usize = CHUNK_LENGHT * CHUNK_LENGHT;
pub const HALF_CHUNK_LEN: usize = CHUNK_LEN / 2;
pub const QUARTER_CHUNK_LEN: usize = CHUNK_LEN / 4;

pub const GRAVITY: u8 = 1;
pub const ATOM_TERM_VEL: u8 = 20;
pub const FRAMES_SLEEP: u8 = 1;

// Must be even
pub const LOAD_WIDTH: i32 = 32;
pub const LOAD_HEIGHT: i32 = 18;
