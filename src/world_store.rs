use crate::prelude::*;

pub const WORLD_DB_PATH: &str = "assets/world/world_db";

fn ensure_world_dir() {
    let _ = std::fs::create_dir_all("assets/world");
}

pub fn open_world_db() -> sled::Db {
    ensure_world_dir();
    sled::open(WORLD_DB_PATH).unwrap()
}

fn chunk_key(pos: IVec2) -> [u8; 8] {
    let mut key = [0_u8; 8];
    key[0..4].copy_from_slice(&pos.x.to_le_bytes());
    key[4..8].copy_from_slice(&pos.y.to_le_bytes());
    key
}

fn encode_chunk(chunk: &Chunk) -> Vec<u8> {
    let raw = bincode::serialize(chunk).unwrap();
    // zstd compression level: 3 is a good speed/ratio default.
    zstd::stream::encode_all(raw.as_slice(), 3).unwrap()
}

fn decode_chunk(bytes: &[u8]) -> Chunk {
    let raw = zstd::stream::decode_all(bytes).unwrap();
    bincode::deserialize(raw.as_slice()).unwrap()
}

pub fn db_has_any_chunks(db: &sled::Db) -> bool {
    db.iter().next().is_some()
}

pub fn load_chunk_from_db(db: &sled::Db, pos: IVec2) -> Option<Chunk> {
    db.get(chunk_key(pos))
        .unwrap()
        .map(|ivec| decode_chunk(&ivec))
}

pub fn save_chunk_to_db(db: &sled::Db, pos: IVec2, chunk: &Chunk) {
    db.insert(chunk_key(pos), encode_chunk(chunk)).unwrap();
}
