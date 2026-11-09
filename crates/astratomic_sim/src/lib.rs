pub mod atom;
pub mod ca;
pub mod chunk;
pub mod chunk_group;
pub mod consts;
pub mod plugin;
pub mod rigidbody;
pub mod world;
pub mod world_store;

pub use world_store as storage;

pub mod prelude {
    pub use crate::{
        atom::*, ca::*, chunk::*, chunk_group::*, consts::*, plugin::*, rigidbody::*, world::*,
        world_store::*,
    };
    pub use bones_framework::prelude::{
        Entity, Handle, IVec2, Image, UVec2, Vec2, Vec3, ivec2, uvec2, vec2, vec3,
    };
    pub use rapier2d::prelude::Collider;
    pub use serde::{Deserialize, Serialize};
    pub use std::collections::{HashMap, HashSet};
}
