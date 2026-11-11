//! Astratomic gameplay modules, content, UI, and Bones sessions.

pub mod camera;
pub mod meta;
pub mod player;
pub mod prefabs;
pub mod sessions;
pub mod ui;
pub mod worldgen;

pub mod prelude {
    pub use crate::camera::*;
    pub use crate::meta::*;
    pub use crate::player::*;
    pub use crate::prefabs::*;
    pub use crate::sessions::*;
    pub use crate::ui::*;
    pub use crate::worldgen::*;
}
