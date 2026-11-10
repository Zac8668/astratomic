pub mod actor;
pub mod plugin;
pub mod rigidbody;

pub use rapier2d;

pub mod prelude {
    pub use crate::{actor::*, plugin::*, rigidbody::*};
    pub use rapier2d;
    pub use rapier2d::prelude::{ColliderHandle, RigidBodyHandle};
}
