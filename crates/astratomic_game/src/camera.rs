//! In-game smooth camera tracking and zoom management ported to Bones.

use bones_framework::prelude::*;

use crate::player::{LocalView, Player};

/// A resource for the state of the in-game smooth camera.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct TrackingCamera {
    /// The position in world space of the origin of this camera.
    pub position: Vec2,

    /// The current target of the camera; what it smoothly focuses on.
    pub target: Vec2,

    /// The half-size of the rectangle around the center of the screen where
    /// the player can move without the camera retargeting. When the player
    /// leaves this rectangle, the camera will retarget to include the player
    /// back into this region of the screen.
    pub tracking_size: Vec2,

    /// The half-size of the rectangle around the center of the screen where
    /// the camera will smoothly interpolate. If the player leaves this region,
    /// the camera will clamp to keep the player within it.
    pub clamp_size: Vec2,

    /// A dead distance from the edge of the tracking region to the player
    /// where the camera will not perform any tracking, even if the player is
    /// minutely outside of the tracking region. This is provided so that the
    /// camera can recenter even if the player has not moved since a track.
    pub dead_zone: Vec2,

    /// The proportion (between 0.0-1.0) that the camera reaches its target
    /// from its initial position during a second's time.
    pub speed: f64,

    /// A timeout to recenter the camera on the player even if the player has
    /// not left the tracking rectangle.
    pub recenter_timeout: f32,

    /// The duration in seconds since the player has left the tracking rectangle.
    ///
    /// When this duration reaches `recenter_timeout`, the player will be
    /// recentered.
    pub last_track: f32,
}

impl Default for TrackingCamera {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            target: Vec2::ZERO,
            tracking_size: vec2(16.0, 9.0),
            clamp_size: vec2(48.0, 28.0),
            dead_zone: Vec2::splat(0.1),
            speed: 0.98,
            recenter_timeout: 3.0,
            last_track: 0.0,
        }
    }
}

impl TrackingCamera {
    /// Constructs a new [`TrackingCamera`] initialized at the specified position.
    pub fn new(initial_pos: Vec2) -> Self {
        Self {
            position: initial_pos,
            target: initial_pos,
            ..Default::default()
        }
    }

    /// Snap the camera directly to a position (resets target and recenter timer).
    pub fn snap_to(&mut self, pos: Vec2) {
        self.position = pos;
        self.target = pos;
        self.last_track = 0.0;
    }

    /// Update the camera with the current position and this frame's delta time.
    pub fn update(&mut self, player_pos: Vec2, dt: f64) {
        // update target with player position
        self.track_player(player_pos);

        // track time since last time we had to track the player
        let new_last_track = self.last_track + dt as f32;

        // test if we've triggered a recenter
        if self.last_track < self.recenter_timeout && new_last_track > self.recenter_timeout {
            // target the player
            self.target = player_pos;
        }

        // update the duration since last track
        self.last_track = new_last_track;

        // lerp the current position towards the target
        // correct lerp degree using delta time
        // perform pow() with high precision
        let lerp = 1.0 - (1.0 - self.speed).powf(dt) as f32;
        self.position = self.position.lerp(self.target, lerp);
    }

    /// Helper function to clamp a rectangle (given as a half-size at the
    /// origin) so that a point lays within it. Returns an offset to apply to
    /// the rectangle, if any was required.
    pub fn clamp_rect(half_size: Vec2, point: Vec2) -> Option<Vec2> {
        let mut ox = None;
        let mut oy = None;

        if point.x > half_size.x {
            ox = Some(point.x - half_size.x);
        } else if point.x < -half_size.x {
            ox = Some(point.x + half_size.x);
        }

        if point.y > half_size.y {
            oy = Some(point.y - half_size.y);
        } else if point.y < -half_size.y {
            oy = Some(point.y + half_size.y);
        }

        if let (None, None) = (ox, oy) {
            None
        } else {
            Some(vec2(ox.unwrap_or(0.0), oy.unwrap_or(0.0)))
        }
    }

    pub fn track_player(&mut self, player_pos: Vec2) {
        // get current relative position to player
        let rel_pos = player_pos - self.position;

        // track the player and reset last track if change was necessary
        if let Some(offset) = Self::clamp_rect(self.tracking_size, rel_pos) {
            // skip tracking if it falls within the dead zone
            if !(self.dead_zone).cmpgt(offset.abs()).all() {
                self.target = self.position + offset;
                self.last_track = 0.0;
            }
        }

        // clamp the player within the screen
        if let Some(offset) = Self::clamp_rect(self.clamp_size, rel_pos) {
            self.position += offset;
        }
    }
}

/// Zoom scale resource for camera sizing.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Zoom(pub f32);

impl Default for Zoom {
    fn default() -> Self {
        Self(0.23)
    }
}

/// Registers camera schemas with the global Bones schema registry.
pub fn register_camera_schemas() {
    TrackingCamera::register_schema();
    Zoom::register_schema();
}

/// Standalone camera update system matching the signature and behavior of the original system.
///
/// If [`LocalView`] is active, camera tracking is handled by [`crate::player::update_local_view_camera_system`]
/// to guarantee sub-tick interpolation with [`crate::player::KinematicActor`]. Otherwise, this system tracks any
/// entity with a [`Player`] component.
pub fn update_camera(
    time: Res<Time>,
    entities: Res<Entities>,
    players: Comp<Player>,
    mut camera_transforms: CompMut<Transform>,
    mut tracking: ResMut<TrackingCamera>,
    cameras: Comp<Camera>,
    local_view: Option<Res<LocalView>>,
) {
    if local_view.is_some() {
        return;
    }
    let dt = time.delta_seconds_f64().clamp(0.0001, 0.1);
    let mut player_pos = None;
    for (ent, _player) in entities.iter_with(&players) {
        if let Some(transform) = camera_transforms.get(ent) {
            player_pos = Some(transform.translation.xy());
            break;
        }
    }

    if let Some(pos) = player_pos {
        tracking.update(pos, dt);
        for (cam_ent, _camera) in entities.iter_with(&cameras) {
            if let Some(camera_transform) = camera_transforms.get_mut(cam_ent) {
                camera_transform.translation.x = tracking.position.x;
                camera_transform.translation.y = tracking.position.y;
            }
        }
    }
}

/// Event-ready sizing and resize system that updates camera scaling and [`CameraSize`] based on window dimensions and [`Zoom`].
pub fn on_resize_system(
    window: Option<Res<Window>>,
    zoom: Option<Res<Zoom>>,
    entities: Res<Entities>,
    mut cameras: CompMut<Camera>,
    mut transforms: CompMut<Transform>,
) {
    let (Some(window), Some(zoom)) = (window, zoom) else {
        return;
    };
    if window.size.x <= 0.0 || window.size.y <= 0.0 {
        return;
    }

    let x = 1920.0 / window.size.x * zoom.0;
    let y = 1080.0 / window.size.y * zoom.0;
    let min_scale = x.min(y);

    for (cam_ent, camera) in entities.iter_with(&mut cameras) {
        if let Some(camera_transform) = transforms.get_mut(cam_ent) {
            camera_transform.scale.x = min_scale;
            camera_transform.scale.y = min_scale;
        }
        camera.size = CameraSize::FixedHeight(1080.0 * zoom.0);
    }
}

/// Bones plugin that registers camera schemas and installs tracking systems.
#[derive(Default, Clone, Copy)]
pub struct CameraPlugin;

pub type AstratomicCameraPlugin = CameraPlugin;

impl GamePlugin for CameraPlugin {
    fn install(self, _game: &mut Game) {
        register_camera_schemas();
    }
}

impl SessionPlugin for CameraPlugin {
    fn install(self, session: &mut Session) {
        register_camera_schemas();
        session.world.init_resource::<TrackingCamera>();
        session.world.init_resource::<Zoom>();
        session
            .add_system_to_stage(CoreStage::PostUpdate, update_camera)
            .add_system_to_stage(CoreStage::PostUpdate, on_resize_system);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_camera_defaults() {
        let cam = TrackingCamera::default();
        assert_eq!(cam.position, Vec2::ZERO);
        assert_eq!(cam.target, Vec2::ZERO);
        assert_eq!(cam.tracking_size, vec2(16.0, 9.0));
        assert_eq!(cam.clamp_size, vec2(48.0, 28.0));
        assert_eq!(cam.dead_zone, Vec2::splat(0.1));
        assert!((cam.speed - 0.98).abs() < f64::EPSILON);
        assert!((cam.recenter_timeout - 3.0).abs() < f32::EPSILON);
        assert_eq!(cam.last_track, 0.0);
    }

    #[test]
    fn test_clamp_rect() {
        let half_size = vec2(16.0, 9.0);

        // Inside rectangle
        assert_eq!(TrackingCamera::clamp_rect(half_size, vec2(10.0, 5.0)), None);
        assert_eq!(TrackingCamera::clamp_rect(half_size, vec2(-10.0, -5.0)), None);

        // Outside on +X
        assert_eq!(
            TrackingCamera::clamp_rect(half_size, vec2(20.0, 0.0)),
            Some(vec2(4.0, 0.0))
        );

        // Outside on -X and +Y
        assert_eq!(
            TrackingCamera::clamp_rect(half_size, vec2(-20.0, 12.0)),
            Some(vec2(-4.0, 3.0))
        );
    }

    #[test]
    fn test_track_player_within_dead_zone() {
        let mut cam = TrackingCamera::default();
        // Move slightly past tracking size (16.05 vs 16.0 -> offset 0.05, within dead_zone 0.1)
        cam.track_player(vec2(16.05, 0.0));
        assert_eq!(cam.target, Vec2::ZERO);
        assert_eq!(cam.last_track, 0.0);
    }

    #[test]
    fn test_track_player_outside_tracking_box() {
        let mut cam = TrackingCamera::default();
        // Move past tracking size and outside dead_zone
        cam.track_player(vec2(20.0, 0.0));
        assert_eq!(cam.target, vec2(4.0, 0.0));
        assert_eq!(cam.last_track, 0.0);
    }

    #[test]
    fn test_track_player_hard_clamping() {
        let mut cam = TrackingCamera::default();
        // Move way past clamp_size (48.0, 28.0)
        cam.track_player(vec2(60.0, 40.0));
        // Camera position clamps immediately so player remains within clamp_size
        assert_eq!(cam.position, vec2(12.0, 12.0));
    }

    #[test]
    fn test_update_lerp_and_recenter() {
        let mut cam = TrackingCamera::default();
        cam.track_player(vec2(30.0, 0.0));
        assert_eq!(cam.target, vec2(14.0, 0.0));

        // Advance 0.5s: position should smoothly move toward target
        cam.update(vec2(30.0, 0.0), 0.5);
        assert!(cam.position.x > 0.0 && cam.position.x <= 14.0);

        // Advance past recenter_timeout without player moving outside tracking size
        cam.update(vec2(cam.position.x, cam.position.y), 3.5);
        // Target should recenter to the player's position
        assert_eq!(cam.target, vec2(cam.position.x, cam.position.y));
    }

    #[test]
    fn test_snap_to() {
        let mut cam = TrackingCamera::default();
        cam.target = vec2(50.0, 50.0);
        cam.position = vec2(20.0, 20.0);
        cam.last_track = 2.5;

        cam.snap_to(vec2(100.0, -200.0));
        assert_eq!(cam.position, vec2(100.0, -200.0));
        assert_eq!(cam.target, vec2(100.0, -200.0));
        assert_eq!(cam.last_track, 0.0);
    }

    #[test]
    fn test_camera_schema_registration() {
        register_camera_schemas();
    }
}
