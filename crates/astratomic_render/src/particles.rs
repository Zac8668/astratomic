//! High-performance GPU-backed particle rendering system for `astratomic_render`.

use std::collections::HashMap;

use astratomic_sim::{
    atom::Atom,
    particle::{Particle, ParticlePool},
};
use bones_framework::prelude::*;

/// Z-layer for rendered particles (in front of terrain at Z=0.0).
pub const PARTICLE_LAYER_Z: f32 = 5.0;

/// Render configuration resource for the particle simulation.
#[derive(HasSchema, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ParticleRenderConfig {
    /// Whether particle rendering is enabled.
    pub enabled: bool,
    /// Whether to render velocity vector trails behind flying particles.
    pub show_trails: bool,
    /// Thickness of particle lines in pixels.
    pub thickness: f32,
}

impl Default for ParticleRenderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_trails: true,
            thickness: 1.0,
        }
    }
}

/// Tracks active particle `Path2d` entities grouped by rendered color.
#[derive(HasSchema, Clone, Default, Debug)]
#[schema(opaque)]
pub struct ParticleRenderState {
    pub entities: Vec<Entity>,
}

/// Registers particle rendering schemas with the global schema registry.
pub fn register_particle_render_schemas() {
    ParticleRenderConfig::register_schema();
    ParticleRenderState::register_schema();
}

/// Determines the rendered RGBA8 color for a particle.
#[inline]
pub fn particle_render_color(p: &Particle) -> [u8; 4] {
    if p.atom.material_id != 0 {
        // Matter-carrying atom: distinct material preview color
        Atom::material_preview_color(p.atom.material_id)
    } else if p.atom.color[3] != 0 {
        // Ephemeral FX: explicit RGBA color
        p.atom.color
    } else {
        [255, 255, 255, 255]
    }
}

/// PostUpdate system that synchronizes particles in [`ParticlePool`] to GPU-drawn [`Path2d`] line batches.
pub fn render_particles_system(
    config: Option<Res<ParticleRenderConfig>>,
    particle_pool: Option<Res<ParticlePool>>,
    mut entities: ResMut<Entities>,
    mut transforms: CompMut<Transform>,
    mut paths: CompMut<Path2d>,
    mut render_state: ResMut<ParticleRenderState>,
) {
    let is_enabled = config.as_ref().map_or(true, |c| c.enabled);
    let show_trails = config.as_ref().map_or(true, |c| c.show_trails);
    let thickness = config.as_ref().map_or(1.0, |c| c.thickness);

    if !is_enabled {
        for &ent in &render_state.entities {
            if entities.is_alive(ent) {
                if let Some(path) = paths.get_mut(ent) {
                    path.points.clear();
                    path.line_breaks.clear();
                }
            }
        }
        return;
    }

    let Some(pool) = particle_pool else {
        for &ent in &render_state.entities {
            if entities.is_alive(ent) {
                if let Some(path) = paths.get_mut(ent) {
                    path.points.clear();
                    path.line_breaks.clear();
                }
            }
        }
        return;
    };

    let particles = pool.particles();
    if particles.is_empty() {
        for &ent in &render_state.entities {
            if entities.is_alive(ent) {
                if let Some(path) = paths.get_mut(ent) {
                    path.points.clear();
                    path.line_breaks.clear();
                }
            }
        }
        return;
    }

    // Group particle coordinates and velocity by their rendered RGBA color.
    let mut groups: HashMap<[u8; 4], Vec<([f32; 2], [f32; 2])>> = HashMap::new();
    for p in particles {
        let rgba = particle_render_color(p);
        groups.entry(rgba).or_default().push((p.pos, p.vel));
    }

    // Ensure enough entities in render_state for all active color groups.
    while render_state.entities.len() < groups.len() {
        let ent = entities.create();
        transforms.insert(
            ent,
            Transform::from_translation(Vec3::new(0.0, 0.0, PARTICLE_LAYER_Z)),
        );
        paths.insert(ent, Path2d::default());
        render_state.entities.push(ent);
    }

    // Render each color group into its dedicated Path2d batch.
    for (i, (&rgba, p_list)) in groups.iter().enumerate() {
        let mut ent = render_state.entities[i];
        if !entities.is_alive(ent) {
            ent = entities.create();
            transforms.insert(
                ent,
                Transform::from_translation(Vec3::new(0.0, 0.0, PARTICLE_LAYER_Z)),
            );
            paths.insert(ent, Path2d::default());
            render_state.entities[i] = ent;
        }

        if let Some(path) = paths.get_mut(ent) {
            path.points.clear();
            path.line_breaks.clear();
            path.color = Color::rgba(
                rgba[0] as f32 / 255.0,
                rgba[1] as f32 / 255.0,
                rgba[2] as f32 / 255.0,
                (rgba[3] as f32 / 255.0).max(0.4),
            );
            path.thickness = thickness;

            for &(pos, vel) in p_list {
                // In ChunkWorld, +Y is downward; in Bones world space, +Y is upward.
                let wx = pos[0];
                let wy = -pos[1];

                // Small cross mark centered at particle position
                path.points.push(Vec2::new(wx - 0.5, wy));
                path.points.push(Vec2::new(wx + 0.5, wy));
                path.line_breaks.push(path.points.len());

                path.points.push(Vec2::new(wx, wy - 0.5));
                path.points.push(Vec2::new(wx, wy + 0.5));
                path.line_breaks.push(path.points.len());

                // Velocity vector trail
                if show_trails {
                    let vx = vel[0] * 0.6;
                    let vy = -vel[1] * 0.6;
                    if vx.abs() > 0.05 || vy.abs() > 0.05 {
                        path.points.push(Vec2::new(wx, wy));
                        path.points.push(Vec2::new(wx + vx, wy + vy));
                        path.line_breaks.push(path.points.len());
                    }
                }
            }
        }
    }

    // Clear unused entities
    for &ent in &render_state.entities[groups.len()..] {
        if entities.is_alive(ent) {
            if let Some(path) = paths.get_mut(ent) {
                path.points.clear();
                path.line_breaks.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use astratomic_sim::plugin::AstratomicSimPlugin;

    #[test]
    fn render_particles_system_allocates_groups_and_cleans_up() {
        let mut session = Session::default();
        session.install_plugin(DefaultSessionPlugin);
        session.install_plugin(AstratomicSimPlugin);
        register_particle_render_schemas();

        session.world.init_resource::<ParticleRenderState>();
        session.world.init_resource::<ParticleRenderConfig>();

        // 1. Spawn two distinct particle colors in ParticlePool:
        // One ephemeral spark (orange), one ephemeral water foam (cyan)
        {
            let mut pool = session.world.resource_mut::<ParticlePool>();
            pool.spawn_fx([10.0, 20.0], [2.0, -1.0], [255, 180, 40, 255], 20, true);
            pool.spawn_fx([30.0, 40.0], [-1.0, 0.5], [80, 200, 255, 230], 15, false);
        }

        // Run render_particles_system
        session.world.run_system(render_particles_system, ());

        {
            let state = session.world.resource::<ParticleRenderState>();
            assert_eq!(state.entities.len(), 2, "should allocate 2 entities for 2 color groups");

            let paths = session.world.component::<Path2d>();
            let p0 = paths.get(state.entities[0]).unwrap();
            let p1 = paths.get(state.entities[1]).unwrap();
            assert!(!p0.points.is_empty());
            assert!(!p1.points.is_empty());
            assert_ne!(p0.color.r(), p1.color.r(), "each group must have distinct color");
        }

        // 2. Clear particles in pool and run system again -> points should be cleared
        {
            let mut pool = session.world.resource_mut::<ParticlePool>();
            pool.clear();
        }

        session.world.run_system(render_particles_system, ());

        {
            let state = session.world.resource::<ParticleRenderState>();
            let paths = session.world.component::<Path2d>();
            for &ent in &state.entities {
                let p = paths.get(ent).unwrap();
                assert!(p.points.is_empty(), "points should be cleared when pool is empty");
            }
        }
    }
}
