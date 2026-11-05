use crate::prelude::*;

#[derive(Component, Clone)]
pub struct Actor {
    pub width: u8,
    pub height: u8,
    pub pos: IVec2,
    pub vel: Vec2,
    pub colliding: Option<f32>,
}

//Called before simulations
pub fn fill_actors(
    mut chunk_manager: ResMut<ChunkManager>,
    actors: Query<(&Actor, Entity)>,
    mut dirty_rects: ResMut<DirtyRects>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
    mut ev_damage: EventWriter<DamageEvent>,
) {
    let materials = materials.0.get(&materials.1 .0).unwrap();
    let mut damages = 0.;

    for (actor, ent) in actors.iter() {
        for x_off in 0..actor.width as i32 {
            for y_off in 0..actor.height as i32 {
                let pos = global_to_chunk(actor.pos + ivec2(x_off, y_off));
                if let Some(atom) = chunk_manager.get_mut_atom(pos) {
                    if atom.is_void() {
                        *atom = Atom::object();
                    } else if materials[atom.id].damage > 0. {
                        damages += materials[atom.id].damage / (actor.width * actor.height) as f32;
                    }
                }
                update_dirty_rects_3x3(&mut dirty_rects.current, pos);
            }
        }

        if damages > 0. {
            ev_damage.send(DamageEvent::new(ent, damages));
        }
    }
}

//Called after simulation, before actor update
pub fn unfill_actors(mut chunk_manager: ResMut<ChunkManager>, actors: Query<&Actor>) {
    for actor in actors.iter() {
        for x_off in 0..actor.width as i32 {
            for y_off in 0..actor.height as i32 {
                let pos = global_to_chunk(actor.pos + ivec2(x_off, y_off));
                if let Some(atom) = chunk_manager.get_mut_atom(pos) {
                    if atom.is_object() {
                        *atom = Atom::default();
                    }
                }
            }
        }
    }
}

pub fn on_ground(chunk_manager: &ChunkManager, actor: &Actor) -> bool {
    for x_off in 0..actor.width {
        let chunk_pos = global_to_chunk(actor.pos + ivec2(x_off as i32, actor.height as i32));

        if let Some(atom) = chunk_manager.get_atom(&chunk_pos) {
            if atom.is_powder() || atom.is_solid() {
                return true;
            }
        } else {
            return true;
        }
    }

    false
}

pub fn update_actors(mut chunk_manager: ResMut<ChunkManager>, mut actors: Query<&mut Actor>) {
    for mut actor in actors.iter_mut() {
        actor.colliding = None;

        let mut prev = actor.pos;
        for v in Line::new(actor.pos, actor.vel.as_ivec2()) {
            let move_hor = match (prev.x != v.x, prev.y != v.y) {
                (true, false) => true,
                (false, true) => false,
                (true, true) => true,
                _ => unreachable!(),
            };

            if move_hor {
                let moved_x = move_x(&mut chunk_manager, &mut actor, (v.x - prev.x).signum());
                if on_ground(&chunk_manager, &actor) {
                    let starting_y = actor.pos.y;
                    if !moved_x {
                        //If we can't move to the left or right
                        //Check if we can get up a stair-like structure
                        for i in 1..=UP_WALK_HEIGHT {
                            let moved_y = move_y(&mut chunk_manager, &mut actor, -1);
                            //Abort if we couldn't move up, or if we moved up but couldn't move sideways on the last step
                            if !moved_y
                                || i == UP_WALK_HEIGHT
                                    && !move_x(
                                        &mut chunk_manager,
                                        &mut actor,
                                        (v.x - prev.x).signum(),
                                    )
                            {
                                abort_stair(&mut chunk_manager, &mut actor, starting_y, 1);
                                break;
                            }
                        }
                    }
                }
            } else {
                let prev_vel = actor.vel.length();
                let moved_y = move_y(&mut chunk_manager, &mut actor, (v.y - prev.y).signum());

                if !moved_y && (v.y - prev.y).signum() != 0 && actor.colliding.is_none() {
                    actor.colliding = Some(prev_vel);
                }
            }

            prev = v;
        }
    }
}

pub fn abort_stair(chunk_manager: &mut ChunkManager, actor: &mut Actor, starting_y: i32, dir: i32) {
    for _ in 0..(starting_y - actor.pos.y) {
        move_y(chunk_manager, actor, dir);
    }
}

pub fn move_x(chunk_manager: &mut ChunkManager, actor: &mut Actor, dir: i32) -> bool {
    //Check if we can move
    for y_off in 0..actor.height as i32 {
        let pos = actor.pos
            + if dir > 0 {
                // Moving right
                ivec2(actor.width as i32, y_off)
            } else {
                // Moving left
                ivec2(-1, y_off)
            };
        let chunk_pos = global_to_chunk(pos);

        // Check bounds and if it's free to move
        if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
            if atom.is_powder() || atom.is_solid() {
                actor.vel.x = 0.0;
                return false;
            }
        } else {
            actor.vel.x = 0.0;
            return false;
        }
    }

    actor.pos.x += dir;

    true
}

pub fn move_y(chunk_manager: &mut ChunkManager, actor: &mut Actor, dir: i32) -> bool {
    //Check if we can move
    for x_off in 0..actor.width as i32 {
        let pos = actor.pos
            + if dir > 0 {
                // Moving down
                ivec2(x_off, actor.height as i32)
            } else {
                // Moving up
                ivec2(x_off, -1)
            };
        let chunk_pos = global_to_chunk(pos);

        // Check bounds and if it's free to move
        if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
            if atom.is_powder() || atom.is_solid() {
                actor.vel.y = 0.0;
                return false;
            }
        } else {
            actor.vel.y = 0.0;
            return false;
        }
    }

    actor.pos.y += dir;

    true
}

pub struct ActorsPlugin;
impl Plugin for ActorsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                fill_actors
                    .before(chunk_manager_update)
                    .before(update_particles),
                unfill_actors.after(chunk_manager_update),
                update_actors.after(unfill_actors),
            )
                .run_if(in_state(GameState::Game)),
        );
    }
}
