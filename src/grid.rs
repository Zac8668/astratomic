use std::ops::Range;
use std::sync::Mutex;
use std::sync::{Arc, RwLock};
use std::{thread, vec};

use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use bevy::math::ivec2;
use bevy::prelude::*;
use bevy::sprite;

use crate::atom::State;
use crate::chunk::*;
use crate::consts::*;
use crate::grid_api::*;

use std::cmp;

#[derive(Component)]
pub struct Grid {
    pub chunks: Vec<Arc<RwLock<Chunk>>>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub dt: f32,
}

fn grid_setup(mut commands: Commands, windows: Query<&Window>, mut images: ResMut<Assets<Image>>) {
    let window = windows.single();
    let side_length = (CHUNK_SIZE * ATOM_SIZE) as f32;

    let (mut grid_width, mut grid_height) = (
        (window.width() / side_length).ceil() as usize,
        (window.height() / side_length).ceil() as usize,
    );

    //If chunks aren't even, make them
    if grid_width % 2 != 0 {
        grid_width += 1
    }
    if grid_height % 2 != 0 {
        grid_height += 1
    }

    println!("{} {}", grid_width, grid_height);

    let mut chunks = vec![];
    for y in 0..grid_height {
        for x in 0..grid_width {
            // Get image position
            let mut pos = Vec2::new(x as f32 * side_length, -(y as f32) * side_length);
            pos.x -= grid_width as f32 / 2. * side_length;
            pos.y += grid_height as f32 / 2. * side_length;

            //Get and spawn texture/chunk image
            let texture = images.add(Chunk::new_image());
            commands.spawn(SpriteBundle {
                texture: texture.clone(),
                sprite: Sprite {
                    anchor: sprite::Anchor::TopLeft,
                    ..Default::default()
                },
                transform: Transform::from_xyz(pos.x, pos.y, 0.),
                ..Default::default()
            });

            //Create chunk
            let chunk = Chunk::new(texture);

            //Update chunk image
            let image = images.get_mut(&chunk.texture).unwrap();
            chunk.update_all(image);

            chunks.push(Arc::new(RwLock::new(chunk)));
        }
    }

    let grid = Grid {
        chunks,
        grid_width,
        grid_height,
        dt: 0.,
    };
    commands.spawn(grid);
}

pub fn grid_update(mut grid: Query<&mut Grid>, mut images: ResMut<Assets<Image>>, time: Res<Time>) {
    let mut grid = grid.single_mut();

    grid.dt += time.delta_seconds();
    let dt = grid.dt;

    if dt < UPDATE_TIME {
        return;
    }

    let row_range = 0..grid.grid_width as i32;
    let column_range = 0..grid.grid_height as i32;

    // Get images
    let images_removed: Vec<(Handle<Image>, Arc<Mutex<Image>>)> = grid
        .chunks
        .iter()
        .map(|chunk| {
            (
                chunk.read().unwrap().texture.clone(),
                Arc::new(Mutex::new(
                    images
                        .remove(chunk.read().unwrap().texture.clone())
                        .unwrap(),
                )),
            )
        })
        .collect();

    let update_vec: Vec<bool> = grid
        .chunks
        .iter()
        .map(|chunk| chunk.read().unwrap().active)
        .collect();

    for chunk in &grid.chunks {
        chunk.write().unwrap().active = false;
    }

    // Run the 4 update steps in checker like pattern
    for y_thread_off in rand_range(0..2) {
        for x_thread_off in rand_range(0..2) {
            let mut handles = vec![];

            //Acess chunks
            for y in (y_thread_off..grid.grid_height).step_by(2) {
                for x in (x_thread_off..grid.grid_width).step_by(2) {
                    if !update_vec[y * grid.grid_width + x] {
                        continue;
                    }

                    let mut chunks = vec![];
                    // Get all 3x3 chunks for each chunk updating
                    for y_off in -1..=1 {
                        for x_off in -1..=1 {
                            if !column_range.contains(&(y as i32 + y_off))
                                || !row_range.contains(&(x as i32 + x_off))
                            {
                                chunks.push(None);
                                continue;
                            }

                            let index = ((y as i32 + y_off) * grid.grid_width as i32
                                + x as i32
                                + x_off) as usize;

                            chunks.push(Some((
                                Arc::clone(&grid.chunks[index]),
                                Arc::clone(&images_removed[index].1),
                            )));
                        }
                    }

                    let handle = thread::spawn(move || update_chunks(chunks, dt));
                    handles.push(handle);
                }
            }

            // Wait for update step to finish
            for handle in handles {
                handle.join().unwrap()
            }
        }
    }

    // Add images back after update
    for image in images_removed {
        images.set_untracked(
            image.0,
            Arc::try_unwrap(image.1).unwrap().into_inner().unwrap(),
        )
    }

    grid.dt -= UPDATE_TIME;
}

fn rand_range(vec: Range<usize>) -> Vec<usize> {
    let mut vec: Vec<usize> = vec.collect();
    vec.shuffle(&mut rand::thread_rng());
    vec
}

pub fn update_chunks(chunks: UpdateChunksType, dt: f32) {
    for y in rand_range(CHUNK_SIZE - 1..CHUNK_SIZE * 2 + 1) {
        for x in rand_range(CHUNK_SIZE - 1..CHUNK_SIZE * 2 + 1) {
            let pos = ivec2(x as i32, y as i32);

            if !dt_upable(&chunks, pos, dt) {
                continue;
            }

            let state;
            {
                let local_pos = global_to_local(pos);
                state = chunks[local_pos.1 as usize]
                    .clone()
                    .unwrap()
                    .0
                    .read()
                    .unwrap()
                    .atoms[local_pos.0.y as usize * CHUNK_SIZE + local_pos.0.x as usize]
                    .state
            }

            match state {
                State::Powder => update_powder(&chunks, pos, dt),
                State::Liquid => update_liquid(&chunks, pos, dt),
                _ => (),
            }
        }
    }
}

fn update_powder(chunks: &UpdateChunksType, pos: IVec2, dt: f32) {
    let fvel = get_fvel(chunks, pos);
    let fvel = cmp::min(
        fvel + (GRAVITY as f32 * rand::thread_rng().gen_range(0.5..=1.)) as u8,
        (TERM_VEL as f32 * rand::thread_rng().gen_range(0.5..=1.)) as u8,
    );

    for i in 1..=fvel {
        let down = get_state(chunks, pos + IVec2::Y * i as i32) == Some(State::Void);

        if !down && i == 1 {
            break;
        }

        if down {
            swap_global(
                chunks,
                pos + IVec2::Y * (i as i32 - 1),
                pos + IVec2::Y * i as i32,
                dt,
            );

            if i == fvel {
                let local = global_to_local(pos + IVec2::Y * i as i32);
                chunks[local.1 as usize]
                    .clone()
                    .unwrap()
                    .0
                    .write()
                    .unwrap()
                    .atoms[local.0.d1()]
                .fall_velocity = fvel;
                return;
            }
        } else {
            if get_fvel(chunks, pos + IVec2::Y * i as i32) == 0 {
                let local = global_to_local(pos + IVec2::Y * (i as i32 - 1));
                chunks[local.1 as usize]
                    .clone()
                    .unwrap()
                    .0
                    .write()
                    .unwrap()
                    .atoms[local.0.d1()]
                .fall_velocity = 0;
                return;
            } else {
                let local = global_to_local(pos + IVec2::Y * i as i32);
                chunks[local.1 as usize]
                    .clone()
                    .unwrap()
                    .0
                    .write()
                    .unwrap()
                    .atoms[local.0.d1()]
                .fall_velocity = fvel;

                let local = global_to_local(pos + IVec2::Y * (i as i32 - 1));
                chunks[local.1 as usize]
                    .clone()
                    .unwrap()
                    .0
                    .write()
                    .unwrap()
                    .atoms[local.0.d1()]
                .fall_velocity = fvel;
            }
            return;
        }
    }

    let mut downsides = vec![
        (
            get_state(chunks, pos + IVec2::Y + IVec2::NEG_X) == Some(State::Void)
                && get_state(chunks, pos + IVec2::NEG_X) == Some(State::Void),
            IVec2::Y + IVec2::NEG_X,
        ),
        (
            get_state(chunks, pos + IVec2::Y + IVec2::X) == Some(State::Void)
                && get_state(chunks, pos + IVec2::X) == Some(State::Void),
            IVec2::Y + IVec2::X,
        ),
    ];
    downsides.shuffle(&mut thread_rng());

    for downside in downsides {
        if downside.0 {
            swap_global(chunks, pos, pos + downside.1, dt);
            return;
        }
    }

    let local_pos = global_to_local(pos);
    chunks[local_pos.1 as usize]
        .clone()
        .unwrap()
        .0
        .write()
        .unwrap()
        .atoms[local_pos.0.d1()]
    .updated_at = dt;
}

fn update_liquid(chunks: &UpdateChunksType, pos: IVec2, dt: f32) {
    let down = get_state(chunks, pos + IVec2::Y) == Some(State::Void);
    if down {
        swap_global(chunks, pos, pos + IVec2::Y, dt);
        return;
    }

    let mut sides = vec![
        (
            get_state(chunks, pos + IVec2::NEG_X) == Some(State::Void),
            IVec2::NEG_X,
        ),
        (
            get_state(chunks, pos + IVec2::X) == Some(State::Void),
            IVec2::X,
        ),
    ];

    let mut downsides = vec![
        (
            get_state(chunks, pos + IVec2::Y + IVec2::NEG_X) == Some(State::Void),
            IVec2::Y + IVec2::NEG_X,
            sides[0].0,
        ),
        (
            get_state(chunks, pos + IVec2::Y + IVec2::X) == Some(State::Void),
            IVec2::Y + IVec2::X,
            sides[1].0,
        ),
    ];

    downsides.shuffle(&mut thread_rng());
    for downside in downsides {
        if downside.0 && downside.2 {
            swap_global(chunks, pos, pos + downside.1, dt);
            return;
        }
    }

    sides.shuffle(&mut thread_rng());
    for side in sides {
        if side.0 {
            swap_global(chunks, pos, pos + side.1, dt);
            return;
        }
    }

    let local_pos = global_to_local(pos);
    chunks[local_pos.1 as usize]
        .clone()
        .unwrap()
        .0
        .write()
        .unwrap()
        .atoms[local_pos.0.d1()]
    .updated_at = dt;
}

pub struct GridPlugin;
impl Plugin for GridPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(grid_setup).add_system(grid_update);
    }
}