use bevy::render::{render_asset::RenderAssetUsages, render_resource::*};
use noise::*;
use std::collections::HashSet;

use crate::{prelude::*, rigidbody};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(with = "BigArray")]
    pub atoms: [Atom; CHUNK_LEN],

    #[serde(skip)]
    pub texture: Handle<Image>,
    #[serde(skip)]
    pub entity: Option<Entity>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            atoms: [Atom::default(); CHUNK_LEN],
            texture: Handle::default(),
            entity: None,
        }
    }
}

#[derive(Clone, Copy)]
struct YCoord;
impl NoiseFn<f64, 2> for YCoord {
    fn get(&self, point: [f64; 2]) -> f64 {
        point[1]
    }
}

impl Chunk {
    pub fn new(
        texture: Handle<Image>,
        index: IVec2,
        materials: &Materials,
        generator: &Generator,
    ) -> Chunk {
        let mut atoms = [Atom::default(); CHUNK_LEN];

        let air_limit = -1000.;

        let (scale, cave) = (generator.1, generator.0.clone());

        let cave: Multiply<f64, RidgedMulti<SuperSimplex>, Constant, 2> =
            Multiply::new(cave, Constant::new(0.7));

        let ground_noise = BasicMulti::<SuperSimplex>::new(generator.2);
        let ground = Add::new(
            ScalePoint::new(ground_noise)
                .set_y_scale(0.3)
                .set_x_scale(0.3),
            YCoord,
        );

        let cave_start = 0.5;
        let falloff = 0.3;
        let control = Add::new(ground.clone(), Constant::new(-cave_start));
        let generator = Select::new(cave, ground.clone(), control.clone())
            .set_bounds(air_limit, cave_start) // For y values within this range, use the ground noise.
            .set_falloff(falloff);

        for x in 0..CHUNK_LENGHT {
            for y in 0..CHUNK_LENGHT {
                let pos = [
                    (index.x as f64 * CHUNK_LENGHT as f64 + x as f64) / scale,
                    (index.y as f64 * CHUNK_LENGHT as f64 + y as f64) / scale,
                ];
                let noise = generator.get(pos);
                let i = y * CHUNK_LENGHT + x;

                /*atoms[i].color = [
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    255,
                ];*/
                let id = if (air_limit..(cave_start - falloff)).contains(&control.get(pos)) {
                    // Ground layer
                    match (noise + 1.) / 2. {
                        ..0.2 => continue,
                        0.2..0.21 => 6,
                        0.21..0.5 => 7,
                        0.5..0.75 => 10,
                        0.75.. => 4,
                        _ => panic!("{noise}"),
                    }
                } else {
                    // Cave layer
                    match (noise + 1.) / 2. {
                        ..0.2 => continue,
                        0.2.. => 8,
                        _ => panic!("{noise}"),
                    }
                };

                atoms[i] = Atom::new(id, materials);
                /*atoms[i].color = [
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    ((noise as f32 + 1.) / 2. * 100.) as u8,
                    255,
                ];*/
            }
        }

        Chunk {
            atoms,
            texture,
            entity: None,
        }
    }

    pub fn new_image() -> Image {
        Image::new(
            Extent3d {
                height: CHUNK_LENGHT as u32,
                width: CHUNK_LENGHT as u32,
                ..Default::default()
            },
            TextureDimension::D2,
            vec![0; CHUNK_LENGHT.pow(2) * 4],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    //This uses the CPU and is not used in-game anymore
    pub fn _update_image_positions(&self, image: &mut Image, positions: &HashSet<IVec2>) {
        for pos in positions {
            let pixel_index = (pos.y as usize * CHUNK_LENGHT + pos.x as usize) * 4;
            image.data[pixel_index..pixel_index + 4].copy_from_slice(&self.atoms[pos.d1()].color);
        }
    }

    pub fn update_all(&self, image: &mut Image) {
        // Hot path: initializing a whole chunk texture during load.
        // Avoid allocating/iterating a HashSet of every position.
        debug_assert_eq!(image.data.len(), CHUNK_LENGHT * CHUNK_LENGHT * 4);

        for y in 0..CHUNK_LENGHT {
            for x in 0..CHUNK_LENGHT {
                let atom_index = y * CHUNK_LENGHT + x;
                let pixel_index = atom_index * 4;
                image.data[pixel_index..pixel_index + 4]
                    .copy_from_slice(&self.atoms[atom_index].color);
            }
        }
    }

    pub fn get_collider(&self) -> Option<Collider> {
        rigidbody::get_collider(&self.get_values(), CHUNK_LENGHT as u32, CHUNK_LENGHT as u32)
    }

    pub fn get_values(&self) -> Vec<f64> {
        let mut values = vec![];

        for row in self.atoms.chunks(CHUNK_LENGHT) {
            for atom in row {
                if atom.is_solid() {
                    values.push(1.)
                } else {
                    values.push(0.)
                }
            }
        }

        values
    }
}
