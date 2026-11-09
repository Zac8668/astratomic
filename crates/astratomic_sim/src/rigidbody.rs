use contour::ContourBuilder;
use geo::{SimplifyVwPreserve, TriangulateEarcut};
use rapier2d::math::{Pose, Vector};
use rapier2d::prelude::{Collider, ColliderBuilder, SharedShape};

pub fn get_collider(values: &[f64], width: u32, height: u32) -> Option<Collider> {
    let c = ContourBuilder::new(width as usize, height as usize, false);
    let res = c.contours(values, &[0.5]).ok()?;

    let mut shapes = Vec::new();
    for contour in res {
        let geometry = contour.geometry().simplify_vw_preserve(&1.5);
        for polygon in geometry {
            for triangle in polygon.earcut_triangles() {
                let (v1, v2, v3) = (triangle.v1(), triangle.v2(), triangle.v3());
                let shape = SharedShape::triangle(
                    Vector::new(v1.x as f32, -v1.y as f32),
                    Vector::new(v2.x as f32, -v2.y as f32),
                    Vector::new(v3.x as f32, -v3.y as f32),
                );
                shapes.push((Pose::IDENTITY, shape));
            }
        }
    }

    if !shapes.is_empty() {
        Some(ColliderBuilder::compound(shapes).build())
    } else {
        None
    }
}
