use bones_framework::prelude::{IVec2, ivec2};

pub struct Line {
    start: IVec2,
    current: IVec2,
    end: IVec2,
    straight: bool,
    hor: bool,
    tan: f32,
    max: f32,
}

impl Line {
    #[inline]
    pub fn new(start: IVec2, vec: IVec2) -> Self {
        let end = start + vec;

        let dx = end.x - start.x;
        let dy = end.y - start.y;

        let hor = dx.abs() > dy.abs();
        let tan = if dx == 0 && dy == 0 {
            0.0
        } else if hor {
            dy as f32 / dx as f32
        } else {
            dx as f32 / dy as f32
        };
        let max = (1.0 - tan.abs()) / 2.0;

        Self {
            start,
            current: start,
            end,
            straight: dx.signum() == 0 || dy.signum() == 0,
            hor,
            tan,
            max,
        }
    }

    /// Returns all integer grid points from `start` to `end` inclusive.
    pub fn points_inclusive(start: IVec2, end: IVec2) -> Vec<IVec2> {
        let mut pts = vec![start];
        pts.extend(Self::new(start, end - start));
        pts
    }
}

impl Iterator for Line {
    type Item = IVec2;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;

        if self.current.x == self.end.x && self.current.y == self.end.y {
            return None;
        }

        if self.straight {
            self.current.y += dy.signum();
            self.current.x += dx.signum();
            return Some(self.current);
        }

        if self.hor {
            let ideal = self.start.y as f32 + (self.current.x - self.start.x) as f32 * self.tan;

            if (ideal - self.current.y as f32) * dy.signum() as f32 >= self.max {
                self.current.y += dy.signum();
            } else {
                self.current.x += dx.signum();
            }
        } else {
            let ideal = self.start.x as f32 + (self.current.y - self.start.y) as f32 * self.tan;

            if (ideal - self.current.x as f32) * dx.signum() as f32 >= self.max {
                self.current.x += dx.signum();
            } else {
                self.current.y += dy.signum();
            }
        }

        Some(self.current)
    }
}

/// Returns all grid points inside a disk of `radius` centered at `center`.
pub fn circle_points(center: IVec2, radius: i32) -> Vec<IVec2> {
    let r = radius.max(0);
    let r2 = r * r;
    let side = (2 * r + 1) as usize;
    let mut points = Vec::with_capacity(side * side);

    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r2 {
                points.push(center + ivec2(dx, dy));
            }
        }
    }

    points
}

pub fn _circle_points(center: IVec2, radius: i32) -> Vec<IVec2> {
    circle_points(center, radius)
}

/// Returns all grid points inside a square of half-extent `radius` centered at `center`.
pub fn square_points(center: IVec2, radius: i32) -> Vec<IVec2> {
    let r = radius.max(0);
    let side = (2 * r + 1) as usize;
    let mut points = Vec::with_capacity(side * side);

    for dy in -r..=r {
        for dx in -r..=r {
            points.push(center + ivec2(dx, dy));
        }
    }

    points
}
