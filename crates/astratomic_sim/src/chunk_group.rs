use std::ptr::NonNull;

use crate::prelude::*;

/// Lightweight 3x3 chunk neighborhood view holding raw pointers to the 9 chunk grids (72 bytes total,
/// zero heap allocations).
pub struct ChunkGroup<'a> {
    /// Pointers to the 3x3 chunk grids (`[Atom; CHUNK_LEN]`), indexed by `chunk_y * 3 + chunk_x` (`0..9`).
    /// Index `4` is the center chunk.
    pub chunks: [Option<NonNull<Atom>>; 9],
    /// Position of the center chunk (`chunks[4]`).
    pub center_pos: IVec2,
    pub _marker: std::marker::PhantomData<&'a mut Atom>,
}

// SAFETY: Within each of the 4 checkerboard passes, active center chunks are separated by 2 chunks
// along both axes and only access their own neighborhood.
unsafe impl Send for ChunkGroup<'_> {}

impl ChunkGroup<'_> {
    #[inline]
    pub fn group_to_chunk(center_pos: IVec2, group_idx: i32) -> IVec2 {
        let x_diff = group_idx % 3 - 1;
        let y_diff = group_idx / 3 - 1;
        ivec2(center_pos.x + x_diff, center_pos.y + y_diff)
    }

    #[inline]
    pub fn center_atom(&self, x: usize, y: usize) -> &Atom {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE);
        let ptr = self.chunks[4].expect("center chunk must exist");
        unsafe { &*ptr.as_ptr().add((y << 6) | x) }
    }

    #[inline]
    pub fn center_atom_mut(&mut self, x: usize, y: usize) -> &mut Atom {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE);
        let ptr = self.chunks[4].expect("center chunk must exist");
        unsafe { &mut *ptr.as_ptr().add((y << 6) | x) }
    }

    #[inline]
    pub fn get_local(&self, idx: (IVec2, i32)) -> Option<&Atom> {
        let (pos, group_idx) = idx;
        if !(0..9).contains(&group_idx)
            || pos.x < 0
            || pos.x >= CHUNK_LENGHT as i32
            || pos.y < 0
            || pos.y >= CHUNK_LENGHT as i32
        {
            return None;
        }
        self.get_global(local_to_global(idx))
    }

    #[inline]
    pub fn get_mut_local(&mut self, idx: (IVec2, i32)) -> Option<&mut Atom> {
        let (pos, group_idx) = idx;
        if !(0..9).contains(&group_idx)
            || pos.x < 0
            || pos.x >= CHUNK_LENGHT as i32
            || pos.y < 0
            || pos.y >= CHUNK_LENGHT as i32
        {
            return None;
        }
        self.get_mut_global(local_to_global(idx))
    }

    #[inline]
    pub fn get_global(&self, idx: IVec2) -> Option<&Atom> {
        const MIN_B: i32 = HALF_CHUNK_LENGHT as i32;
        const MAX_B: i32 = (CHUNK_LENGHT * 2 + HALF_CHUNK_LENGHT) as i32;
        if idx.x < MIN_B || idx.x >= MAX_B || idx.y < MIN_B || idx.y >= MAX_B {
            return None;
        }
        let gx = idx.x as usize;
        let gy = idx.y as usize;
        let group_idx = (gy >> 6) * 3 + (gx >> 6);
        let atom_idx = ((gy & 63) << 6) | (gx & 63);
        let ptr = self.chunks[group_idx]?;
        unsafe { Some(&*ptr.as_ptr().add(atom_idx)) }
    }

    #[inline]
    pub fn get_mut_global(&mut self, idx: IVec2) -> Option<&mut Atom> {
        const MIN_B: i32 = HALF_CHUNK_LENGHT as i32;
        const MAX_B: i32 = (CHUNK_LENGHT * 2 + HALF_CHUNK_LENGHT) as i32;
        if idx.x < MIN_B || idx.x >= MAX_B || idx.y < MIN_B || idx.y >= MAX_B {
            return None;
        }
        let gx = idx.x as usize;
        let gy = idx.y as usize;
        let group_idx = (gy >> 6) * 3 + (gx >> 6);
        let atom_idx = ((gy & 63) << 6) | (gx & 63);
        let ptr = self.chunks[group_idx]?;
        unsafe { Some(&mut *ptr.as_ptr().add(atom_idx)) }
    }
}

impl std::ops::Index<IVec2> for ChunkGroup<'_> {
    type Output = Atom;
    #[track_caller]
    #[inline]
    fn index(&self, idx: IVec2) -> &Self::Output {
        self.get_global(idx).expect("Invalid index position.")
    }
}

impl std::ops::IndexMut<IVec2> for ChunkGroup<'_> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, idx: IVec2) -> &mut Self::Output {
        self.get_mut_global(idx).expect("Invalid index position.")
    }
}

impl std::ops::Index<(IVec2, i32)> for ChunkGroup<'_> {
    type Output = Atom;
    #[track_caller]
    #[inline]
    fn index(&self, idx: (IVec2, i32)) -> &Self::Output {
        self.get_local(idx).expect("Invalid index position.")
    }
}

impl std::ops::IndexMut<(IVec2, i32)> for ChunkGroup<'_> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, idx: (IVec2, i32)) -> &mut Self::Output {
        self.get_mut_local(idx).expect("Invalid index position.")
    }
}

/// Transforms global 3x3 chunk position to local `(0..64, 0..64)` chunk position and group index `0..9`.
#[inline]
pub fn global_to_local(pos: IVec2) -> (IVec2, i32) {
    debug_assert!(
        (0..(CHUNK_LENGHT as i32 * 3)).contains(&pos.x)
            && (0..(CHUNK_LENGHT as i32 * 3)).contains(&pos.y),
        "Invalid position on global_to_local: {pos:?}"
    );
    let gx = pos.x as usize;
    let gy = pos.y as usize;
    let chunk_x = (gx >> 6) as i32;
    let chunk_y = (gy >> 6) as i32;
    let local_x = (gx & 63) as i32;
    let local_y = (gy & 63) as i32;
    (IVec2::new(local_x, local_y), chunk_y * 3 + chunk_x)
}

/// Transforms local `(0..64, 0..64)` chunk position and group index `0..9` to global 3x3 chunk position.
#[inline]
pub fn local_to_global(pos: (IVec2, i32)) -> IVec2 {
    debug_assert!(
        (0..CHUNK_LENGHT as i32).contains(&pos.0.x)
            && (0..CHUNK_LENGHT as i32).contains(&pos.0.y)
            && (0..9).contains(&pos.1),
        "Invalid position on local_to_global: {pos:?}"
    );
    let chunk_index = pos.1;
    let chunk_x = chunk_index % 3;
    let chunk_y = chunk_index / 3;
    IVec2::new(pos.0.x + (chunk_x << 6), pos.0.y + (chunk_y << 6))
}

fn merge_rect_map(dst: &mut HashMap<IVec2, URect>, src: HashMap<IVec2, URect>) {
    for (chunk, rect) in src {
        if let Some(existing) = dst.get_mut(&chunk) {
            existing.min = existing.min.min(rect.min);
            existing.max = existing.max.max(rect.max);
        } else {
            dst.insert(chunk, rect);
        }
    }
}

/// Prepares and updates all matching chunk groups for a single `(x_toff, y_toff)` checkerboard pass.
pub fn update_chunk_groups(
    chunks: &mut HashMap<IVec2, Chunk>,
    (x_toff, y_toff): (i32, i32),
    dirty_rects: &HashMap<IVec2, URect>,
    manager_pos: IVec2,
    new_dirty_rects: &mut HashMap<IVec2, URect>,
    render_dirty_rects: &mut HashMap<IVec2, URect>,
    (dt, materials, tick, world_seed): (u8, &Materials, u64, u32),
) {
    puffin::profile_function!();

    let mx_parity = manager_pos.x.abs() & 1;
    let my_parity = manager_pos.y.abs() & 1;

    let active_centers: Vec<(IVec2, URect)> = dirty_rects
        .iter()
        .filter_map(|(&chunk_pos, &rect)| {
            let same_x = (chunk_pos.x + x_toff + mx_parity).rem_euclid(2) == 0;
            let same_y = (chunk_pos.y + y_toff + my_parity).rem_euclid(2) == 0;
            (same_x && same_y && chunks.contains_key(&chunk_pos)).then_some((chunk_pos, rect))
        })
        .collect();

    if active_centers.is_empty() {
        return;
    }

    // 1. Perform Copy-on-Write (`Arc::make_mut`) on all chunks in the 3x3 neighborhoods up front
    // so every active `ChunkGrid` is uniquely owned and has a stable memory address.
    for &(center_pos, _) in &active_centers {
        for gy in -1..=1 {
            for gx in -1..=1 {
                if let Some(chunk) = chunks.get_mut(&(center_pos + ivec2(gx, gy))) {
                    let _ = std::sync::Arc::make_mut(&mut chunk.grid);
                }
            }
        }
    }

    // 2. Build the 72-byte `ChunkGroup` pointer tables on the main thread without touching `HashMap`
    // from worker threads.
    let mut groups: Vec<(ChunkGroup<'_>, URect)> = Vec::with_capacity(active_centers.len());
    for (center_pos, rect) in active_centers {
        let mut ptrs: [Option<NonNull<Atom>>; 9] = [None; 9];
        for gy in 0..3_i32 {
            for gx in 0..3_i32 {
                let pos = center_pos + ivec2(gx - 1, gy - 1);
                if let Some(chunk) = chunks.get_mut(&pos) {
                    let raw_ptr = std::sync::Arc::make_mut(&mut chunk.grid).atoms.as_mut_ptr();
                    ptrs[(gy * 3 + gx) as usize] = NonNull::new(raw_ptr);
                }
            }
        }
        groups.push((
            ChunkGroup {
                chunks: ptrs,
                center_pos,
                _marker: std::marker::PhantomData,
            },
            rect,
        ));
    }

    // 3. For small counts, run directly on the current thread without thread-spawn overhead.
    if groups.len() <= 2 {
        for (group, rect) in &mut groups {
            update_chunks(
                &mut UpdateChunksType {
                    group,
                    dirty_update_rects: new_dirty_rects,
                    dirty_render_rects: render_dirty_rects,
                    materials,
                    tick,
                    world_seed,
                },
                dt,
                rect,
            );
        }
        return;
    }

    // 4. Partition work across bounded worker threads and merge thread-local dirty rect maps.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let batch_size = groups.len().div_ceil(threads).max(1);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for slice in groups.chunks_mut(batch_size) {
            handles.push(s.spawn(move || {
                let mut local_update = HashMap::with_capacity(slice.len() * 2);
                let mut local_render = HashMap::with_capacity(slice.len() * 2);
                for (group, rect) in slice {
                    update_chunks(
                        &mut UpdateChunksType {
                            group,
                            dirty_update_rects: &mut local_update,
                            dirty_render_rects: &mut local_render,
                            materials,
                            tick,
                            world_seed,
                        },
                        dt,
                        rect,
                    );
                }
                (local_update, local_render)
            }));
        }

        for handle in handles {
            if let Ok((local_update, local_render)) = handle.join() {
                merge_rect_map(new_dirty_rects, local_update);
                merge_rect_map(render_dirty_rects, local_render);
            }
        }
    });
}
