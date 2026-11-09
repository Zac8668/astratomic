//! 4-pass deterministic cellular automata simulation step.

use async_channel::Sender;

use crate::{
    atom::Materials,
    chunk_group::ChunkGroup,
    world::{DeferredDirtyRectUpdate, URect},
};

pub struct UpdateChunksType<'a> {
    pub group: &'a mut ChunkGroup<'a>,
    pub dirty_update_rect_send: &'a Sender<DeferredDirtyRectUpdate>,
    pub dirty_render_rect_send: &'a Sender<DeferredDirtyRectUpdate>,
    pub materials: &'a Materials,
}

pub fn update_chunks(_chunks: &mut UpdateChunksType, _dt: u8, _dirty_rect: &URect) {}

/// Scoped task runner backed by `std::thread::Scope`.
pub struct Scope<'scope, 'env: 'scope, T> {
    inner: &'scope std::thread::Scope<'scope, 'env>,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<'scope, 'env: 'scope, T: Send + 'scope> Scope<'scope, 'env, T> {
    pub fn new(inner: &'scope std::thread::Scope<'scope, 'env>) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: std::future::Future<Output = T> + Send + 'scope,
    {
        self.inner.spawn(move || futures_lite::future::block_on(f));
    }
}
