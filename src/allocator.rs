use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::thread_local;

pub struct ThreadCountingAllocator;

thread_local! {
    // Per-thread allocation counter
    static THREAD_ALLOCATED: Cell<usize> = Cell::new(0);
    static THREAD_DEALLOCATED: Cell<usize> = Cell::new(0);

    // Per-thread tracking flag
    static TRACKING_ENABLED: Cell<bool> = Cell::new(true);
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        let ptr = unsafe {
            redis_module::alloc::RedisAlloc::alloc(&redis_module::alloc::RedisAlloc, layout)
        };
        if !ptr.is_null() {
            TRACKING_ENABLED.with(|enabled| {
                if enabled.get() {
                    THREAD_ALLOCATED.with(|c| c.set(c.get() + layout.size()));
                }
            });
        }
        ptr
    }

    unsafe fn dealloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
    ) {
        unsafe {
            redis_module::alloc::RedisAlloc::dealloc(&redis_module::alloc::RedisAlloc, ptr, layout);
        };
        TRACKING_ENABLED.with(|enabled| {
            if enabled.get() {
                THREAD_DEALLOCATED.with(|c| c.set(c.get() + layout.size()));
            }
        });
    }
}

// Helper functions for the current thread
pub fn enable_tracking() {
    TRACKING_ENABLED.with(|flag| flag.set(true));
}

pub fn disable_tracking() {
    TRACKING_ENABLED.with(|flag| flag.set(false));
}

pub fn current_thread_usage() -> (usize, usize) {
    (
        THREAD_ALLOCATED.with(std::cell::Cell::get),
        THREAD_DEALLOCATED.with(std::cell::Cell::get),
    )
}

// Reset the counter for the current thread
pub fn reset_counter() {
    THREAD_ALLOCATED.with(|c| c.set(0));
    THREAD_DEALLOCATED.with(|c| c.set(0));
}
