//! Thread pool for parallel query execution.
//!
//! This module provides a global thread pool used to execute queries off
//! the Redis main thread. This prevents long-running queries from blocking
//! Redis command processing.
//!
//! ## Architecture
//!
//! ```text
//! Redis Main Thread                Thread Pool
//!       │                              │
//!   GRAPH.QUERY ───spawn()───→  [Worker 1] → execute query
//!       │                       [Worker 2]
//!   (continues)                 [Worker N]
//!       │                              │
//!   BlockedClient ←────────────── result
//! ```
//!
//! ## Thread Affinity
//!
//! Jobs can optionally specify a worker index for affinity (useful when
//! a query needs to run on the same thread as related work).

use std::thread::{self, JoinHandle};

use crossfire::{BlockingTxTrait, Tx, spsc::Array};
use once_cell::sync::OnceCell;

/// A closure that can be sent to a worker thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A pool of worker threads for executing jobs.
struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    sender: Vec<Tx<Array<Job>>>,
}

unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let mut sender = Vec::with_capacity(size);
        for _ in 0..size {
            let (tx, rx) = crossfire::spsc::bounded_blocking::<Job>(1024);
            sender.push(tx);
            let worker = thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            });
            workers.push(worker);
        }
        Self { workers, sender }
    }

    pub fn spawn<F>(
        &self,
        job: F,
        idx: Option<usize>,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let sender = if let Some(i) = idx {
            &self.sender[i % self.workers.len()]
        } else {
            let mut min_tx = &self.sender[0];
            let mut min_len = usize::MAX;
            for tx in &self.sender {
                if tx.is_empty() {
                    return tx.send(Box::new(job)).unwrap();
                }
                let len = tx.len();
                if len < min_len {
                    min_len = len;
                    min_tx = tx;
                }
            }
            min_tx
        };
        sender.send(Box::new(job)).unwrap();
    }
}

static GLOBAL_THREAD_POOL: OnceCell<ThreadPool> = OnceCell::new();

pub fn spawn<F>(
    job: F,
    idx: Option<usize>,
) where
    F: FnOnce() + Send + 'static,
{
    GLOBAL_THREAD_POOL
        .get_or_init(|| ThreadPool::new(num_cpus::get()))
        .spawn(job, idx);
}
