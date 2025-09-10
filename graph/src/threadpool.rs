use std::{
    sync::mpsc::{Sender, channel},
    thread::{self, JoinHandle},
};

use once_cell::sync::OnceCell;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    sender: Vec<Sender<Job>>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let mut sender: Vec<Sender<Job>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (tx, rx) = channel();
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
        let idx = idx.unwrap_or_else(|| rand::random::<u32>() as usize) % self.workers.len();
        self.sender[idx].send(Box::new(job)).unwrap();
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
