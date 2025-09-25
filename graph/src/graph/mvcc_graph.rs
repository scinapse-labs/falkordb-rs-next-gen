use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::graph::graph::Graph;

pub struct MvccGraph {
    graph: Arc<AtomicRefCell<Graph>>,
}

unsafe impl Send for MvccGraph {}
unsafe impl Sync for MvccGraph {}

impl MvccGraph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
    ) -> Self {
        Self {
            graph: Arc::new(AtomicRefCell::new(Graph::new(n, e, cache_size, 0))),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    #[must_use]
    pub fn write(&self) -> Arc<AtomicRefCell<Graph>> {
        Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()))
    }

    pub fn commit(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) {
        if self.graph.borrow().version + 1 == new_graph.borrow().version {
            new_graph.borrow_mut().wait();
            self.graph = new_graph;
        } else {
            todo!();
        }
    }
}
