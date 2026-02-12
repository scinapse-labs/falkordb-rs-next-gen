# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FalkorDB-rs is a Rust implementation of FalkorDB, a graph database that runs as a Redis module. It implements the Cypher query language using GraphBLAS sparse matrices for efficient graph storage and traversal.

## Build Commands

```bash
# Build (debug)
cargo build

# Build (release)
cargo build --release

# Format check
cargo fmt --all -- --check

# Lint (requires clang++)
CXX=clang++ cargo clippy --all-targets
```

## Testing

**Unit tests (Rust):**
```bash
cargo test -p graph
```

**E2E and function tests (Python):**
```bash
# Activate virtualenv first
source venv/bin/activate
pytest tests/test_e2e.py tests/test_functions.py -vv
```

**TCK (Technology Compatibility Kit) tests:**
```bash
# Run all passing TCK tests
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s

# Run specific TCK test subset
TCK_INCLUDE=tests/tck/features/expressions/list pytest tests/tck/test_tck.py -s
```

**MVCC and concurrency tests:**
```bash
pytest tests/test_mvcc.py tests/test_concurrency.py -vv
```

**Flow tests:**
```bash
./flow.sh
```

## Dependencies

Before building, GraphBLAS and RediSearch must be compiled and installed:
- GraphBLAS: `./graphblas.sh` or build manually with `make static CMAKE_OPTIONS='-DGRAPHBLAS_COMPACT=1 -DCMAKE_POSITION_INDEPENDENT_CODE=on'`
- RediSearch: `./redisearch.sh`

Python tests require a virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r tests/requirements.txt
```

## Architecture

### Crate Structure

- **`falkordb-rs`** (root `src/`) - Redis module integration, command handlers, response serialization
- **`graph`** (`graph/src/`) - Core graph database implementation, query processing, runtime

### Query Processing Pipeline

```text
Cypher Query String
       │
       ▼
┌─────────────────┐
│  Parser         │  cypher.rs - Hand-written recursive descent parser
│  (cypher.rs)    │  Produces: DynTree<ExprIR<Arc<String>>>
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Binder         │  binder.rs - Semantic analysis, name resolution
│  (binder.rs)    │  Resolves variable names to numeric IDs
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Planner        │  planner.rs - Converts AST to execution plan (IR tree)
│  (planner.rs)   │  IR operators: NodeByLabelScan, CondTraverse, Filter, etc.
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Optimizer      │  optimizer.rs - Index utilization, node-by-ID optimization
│  (optimizer.rs) │  Rewrites plan for better performance
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Runtime        │  runtime/runtime.rs - Pull-based iterator execution
│  (runtime/)     │  Evaluates IR tree, manages pending mutations
└─────────────────┘
```

### Graph Storage Model (`graph/src/graph/`)

- **graph.rs** - Main Graph struct: adjacency matrices, label matrices, relationship tensors, attribute stores
- **matrix.rs** - GraphBLAS sparse matrix wrapper (FFI to C library)
- **vector.rs** - GraphBLAS sparse vector wrapper
- **tensor.rs** - 3D sparse structure for relationship storage (src × dst × edge_id)
- **versioned_matrix.rs** - Copy-on-write matrix with version tracking
- **mvcc_graph.rs** - MVCC wrapper providing snapshot isolation for reads, serialized writes
- **attribute_store.rs** - Property storage for nodes and relationships
- **block_vec.rs** - Block-allocated vector for efficient memory usage

### Concurrency Model

- **Read queries**: Execute concurrently on thread pool using MVCC snapshots
- **Write queries**: Serialized through mpsc channel, processed by single thread
- **ThreadedGraph** (`src/lib.rs`): Wraps MvccGraph with write queue for Redis integration

### Redis Commands

- `GRAPH.QUERY` - Read/write Cypher queries (blocks client, executes on thread pool)
- `GRAPH.RO_QUERY` - Read-only queries (errors on write operations)
- `GRAPH.EXPLAIN` - Show execution plan without executing
- `GRAPH.DELETE` - Delete a graph
- `GRAPH.LIST` - List all graphs
- `GRAPH.MEMORY` - Get memory usage

### Key Types

- **Value** (`runtime/value.rs`) - Runtime value enum: Null, Bool, Int, Float, String, List, Map, Node, Relationship, Path, Point, etc.
- **IR** (`planner.rs`) - Intermediate representation nodes for execution plan
- **Env** (`runtime/value.rs`) - Variable bindings flowing through pipeline
- **Pending** (`runtime/pending.rs`) - Batched mutations for transactional semantics

## Code Style

- Uses `rustfmt` with vertical function parameter layout (`fn_params_layout = "Vertical"`)
- The parser (`cypher.rs`) is hand-written, not generated from Cypher.g4 grammar file
- GraphBLAS FFI bindings are in `graph/src/graph/GraphBLAS.rs` (auto-generated, do not edit)

## Configuration

- `CACHE_SIZE` - Query plan cache size (default: 25, range: 0-1000)
- `IMPORT_FOLDER` - CSV import path (default: `/var/lib/FalkorDB/import/`)
