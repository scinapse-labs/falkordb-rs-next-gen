# /test - Smart Test Runner

Run the appropriate tests based on what changed or what the user specifies.

## Usage

- `/test` - Run all unit tests
- `/test unit` - Run Rust unit tests only
- `/test e2e` - Run Python E2E tests
- `/test functions` - Run Python function tests
- `/test mvcc` - Run MVCC tests
- `/test concurrency` - Run concurrency tests

## Instructions

When the user invokes `/test`, determine which tests to run:

1. **Unit tests (default)**: Run Rust unit tests
   ```bash
   cargo test -p graph
   ```

2. **E2E tests**: Run Python end-to-end tests
   ```bash
   source venv/bin/activate && pytest tests/test_e2e.py -vv
   ```

3. **Function tests**: Run Python function tests
   ```bash
   source venv/bin/activate && pytest tests/test_functions.py -vv
   ```

4. **MVCC tests**: Run MVCC concurrency tests
   ```bash
   source venv/bin/activate && pytest tests/test_mvcc.py -vv
   ```

5. **Concurrency tests**: Run concurrency tests
   ```bash
   source venv/bin/activate && pytest tests/test_concurrency.py -vv
   ```

If no argument is provided, run the Rust unit tests (`cargo test -p graph`).

If tests fail, analyze the output and help the user understand what went wrong.
