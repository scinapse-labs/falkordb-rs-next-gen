# /tck - TCK Test Runner

Run Technology Compatibility Kit (TCK) tests for Cypher language compliance.

## Usage

- `/tck` - Run all passing TCK tests
- `/tck <path>` - Run specific TCK feature tests

## Instructions

When the user invokes `/tck`:

1. **All passing TCK tests** (no argument):
   ```bash
   source venv/bin/activate && TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s
   ```

2. **Specific feature tests** (with path argument):
   ```bash
   source venv/bin/activate && TCK_INCLUDE=<path> pytest tests/tck/test_tck.py -s
   ```

   Common paths:
   - `tests/tck/features/expressions/list` - List expression tests
   - `tests/tck/features/expressions/map` - Map expression tests
   - `tests/tck/features/clauses/match` - MATCH clause tests
   - `tests/tck/features/clauses/return` - RETURN clause tests

## Notes

- TCK tests verify Cypher language compliance against the openCypher spec
- `tck_done.txt` contains the list of passing test scenarios
- Tests that are not yet passing are tracked in `flow_tests_todo.txt`
