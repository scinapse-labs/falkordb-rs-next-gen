# /flow - Flow Test Runner

Run the flow test suite which tests query flows against the running database.

## Usage

- `/flow` - Run all flow tests

## Instructions

When the user invokes `/flow`:

```bash
./flow.sh
```

## Notes

- Flow tests are located in `tests/flow/` directory
- These tests verify complete query execution flows
- Progress is tracked in `flow_tests_done.txt` and `flow_tests_todo.txt`
- The flow.sh script handles test execution and result tracking
