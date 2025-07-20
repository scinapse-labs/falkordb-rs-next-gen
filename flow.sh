cargo build
if [ Darwin == "$(uname -s)" ]; then
  RLTest -f flow_tests_done.txt --module target/debug/libfalkordb.dylib --no-progress --parallelism 8 --clear-logs --log-dir tests/flow/logs
else
  RLTest -f flow_tests_done.txt --module target/debug/libfalkordb.so --no-progress --parallelism 8 --clear-logs --log-dir tests/flow/logs
fi