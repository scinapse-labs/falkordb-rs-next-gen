#!/bin/bash
if [[ "$(uname -s)" == "Darwin" ]]; then
  TARGET=libfalkordb.dylib
else
  TARGET=libfalkordb.so
fi

if [[ "$TARGET_DIR" == "" ]]; then
  if [[ "$RELEASE" == 1 ]]; then
    TARGET_DIR=target/release
  else
    TARGET_DIR=target/debug
  fi
fi

if [[ "$VERBOSE" == 1 ]]; then
  V=-v
else
  V=
fi

if [[ "$TESTS_FILE" == "" ]]; then
  TESTS_FILE=flow_tests_done.txt
fi

STOP_ON_FAILURE=""
PARALLELISM="--parallelism 8"
if [[ "$FAIL_FAST" == 1 ]]; then
	STOP_ON_FAILURE="--stop-on-failure"
	PARALLELISM="--parallelism 1"
fi

# Add test filter support
TEST_FILTER=()
if [[ "$TEST" != "" ]]; then
    TEST_FILTER=(-t "$TEST")
fi

set -x 
RLTest ${TEST_FILTER[@]:--f "$TESTS_FILE"} --module "$TARGET_DIR/$TARGET" --no-progress $PARALLELISM $STOP_ON_FAILURE --clear-logs --log-dir tests/flow/logs $V