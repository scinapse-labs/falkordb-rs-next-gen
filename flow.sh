#!/bin/bash
if [[ "$(uname -s)" == "Darwin" ]]; then
  TARGET=libfalkordb.dylib
else
  TARGET=libfalkordb.so
fi

if [[ "$TARGET_DIR" == "" ]]; then
  if [[ "$RELEASE" == 1 ]]; then
    TARGET_DIR=target/release
    cargo build -r
  else
    TARGET_DIR=target/debug
    cargo build
  fi
fi

if [[ "$VERBOSE" == 1 ]]; then
  V=-v
else
  V=
fi

if [[ "$TESTS_FILE" == "" ]]; then
  TESTS=flow_tests_done.txt
fi

RLTest -f $TESTS --module $TARGET_DIR/$TARGET --no-progress --parallelism 8 --clear-logs --log-dir tests/flow/logs $V