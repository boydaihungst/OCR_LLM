#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_proto_files>"
    exit 1
fi

PROTO_PATH="$1"

mkdir -p py_lens
protoc "--proto_path=$PROTO_PATH" -I $PROTO_PATH --python_betterproto_out=. $PROTO_PATH/*.proto