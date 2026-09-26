#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:?usage: build.sh OUTPUT_BINARY}
src=vendor/BigMoeOnEdge/third_party/llama.cpp
libs=$(realpath vendor/BigMoeOnEdge/build/bin)
g++ -std=c++17 -O2 -fopenmp -pthread -Wall -Wextra scripts/resumable/probe.cpp \
  -I"$src/include" -I"$src/ggml/include" -I"$src/vendor" \
  -L"$libs" -Wl,-rpath,"$libs" -lllama -lggml -lggml-base -lggml-cpu -o "$out"
