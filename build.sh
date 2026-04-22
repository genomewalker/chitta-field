#!/usr/bin/env bash
# Build chitta-field with the correct Rust toolchain and system linker.
# Run from any directory: ./build.sh [cargo args...]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER
export PYO3_PYTHON=/maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/python3
export PATH="/usr/bin:$HOME/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH"
exec cargo "$@"
