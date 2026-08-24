#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${project_dir}/public/engine"

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup is required to build the browser engine" >&2
  exit 2
fi

cargo_bin="$(rustup which cargo)"
rust_bin_dir="$(dirname "${cargo_bin}")"
toolchain_root="$(dirname "${rust_bin_dir}")"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen CLI is required; install version 0.2.127" >&2
  exit 2
fi

rustup target add wasm32-unknown-unknown >/dev/null
mkdir -p "${output_dir}"

# rust-lld on macOS needs the Rust toolchain's LLVM dylib discoverable at link time.
export DYLD_LIBRARY_PATH="${toolchain_root}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export PATH="${rust_bin_dir}:${PATH}"

"${cargo_bin}" build \
  --manifest-path "${project_dir}/engine-rs/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --features wasm \
  --lib \
  --release

wasm-bindgen \
  "${project_dir}/engine-rs/target/wasm32-unknown-unknown/release/pathagon_engine.wasm" \
  --target web \
  --out-dir "${output_dir}" \
  --no-typescript

echo "Built Rust/WASM engine in ${output_dir}"
