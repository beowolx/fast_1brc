# Fast 1BRC (Rust, std-only)

## Overview

This repository contains a Rust implementation of [The One Billion Row Challenge](https://github.com/gunnarmorling/1brc).

Input format:

```text
<station-name>;<temperature-with-1-decimal>
```

Output format:

```text
<station-name>;<min>;<mean>;<max>
```

Stations are emitted in alphabetical order.

## Challenge Compliance

The workspace is now fully standard-library-only.

- `fast_1brc`: no external crates
- `generate-dataset`: no external crates

Validation command:

```bash
cargo tree -e normal
```

Expected output is only workspace packages (`fast_1brc` and `generate-dataset`) with no third-party dependencies.

## Quick Start

### 1. Generate the benchmark dataset

```bash
cargo run --release --package generate-dataset 1000000000
```

This produces `measurements.txt` in the repo root.

### 2. Run the processor

```bash
cargo build --release
/usr/bin/time -p target/release/fast_1brc >/dev/null
```

### 3. Optional tuning knobs

- `FAST_1BRC_THREADS` (default: `available_parallelism()`)
- `FAST_1BRC_CHUNK_MB` (default: `4`)

Example:

```bash
FAST_1BRC_THREADS=10 FAST_1BRC_CHUNK_MB=4 target/release/fast_1brc
```

## Implementation Notes

### Processor (`src/main.rs`)

- Uses `FileExt::read_at` to read independent file chunks in parallel.
- Adds overlap between chunks and trims boundaries at newline to avoid double-counting or truncation.
- Assigns contiguous chunk ranges per worker to keep disk access more sequential.
- Parses temperatures as fixed-point tenths (`i16`) to avoid floating-point parsing in the hot path.
- Aggregates per-thread stats in local maps, then merges into a global map.
- Uses an in-file `FxHasher64` implementation via `BuildHasherDefault`.
- Sorts final station keys and prints `min/mean/max` with one decimal.

## Correctness

Checked with:

- Official upstream `1brc` sample suite via `test.sh` and `tocsv.sh` normalization.
- Real-data output hash match against a previously validated output.

To reproduce the official sample validation:

```bash
cargo build --release

git clone --depth 1 https://github.com/gunnarmorling/1brc.git /tmp/onebrc_upstream
cat >/tmp/onebrc_upstream/calculate_average_fast1brc.sh <<SH
#!/bin/sh
exec "$(pwd)/target/release/fast_1brc"
SH
chmod +x /tmp/onebrc_upstream/calculate_average_fast1brc.sh

/tmp/onebrc_upstream/test.sh fast1brc 'src/test/resources/samples/*.txt'
```

Expected result: all sample files validate without diffs (currently `12/12` passing).

## Benchmark Snapshot

Machine: macOS (Apple Silicon), real `measurements.txt` (~1B rows, generated with `generate-dataset`)

Command:

```bash
/usr/bin/time -p target/release/fast_1brc >/dev/null
```

Recent runs (after short cooldown):

- `4.22s`
- `4.09s`
- `4.13s`

Median: `4.13s`
