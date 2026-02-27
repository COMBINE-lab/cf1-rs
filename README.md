# cf1-rs

A fast, parallel Rust implementation of the [Cuttlefish](https://github.com/COMBINE-lab/cuttlefish) algorithm for constructing compacted reference de Bruijn graphs (cdBGs).

## Overview

cf1-rs builds a compacted de Bruijn graph from reference sequences (genomes, transcriptomes) using a 5-phase pipeline:

1. **Minimizer counting** — SIMD-accelerated canonical minimizer extraction
2. **Super k-mer routing** — Parallel partitioning of sequences into minimizer-based bins using 2-bit packed encoding
3. **MPHF construction** — Global minimal perfect hash function via fingerprint-based hashing (BBHash), with parallel multi-threaded construction from disk
4. **DFA classification** — Lock-free vertex classification using atomic compare-and-swap on a compact state vector
5. **Unitig extraction** — Parallel traversal and output in GFA-reduced format

cf1-rs supports k-mer lengths from 1 to 63 (odd values) using const-generic k-mer types with `u64` storage (k <= 32) or `u128` storage (k <= 63).

## Installation

```bash
cargo install cf1-rs
```

Or build from source:

```bash
git clone https://github.com/COMBINE-lab/cf1-rs.git
cd cf1-rs
cargo build --release
```

## Usage

```bash
# Build a cdBG from a single FASTA file
cf1-rs build -s genome.fa.gz -k 31 -t 8 -o output/k31_dbg

# Build from a transcriptome with short sequence tracking
cf1-rs build -s transcripts.fa.gz -k 31 -t 4 -o output/k31_dbg \
    --track-short-seqs --poly-N-stretch

# Specify working directory and memory budget
cf1-rs build -s genome.fa.gz -k 31 -t 8 -o output/k31_dbg \
    -w /tmp/cf1_work --memory-budget 8.0
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-s` | Input FASTA/FASTQ file (optionally gzipped) | — |
| `-l` | File listing input paths (one per line) | — |
| `-d` | Directory containing input sequence files | — |
| `-k` | K-mer length (odd, 1–63) | 31 |
| `-t` | Number of threads | 1 |
| `-o` | Output file prefix | — |
| `-f` | Output format (0=FASTA, 1=GFA1, 2=GFA2, 3=GFA-reduced) | 3 |
| `-w` | Working directory for temporary files | output dir |
| `--memory-budget` | Memory budget in GB for MPHF construction | 4.0 |
| `--track-short-seqs` | Report sequences shorter than k | off |
| `--poly-N-stretch` | Handle poly-N gaps in tiling output | off |
| `--collate-output-in-mem` | Buffer output per-thread for ordered writes | off |
| `--num-bins` | Number of minimizer partition bins | 128 |

### Output

cf1-rs produces three output files:

- `<prefix>.cf_seg` — Unitig segments (GFA S-lines)
- `<prefix>.cf_seq` — Per-sequence unitig tilings (GFA P-lines)
- `<prefix>.json` — Summary statistics (vertex count, unitig count, length distribution)

## Performance

cf1-rs is designed for high performance on both small transcriptomes and large genomes. On the human genome (GRCh38, ~3.1 Gbp) with k=31 and 4 threads, cf1-rs constructs the compacted dBG in under 15 minutes, using about 6GB RAM.

## License

BSD-3-Clause
