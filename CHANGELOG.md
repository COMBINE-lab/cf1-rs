# Changelog

## Unreleased

Performance:

- Read the current RSS that each pipeline phase logs straight from the kernel
  (`/proc/self/statm` on Linux, `proc_pidinfo` on macOS) instead of spawning
  `ps`. On a busy many-core host each `ps` took about 0.45 s, so every run,
  however small its input, spent seconds just logging RSS; salmon's index build
  on a 12-transcript input went from 4.5 s to 0.4 s there. Other platforms log
  0 for the current RSS. Output is unchanged.

## 0.6.1 — 2026-11-18

Performance:

- Batch small references during phase-5 output, amortizing buffer allocation
  and synchronization while preserving ordered tilings (#4).
- Reuse previously walked unitigs within each small-reference batch. Cache
  only unitigs of at least 16 k-mers, store lengths as `usize`, and retain the
  eager batched walk for large-reference chunks (#5).
- Pack super k-mers four bases per output byte with the same representation
  as before (#6).

Memory-budget behavior:

- Use the existing MPHF memory budget and key width to select in-memory or
  streaming construction (#7). This is a soft construction budget in GiB,
  not a bound on process RSS.
- Account for the offline builder's approximately 256 MiB of fixed buffers.
  Budgets below 0.25 GiB, including zero, use that floor for regime selection
  and emit a warning, avoiding a slower and larger mode for small inputs.
- Large inputs may now choose streaming at the default 4 GiB setting,
  especially with 128-bit keys. This reduces peak memory at the cost of more
  construction time; total process memory can still exceed the setting.

Dependencies:

- Update the lockfile from yanked `chacha20` 0.10.1 to 0.10.2, which fixes
  use of an SSE4.1 instruction in its SSE2 backend.

Compatibility and validation:

- Public API compatibility, output formats and unitig IDs are preserved
  relative to 0.6.0. No index-format migration is required.
- Add regression coverage for cached lengths, sequence and chunk boundaries,
  placeholders, multiple files, and memory-policy thresholds.
- Validate graph and downstream index parity on GENCODE v50, and mapping
  parity with piscem's strict and permissive skipping strategies. On the
  reviewed Apple M5 Max workload, the final changes reduce full Salmon index
  construction from 8.045 to 7.160 seconds relative to #4+#6 (four measured
  rounds, warm input); process peak RSS is similar. Detailed comparisons and
  adverse-workload results are recorded in the reviews of #4–#7.

## 0.6.0 — 2026-10-30

Behavior change (output):

- **Unitig ids change.** The MPHF is now sux 0.15's `PHastR` (in place of
  `ph`'s FMPH), so the MPHF indices that name unitigs, and hence the ids in
  `.cf_seg`/`.cf_seq`, differ from 0.5.x. The compacted de Bruijn graph itself
  is identical (same unitig sequences and tilings, up to renaming). Output is
  now also independent of the MPHF construction regime (in-memory vs.
  streaming) and of the thread count. Anything that compared cf1-rs output
  across versions by unitig id must re-baseline.

Breaking (library API; all modules are `pub`):

- `classify::Classifier::process_substring` takes an extra
  `seen: Option<&mut SeenWindows>` argument.
- `directed_kmer::AnnotatedKmer` has a new public field `hash` (its MPHF
  index); `AnnotatedKmer::roll_to_next_kmer` is removed in favor of the new
  batched `directed_kmer::AnnotatedRun`.
- The `ph` dependency is dropped. `cf_build` and its options are unchanged.

Performance:

- MPHF: `PHastR` in both regimes. In memory it is built straight from the key
  slice (k-mers no longer touch disk); the streaming regime feeds the keys once
  into PHastR's offline construction, deduplicating later bins in the
  background, so the dedup-file write and consolidation copy are gone.
- Phases 4 and 5 resolve MPHF lookups in batches of up to 32 k-mers with state
  prefetching (x86-64 and aarch64); phase 4 skips content-defined windows a
  task has already applied.
- Phases 1 and 2 batch small sequences (~4 Mbp per task, task-local
  histograms, fewer bin flushes); routing drops exact repeat super k-mers
  within a task. In-memory keys are gathered a wave of bins at a time into one
  pre-reserved vector.
- GENCODE v50, 16 threads: P1 1.95 → 0.41 s, P2 2.57 → 0.95 s, P4 10.2 → 1.6 s,
  P5 8.7 → 3.0 s. GRCh38 genome (2.5B k-mers, streaming), 16 threads:
  110.7 → 76.2 s wall, peak RSS unchanged (~5.8 GB). Peak RSS of in-memory
  (transcriptome-scale) builds grows modestly (GENCODE v50: 3.8 → 4.5 GB).

## 0.5.1

Dependency updates (needletail 0.7, simd-minimizers 3.0, semver-compatible
bumps).

## 0.5.0

New features (both backward compatible — defaults preserve 0.4.x behavior):

- **`synchronize_output`** (CLI `--synchronize-output`, `cf_build` builder option,
  default `false`): emit the `.cf_seq` tiling in input order via a bounded reorder
  buffer, instead of task-completion order. Makes the build deterministic and keeps
  any trailing sequences (e.g. decoys) contiguous in the downstream reference
  numbering. Off by default, so existing callers are unaffected.
- **Ambiguous-base (`N`) handling**, gated on `poly_n_stretch`: with it enabled the
  minimizer-counting and super-k-mer-routing phases split each sequence on `N` runs
  (`for_each_acgt_segment`), so no k-mer spans a placeholder and the de Bruijn graph
  is built natively over the ACGT segments. With `poly_n_stretch` disabled, an input
  containing `N` is now rejected with a clear error (`pass --poly-N-stretch ...`)
  rather than silently corrupting or cryptically aborting, since `packed-seq`'s
  behavior on `N` is path-dependent.

Internal:

- `dna::contains_non_acgt` — a vectorizable `N` check (`& 0xDF` upper-case maps
  exactly `ACGTacgt` onto `{A,C,G,T}`) used on the `poly_n`-off guard path.
- `pipeline::PipelineResult` type alias for the `run_pipeline` return.

## 0.4.1

cargo fmt, bench updates, MSRV 1.91.
