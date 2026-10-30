# Changelog

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
