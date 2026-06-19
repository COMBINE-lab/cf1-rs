# Changelog

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
