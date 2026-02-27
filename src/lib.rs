pub mod budget;
pub mod classify;
pub mod directed_kmer;
pub mod dna;
pub mod kmer;
pub mod minimizer;
pub mod mphf;
pub mod output;
pub mod params;
pub mod pipeline;
pub mod state;
pub mod state_vector;
pub mod superkmer;

use std::path::PathBuf;

use crate::kmer::{Kmer, KmerBits};
use crate::mphf::RadixSortDedup;
use crate::output::UnipathsMeta;
use crate::params::Params;
use crate::pipeline::run_pipeline;

/// How to specify input sequences for `cf_build`.
pub enum CfInput {
    /// Pre-resolved file paths (FASTA/FASTQ, optionally gzipped).
    Files(Vec<PathBuf>),
    /// Path to a file listing input paths, one per line.
    ListFile(PathBuf),
    /// Directory containing FASTA/FASTQ files.
    Directory(PathBuf),
}

/// Result of a successful `cf_build` invocation.
pub struct CfBuildResult {
    /// Path to the segment file (`{prefix}.cf_seg`).
    pub seg_file: PathBuf,
    /// Path to the sequence/tiling file (`{prefix}.cf_seq`).
    pub seq_file: PathBuf,
    /// Path to the JSON metadata file (`{prefix}.json`).
    pub json_file: PathBuf,
    /// Number of distinct k-mers (vertices).
    pub vertex_count: u64,
    /// Number of maximal unitigs.
    pub unitig_count: u64,
    /// Length of the longest unitig (in bases).
    pub max_unitig_len: usize,
    /// Length of the shortest unitig (in bases).
    pub min_unitig_len: usize,
    /// Sum of all unitig lengths (in bases).
    pub sum_unitig_len: u64,
    /// Input sequences shorter than k (name, length).
    pub short_seqs: Vec<(String, usize)>,
}

/// Build a compacted de Bruijn graph from input sequences.
///
/// Uses a scoped rayon thread pool internally, so it is safe to call from
/// a program that already has its own rayon global pool.
///
/// # Required parameters
/// - `input` — input sequence specification
/// - `output_prefix` — prefix for output files (`.cf_seg`, `.cf_seq`, `.json`)
///
/// # Optional parameters (with defaults)
/// - `k` — k-mer length, must be odd and in \[1, 63\] (default: 31)
/// - `threads` — number of worker threads (default: 1)
/// - `work_dir` — directory for temporary files (default: parent of `output_prefix`)
/// - `num_bins` — number of minimizer bins (default: 128)
/// - `memory_budget_gb` — memory budget for MPHF construction in GB (default: 4.0)
#[bon::builder]
pub fn cf_build(
    input: CfInput,
    output_prefix: PathBuf,
    #[builder(default = 31)] k: usize,
    #[builder(default = 1)] threads: usize,
    work_dir: Option<PathBuf>,
    #[builder(default = 128)] num_bins: usize,
    #[builder(default = 4.0)] memory_budget_gb: f64,
) -> anyhow::Result<CfBuildResult> {
    let input_files = resolve_input_files(&input)?;

    let params = Params::from_resolved(
        input_files,
        k,
        threads,
        output_prefix,
        3, // format=3 (GFA-reduced)
        work_dir,
        true,  // track_short_seqs
        true,  // poly_n_stretch
        true,  // collate_in_mem
        num_bins,
        memory_budget_gb,
    )?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    pool.install(|| dispatch_k!(k, run_and_collect, &params))
}

/// Resolve input files from a `CfInput` specification.
fn resolve_input_files(input: &CfInput) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    match input {
        CfInput::Files(paths) => {
            files.extend(paths.iter().cloned());
        }
        CfInput::ListFile(list_path) => {
            let content = std::fs::read_to_string(list_path)?;
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    files.push(PathBuf::from(line));
                }
            }
        }
        CfInput::Directory(dir) => {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if matches!(ext, "fa" | "fasta" | "fna" | "gz" | "fq" | "fastq") {
                        files.push(path);
                    }
                }
            }
        }
    }
    anyhow::ensure!(!files.is_empty(), "No input files found");
    Ok(files)
}

/// Internal: run pipeline and collect results into `CfBuildResult`.
fn run_and_collect<const K: usize>(params: &Params) -> anyhow::Result<CfBuildResult>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    let (meta, short_seqs) = run_pipeline::<K>(params)?;
    Ok(build_result(params, &meta, short_seqs))
}

/// Map pipeline outputs to `CfBuildResult`.
fn build_result(params: &Params, meta: &UnipathsMeta, short_seqs: Vec<(String, usize)>) -> CfBuildResult {
    CfBuildResult {
        seg_file: params.segment_file_path(),
        seq_file: params.sequence_file_path(),
        json_file: params.json_file_path(),
        vertex_count: meta.kmer_count,
        unitig_count: meta.unipath_count,
        max_unitig_len: meta.max_len,
        min_unitig_len: if meta.min_len == usize::MAX { 0 } else { meta.min_len },
        sum_unitig_len: meta.sum_len,
        short_seqs,
    }
}
