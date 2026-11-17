pub mod batch;
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
    /// Input sequences with length >= k that produced no tiles (name, length).
    pub untiled_seqs: Vec<(String, usize)>,
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
    /// Emit the tiling in input-sequence order (off by default; see
    /// [`Params::synchronize_output`]). Enable when a downstream consumer numbers
    /// references by `.cf_seq` line order (e.g. salmon's contiguous decoy block).
    #[builder(default = false)]
    synchronize_output: bool,
) -> anyhow::Result<CfBuildResult> {
    let input_files = resolve_input_files(&input)?;

    let params = Params::from_resolved(
        input_files,
        k,
        threads,
        output_prefix,
        3, // format=3 (GFA-reduced)
        work_dir,
        true, // track_short_seqs
        true, // poly_n_stretch
        true, // collate_in_mem
        num_bins,
        memory_budget_gb,
        synchronize_output,
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
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
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
    let (meta, short_seqs, untiled_seqs) = run_pipeline::<K>(params)?;
    Ok(build_result(params, &meta, short_seqs, untiled_seqs))
}

/// Map pipeline outputs to `CfBuildResult`.
fn build_result(
    params: &Params,
    meta: &UnipathsMeta,
    short_seqs: Vec<(String, usize)>,
    untiled_seqs: Vec<(String, usize)>,
) -> CfBuildResult {
    CfBuildResult {
        seg_file: params.segment_file_path(),
        seq_file: params.sequence_file_path(),
        json_file: params.json_file_path(),
        vertex_count: meta.kmer_count,
        unitig_count: meta.unipath_count,
        max_unitig_len: meta.max_len,
        min_unitig_len: if meta.min_len == usize::MAX {
            0
        } else {
            meta.min_len
        },
        sum_unitig_len: meta.sum_len,
        short_seqs,
        untiled_seqs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic generator (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn bases(&mut self, len: usize) -> Vec<u8> {
            (0..len)
                .map(|_| b"ACGT"[(self.next() & 3) as usize])
                .collect()
        }
    }

    fn revcomp(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|&b| crate::dna::complement_char(b))
            .collect()
    }

    /// Every reference is spelled exactly by its tiling (consecutive unitigs
    /// overlapping by k - 1 bases), tilings are in input order, and every
    /// unitig is written once. The input looks like a transcriptome — genes
    /// whose isoforms share exons, adjacent in the file — and spans several
    /// phase-5 batches.
    #[test]
    fn tilings_spell_the_references() {
        let k = 31;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut refs: Vec<(String, Vec<u8>)> = Vec::new();
        let mut total = 0;
        while total < 2 * crate::batch::BATCH_BASES {
            let gene = refs.len();
            let exons: Vec<Vec<u8>> = (0..6)
                .map(|_| {
                    let len = 40 + (rng.next() % 300) as usize;
                    rng.bases(len)
                })
                .collect();
            for iso in 0..4 {
                let mut seq = Vec::new();
                for (e, exon) in exons.iter().enumerate() {
                    // Each isoform skips a different exon (or none).
                    if e != iso + 1 {
                        seq.extend_from_slice(exon);
                    }
                }
                if iso == 3 {
                    seq = revcomp(&seq);
                }
                total += seq.len();
                refs.push((format!("g{gene}_i{iso}"), seq));
            }
        }
        // A reference shorter than k gets no tiling line.
        refs.push(("short".to_string(), b"ACGTACGT".to_vec()));

        let dir = std::env::temp_dir().join(format!("cf1_tiling_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fasta = dir.join("refs.fa");
        let mut text = Vec::new();
        for (name, seq) in &refs {
            text.extend_from_slice(format!(">{name}\n").as_bytes());
            text.extend_from_slice(seq);
            text.push(b'\n');
        }
        std::fs::write(&fasta, text).unwrap();

        let result = cf_build()
            .input(CfInput::Files(vec![fasta]))
            .output_prefix(dir.join("out"))
            .k(k)
            .threads(4)
            .work_dir(dir.clone())
            .synchronize_output(true)
            .call()
            .unwrap();

        let segs = std::fs::read_to_string(&result.seg_file).unwrap();
        let mut unitigs = std::collections::HashMap::new();
        for line in segs.lines() {
            let (id, seq) = line.split_once('\t').unwrap();
            assert!(
                unitigs
                    .insert(id.to_string(), seq.as_bytes().to_vec())
                    .is_none()
            );
        }
        assert_eq!(unitigs.len() as u64, result.unitig_count);

        let tilings = std::fs::read_to_string(&result.seq_file).unwrap();
        let long_refs: Vec<_> = refs.iter().filter(|(_, s)| s.len() >= k).collect();
        assert_eq!(tilings.lines().count(), long_refs.len());
        for (line, (name, seq)) in tilings.lines().zip(long_refs) {
            let (header, tiles) = line.split_once('\t').unwrap();
            assert_eq!(header, format!("Reference:1_Sequence:{name}"));
            let mut spelled: Vec<u8> = Vec::new();
            for tile in tiles.split(' ') {
                let (id, dir) = tile.split_at(tile.len() - 1);
                let unitig = &unitigs[id];
                let oriented = if dir == "+" {
                    unitig.clone()
                } else {
                    revcomp(unitig)
                };
                if spelled.is_empty() {
                    spelled = oriented;
                } else {
                    assert_eq!(spelled[spelled.len() - (k - 1)..], oriented[..k - 1]);
                    spelled.extend_from_slice(&oriented[k - 1..]);
                }
            }
            assert!(
                spelled == *seq,
                "{name}: tiling does not spell the reference"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
