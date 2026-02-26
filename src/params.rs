use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cf1-rs", about = "Compacted reference de Bruijn graph construction")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Build a compacted de Bruijn graph
    Build {
        /// Input sequence file (FASTA/FASTQ, optionally gzipped)
        #[arg(short = 's')]
        seq_file: Option<PathBuf>,

        /// File listing input sequence file paths
        #[arg(short = 'l')]
        list_file: Option<PathBuf>,

        /// Directory containing input sequence files
        #[arg(short = 'd')]
        dir_path: Option<PathBuf>,

        /// K-mer length (odd, 1-63)
        #[arg(short = 'k', default_value = "31")]
        k: usize,

        /// Number of threads
        #[arg(short = 't', default_value = "1")]
        threads: usize,

        /// Output file prefix
        #[arg(short = 'o')]
        output: PathBuf,

        /// Output format (0=FASTA, 1=GFA1, 2=GFA2, 3=GFA-reduced)
        #[arg(short = 'f', default_value = "3")]
        format: u8,

        /// Working directory for temporary files
        #[arg(short = 'w', long = "work-dir")]
        work_dir: Option<PathBuf>,

        /// Track sequences shorter than k
        #[arg(long = "track-short-seqs")]
        track_short_seqs: bool,

        /// Handle poly-N stretches between unitigs
        #[arg(long = "poly-N-stretch")]
        poly_n_stretch: bool,

        /// Collate output in memory (one sequence per thread)
        #[arg(long = "collate-output-in-mem")]
        collate_in_mem: bool,

        /// Number of minimizer bins
        #[arg(long = "num-bins", default_value = "128")]
        num_bins: usize,

        /// Memory budget in GB for MPHF construction (determines how many k-mers
        /// can be cached in RAM during construction)
        #[arg(long = "memory-budget", default_value = "4.0")]
        memory_budget_gb: f64,
    },
}

/// Resolved build parameters.
pub struct Params {
    pub input_files: Vec<PathBuf>,
    pub k: usize,
    pub threads: usize,
    pub output: PathBuf,
    pub format: u8,
    pub work_dir: PathBuf,
    pub track_short_seqs: bool,
    pub poly_n_stretch: bool,
    pub collate_in_mem: bool,
    pub num_bins: usize,
    pub memory_budget_bytes: usize,
}

impl Params {
    pub fn from_build_args(
        seq_file: Option<PathBuf>,
        list_file: Option<PathBuf>,
        dir_path: Option<PathBuf>,
        k: usize,
        threads: usize,
        output: PathBuf,
        format: u8,
        work_dir: Option<PathBuf>,
        track_short_seqs: bool,
        poly_n_stretch: bool,
        collate_in_mem: bool,
        num_bins: usize,
        memory_budget_gb: f64,
    ) -> anyhow::Result<Self> {
        assert!(k % 2 == 1 && k >= 1 && k <= 63, "k must be odd and in [1, 63]");

        let mut input_files = Vec::new();

        if let Some(s) = seq_file {
            input_files.push(s);
        }

        if let Some(l) = list_file {
            let content = std::fs::read_to_string(&l)?;
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    input_files.push(PathBuf::from(line));
                }
            }
        }

        if let Some(d) = dir_path {
            for entry in std::fs::read_dir(&d)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if matches!(ext, "fa" | "fasta" | "fna" | "gz") {
                        input_files.push(path);
                    }
                }
            }
        }

        assert!(!input_files.is_empty(), "No input files specified");

        let work_dir = work_dir.unwrap_or_else(|| {
            output
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        });

        std::fs::create_dir_all(&work_dir)?;

        let memory_budget_bytes = (memory_budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;

        Ok(Params {
            input_files,
            k,
            threads,
            output,
            format,
            work_dir,
            track_short_seqs,
            poly_n_stretch,
            collate_in_mem,
            num_bins,
            memory_budget_bytes,
        })
    }

    pub fn segment_file_path(&self) -> PathBuf {
        let mut p = self.output.clone();
        let name = p.file_name().unwrap().to_str().unwrap().to_string() + ".cf_seg";
        p.set_file_name(name);
        p
    }

    pub fn sequence_file_path(&self) -> PathBuf {
        let mut p = self.output.clone();
        let name = p.file_name().unwrap().to_str().unwrap().to_string() + ".cf_seq";
        p.set_file_name(name);
        p
    }

    pub fn json_file_path(&self) -> PathBuf {
        let mut p = self.output.clone();
        let name = p.file_name().unwrap().to_str().unwrap().to_string() + ".json";
        p.set_file_name(name);
        p
    }

    /// Minimizer length: min(k/2 - 1, 15), always odd.
    pub fn minimizer_len(&self) -> usize {
        let m = std::cmp::min(self.k / 2, 15);
        if m % 2 == 0 {
            m.saturating_sub(1).max(1)
        } else {
            m
        }
    }
}
