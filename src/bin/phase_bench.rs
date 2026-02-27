//! Benchmark individual pipeline phases in isolation.
//! Usage:
//!   phase_bench <phase> -k <K> -t <threads> -s <input.fa.gz> -o <prefix> [--num-bins N] [--memory-budget G]
//!
//! Phases:
//!   p1   - Count minimizer histogram only
//!   p12  - P1 + P2 (partition + route super k-mers to bin files)
//!   p3   - Build MPHF from existing bin files (must run p12 first)
//!   p4   - Classify vertices (must run p12 + p3 first, but we rebuild MPHF inline)
//!   p12_only - P1+P2, then exit (leaves bin files for p3)

use cf1_rs::budget::InFlightBudget;
use rayon::prelude::*;
use cf1_rs::classify::classify_vertices;
use cf1_rs::kmer::{Kmer, KmerBits};
use cf1_rs::minimizer::{count_minimizer_histogram, partition_minimizers};
use cf1_rs::mphf::{Mphf, RadixSortDedup};
use cf1_rs::params::Params;
use cf1_rs::pipeline::{current_rss_mb, peak_rss_mb};
use cf1_rs::state_vector::AtomicStateVector;
use cf1_rs::superkmer::{cleanup_bin_files, route_superkmers};
use tracing::info;

const DEFAULT_IN_FLIGHT_BYTES: usize = 1024 * 1024 * 1024;

fn purge_allocator() {
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: phase_bench <phase> -k <K> -t <threads> -s <input> -o <prefix> [--num-bins N] [--memory-budget G]");
        eprintln!("Phases: p1, p2, p3, p4, p12_only");
        std::process::exit(1);
    }

    let phase = args[1].clone();

    // Parse remaining args as build args
    let mut k = 31usize;
    let mut threads = 8usize;
    let mut input = String::new();
    let mut output = String::new();
    let mut num_bins = 128usize;
    let mut memory_budget_gb = 4.0f64;
    let mut poly_n = false;
    let mut clone_threshold: Option<usize> = None;
    let mut consolidated_path: Option<String> = None;
    let mut total_kmers: Option<usize> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-k" => { k = args[i+1].parse()?; i += 2; }
            "-t" => { threads = args[i+1].parse()?; i += 2; }
            "-s" => { input = args[i+1].clone(); i += 2; }
            "-o" => { output = args[i+1].clone(); i += 2; }
            "--num-bins" => { num_bins = args[i+1].parse()?; i += 2; }
            "--memory-budget" => { memory_budget_gb = args[i+1].parse()?; i += 2; }
            "--poly-N-stretch" => { poly_n = true; i += 1; }
            "--clone-threshold" => { clone_threshold = Some(args[i+1].parse()?); i += 2; }
            "--consolidated" => { consolidated_path = Some(args[i+1].clone()); i += 2; }
            "--total-kmers" => { total_kmers = Some(args[i+1].parse()?); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    let params = Params::from_build_args(
        Some(std::path::PathBuf::from(&input)),
        None, None,
        k, threads,
        std::path::PathBuf::from(&output),
        3, // GFA-reduced
        None,
        false,
        poly_n,
        false,
        num_bins,
        memory_budget_gb,
    )?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(params.threads)
        .build_global()
        .ok();

    info!("Phase benchmark: phase={}, k={}, threads={}, clone_threshold={:?}", phase, k, threads, clone_threshold);

    cf1_rs::dispatch_k!(k, run_phase, &params, &phase, clone_threshold, &consolidated_path, &total_kmers)?;

    Ok(())
}

fn run_phase<const K: usize>(
    params: &Params,
    phase: &str,
    clone_threshold: Option<usize>,
    consolidated_path: &Option<String>,
    total_kmers: &Option<usize>,
) -> anyhow::Result<()>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    let budget = InFlightBudget::new(DEFAULT_IN_FLIGHT_BYTES);

    info!("RSS at start: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    match phase {
        "p1" => {
            info!("=== Phase 1 only: Count minimizer histogram ===");
            let histogram = count_minimizer_histogram(params, &budget)?;
            purge_allocator();
            info!("RSS after P1: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
            let total: u64 = histogram.iter().sum();
            info!("Histogram total: {}", total);
        }

        "p2" => {
            info!("=== Phase 2 only: Route super k-mers (assumes bin files don't exist yet) ===");
            // Must run P1 first to get partitioning
            let histogram = count_minimizer_histogram(params, &budget)?;
            let partitioning = partition_minimizers(&histogram, params.num_bins)?;
            drop(histogram);
            purge_allocator();
            info!("RSS before P2: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

            route_superkmers(params, &partitioning, &budget)?;
            purge_allocator();
            info!("RSS after P2: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
        }

        "p3" => {
            info!("=== Phase 3 only: Build MPHF from existing bin files ===");
            // We need a Partitioning. Re-run P1 to get it, then measure P3 in isolation.
            let histogram = count_minimizer_histogram(params, &budget)?;
            let partitioning = partition_minimizers(&histogram, params.num_bins)?;
            drop(histogram);
            purge_allocator();
            info!("RSS before P3 (after P1 for partitioning): current={} MB, peak={} MB",
                current_rss_mb(), peak_rss_mb());

            // Reset peak by forking a child... can't easily reset getrusage peak.
            // Instead, just note the current values and compute delta.
            let pre_current = current_rss_mb();
            let pre_peak = peak_rss_mb();

            let ct = clone_threshold.unwrap_or(0);
            info!("Using clone_threshold={}", ct);
            let mphf = Mphf::<K>::build_with_clone_threshold(&partitioning, &params.work_dir, ct)?;
            purge_allocator();
            info!("RSS after P3 MPHF: current={} MB, peak={} MB (delta_current={}, delta_peak={})",
                current_rss_mb(), peak_rss_mb(),
                current_rss_mb() as i64 - pre_current as i64,
                peak_rss_mb() as i64 - pre_peak as i64);

            let states = AtomicStateVector::new(mphf.total_kmers() as usize);
            info!("RSS after StateVector: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
            info!("Total k-mers: {}", mphf.total_kmers());
            drop(states);
            drop(mphf);
        }

        "p4" => {
            info!("=== Phase 4 only: Classify vertices ===");
            // Need MPHF + states. Run P1+P3 to build them, then measure P4.
            let histogram = count_minimizer_histogram(params, &budget)?;
            let partitioning = partition_minimizers(&histogram, params.num_bins)?;
            drop(histogram);

            let mphf = Mphf::<K>::build(&partitioning, &params.work_dir, params.memory_budget_bytes)?;
            purge_allocator();
            let states = AtomicStateVector::new(mphf.total_kmers() as usize);
            cleanup_bin_files(&params.work_dir, partitioning.num_bins());
            purge_allocator();

            info!("RSS before P4: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
            let pre_current = current_rss_mb();

            let _short_seqs = classify_vertices::<K>(params, &mphf, &states, &budget)?;
            purge_allocator();
            info!("RSS after P4: current={} MB, peak={} MB (delta_current={})",
                current_rss_mb(), peak_rss_mb(),
                current_rss_mb() as i64 - pre_current as i64);
        }

        "p3_prep" => {
            // Dedup raw bin files and create consolidated file, preserving raw bins.
            // Output: <work_dir>/cf1_dedup_all.bin + prints total k-mer count.
            info!("=== P3 prep: dedup + consolidate (preserving raw bins) ===");
            let histogram = count_minimizer_histogram(params, &budget)?;
            let partitioning = partition_minimizers(&histogram, params.num_bins)?;
            drop(histogram);
            purge_allocator();

            let num_bins = partitioning.num_bins();
            info!("Deduplicating {} bins...", num_bins);
            let per_bin_counts: Vec<usize> = (0..num_bins)
                .into_par_iter()
                .map(|bin| {
                    let src = cf1_rs::superkmer::bin_file_path(&params.work_dir, bin);
                    let dst = params.work_dir.join(format!("cf1_dedup_{:05}.bin", bin));
                    cf1_rs::mphf::dedup_bin_public::<K>(&src, &dst)
                })
                .collect();
            let total: usize = per_bin_counts.iter().sum();
            info!("Total {} distinct canonical k-mers", total);

            // Consolidate into single file
            let consolidated_path = params.work_dir.join("cf1_dedup_all.bin");
            {
                let out_file = std::fs::File::create(&consolidated_path)?;
                let mut writer = std::io::BufWriter::with_capacity(256 * 1024, out_file);
                for bin in 0..num_bins {
                    let bin_path = params.work_dir.join(format!("cf1_dedup_{:05}.bin", bin));
                    if let Ok(mut f) = std::fs::File::open(&bin_path) {
                        std::io::copy(&mut f, &mut writer)?;
                    }
                    let _ = std::fs::remove_file(&bin_path);
                }
                use std::io::Write;
                writer.flush()?;
            }

            info!("Consolidated file: {}", consolidated_path.display());
            info!("Use with: p3_mphf --consolidated {} --total-kmers {}", consolidated_path.display(), total);
        }

        "p3_mphf" => {
            // Test MPHF construction only, from a pre-existing consolidated dedup file.
            // Requires --consolidated <path> --total-kmers <N> --clone-threshold <N>
            let cpath = consolidated_path.as_ref()
                .expect("p3_mphf requires --consolidated <path>");
            let total = total_kmers
                .expect("p3_mphf requires --total-kmers <N>");
            let ct = clone_threshold.unwrap_or(300_000_000);

            info!("=== P3 MPHF only: consolidated={}, total={}, clone_threshold={} ===",
                cpath, total, ct);
            info!("RSS before MPHF: current={} MB, peak={} MB",
                current_rss_mb(), peak_rss_mb());

            let start = std::time::Instant::now();
            let mphf = Mphf::<K>::build_from_consolidated(
                std::path::Path::new(cpath), total, ct,
            )?;
            let elapsed = start.elapsed();
            purge_allocator();

            info!("RSS after MPHF: current={} MB, peak={} MB",
                current_rss_mb(), peak_rss_mb());
            info!("MPHF build time: {:.1}s", elapsed.as_secs_f64());
            info!("Total k-mers: {}", mphf.total_kmers());
            drop(mphf);
        }

        "p12_only" => {
            info!("=== P1+P2: Leave bin files on disk for subsequent p3 ===");
            let histogram = count_minimizer_histogram(params, &budget)?;
            purge_allocator();
            info!("RSS after P1: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

            let partitioning = partition_minimizers(&histogram, params.num_bins)?;
            drop(histogram);
            route_superkmers(params, &partitioning, &budget)?;
            purge_allocator();
            info!("RSS after P2: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
            info!("Bin files left in work_dir for p3");
        }

        _ => {
            eprintln!("Unknown phase: {}. Use p1, p2, p3, p4, p12_only", phase);
            std::process::exit(1);
        }
    }

    info!("Phase benchmark done.");
    Ok(())
}
