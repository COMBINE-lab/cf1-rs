use crate::budget::InFlightBudget;
use crate::classify::classify_vertices;
use crate::kmer::{Kmer, KmerBits};
use crate::minimizer::{count_minimizer_histogram, partition_minimizers};
use crate::mphf::Mphf;
use crate::output::{extract_and_output, write_json};
use crate::params::Params;
use crate::state_vector::AtomicStateVector;
use crate::superkmer::{cleanup_bin_files, route_superkmers};
use tracing::info;

/// Default in-flight budget: 1 GB of sequence data.
/// Must be large enough to keep all threads busy (>= num_threads × max_seq_len).
const DEFAULT_IN_FLIGHT_BYTES: usize = 1024 * 1024 * 1024;

/// Return peak RSS (high-water mark) in MB via getrusage.
/// On macOS ru_maxrss is in bytes; on Linux it is in KB.
pub fn peak_rss_mb() -> usize {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            let bytes = usage.ru_maxrss as usize;
            if cfg!(target_os = "macos") {
                bytes / (1024 * 1024)
            } else {
                // Linux: ru_maxrss is in KB
                bytes / 1024
            }
        } else {
            0
        }
    }
}

/// Ask the system allocator to release unused pages back to the OS.
fn purge_allocator() {
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(
                zone: *mut std::ffi::c_void,
                goal: usize,
            ) -> usize;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Return current RSS in MB via ps (point-in-time snapshot).
pub fn current_rss_mb() -> usize {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0)
        / 1024
}

pub fn run_pipeline<const K: usize>(params: &Params) -> anyhow::Result<(crate::output::UnipathsMeta, Vec<(String, usize)>)>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: crate::mphf::RadixSortDedup,
{
    info!("RSS at start: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    let budget = InFlightBudget::new(DEFAULT_IN_FLIGHT_BYTES);

    // Phase 1: Count minimizer frequencies.
    info!("Phase 1: Counting minimizer frequencies...");
    let histogram = count_minimizer_histogram(params, &budget)?;
    purge_allocator();
    info!("RSS after P1: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    // Phase 2: Partition + route super k-mers.
    info!("Phase 2: Partitioning minimizers and routing super k-mers...");
    let partitioning = partition_minimizers(&histogram, params.num_bins)?;
    drop(histogram);
    route_superkmers(params, &partitioning, &budget)?;
    purge_allocator();
    info!("RSS after P2: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    // Phase 3: Build global MPHF.
    info!("Phase 3: Building global MPHF...");
    let num_bins = partitioning.num_bins();
    let mphf = Mphf::<K>::build(&partitioning, &params.work_dir, params.memory_budget_bytes)?;
    purge_allocator();
    info!("RSS after P3 MPHF: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());
    let states = AtomicStateVector::new(mphf.total_kmers() as usize);
    info!("RSS after P3 StateVector: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    // Clean up bin temp files.
    cleanup_bin_files(&params.work_dir, num_bins);

    // Phase 4: DFA classification.
    info!("Phase 4: Classifying k-mer vertices...");
    let short_seqs = classify_vertices::<K>(params, &mphf, &states, &budget)?;
    purge_allocator();
    info!("RSS after P4: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    // Phase 5: Output.
    info!("Phase 5: Extracting unitigs and writing output...");
    let meta = extract_and_output::<K>(params, &mphf, &states, &budget)?;
    purge_allocator();
    info!("RSS after P5: current={} MB, peak={} MB", current_rss_mb(), peak_rss_mb());

    // Write JSON.
    write_json(params, &meta, &short_seqs)?;

    info!("Pipeline complete.");
    Ok((meta, short_seqs))
}
