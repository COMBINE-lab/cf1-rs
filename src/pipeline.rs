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
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Return current RSS in MB (point-in-time snapshot), or 0 where it is not
/// available.
///
/// Read straight from the kernel: `/proc/self/statm` on Linux, `proc_pidinfo`
/// on macOS (the resident size `ps` reports, without spawning it). Spawning
/// `ps` took ~0.45 s per call on a busy host, which made every pipeline run,
/// however small its input, take seconds just to log RSS.
pub fn current_rss_mb() -> usize {
    current_rss_bytes() / (1024 * 1024)
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> usize {
    // statm fields are in pages; the second is the resident set.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map_or(0, |pages| pages * page.max(0) as usize)
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> usize {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // Returns the bytes written: anything short of the whole struct is a failure.
    let n = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            (&raw mut info).cast(),
            size,
        )
    };
    if n == size {
        info.pti_resident_size as usize
    } else {
        0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn current_rss_bytes() -> usize {
    0
}

/// Output of [`run_pipeline`]: the unitig/tiling metadata, plus two `(name, length)`
/// lists of input sequences that produced no tiles — `short_seqs` (length `< k`) and
/// `untiled_seqs` (length `>= k` but no run of `>= k` consecutive ACGT bases, e.g.
/// a sequence fragmented below `k` by `N`s).
pub type PipelineResult = (
    crate::output::UnipathsMeta,
    Vec<(String, usize)>,
    Vec<(String, usize)>,
);

pub fn run_pipeline<const K: usize>(params: &Params) -> anyhow::Result<PipelineResult>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: crate::mphf::RadixSortDedup,
{
    info!(
        "RSS at start: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    let budget = InFlightBudget::new(DEFAULT_IN_FLIGHT_BYTES);

    // Phase 1: Count minimizer frequencies.
    info!("Phase 1: Counting minimizer frequencies...");
    let histogram = count_minimizer_histogram(params, &budget)?;
    purge_allocator();
    info!(
        "RSS after P1: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    // Phase 2: Partition + route super k-mers.
    info!("Phase 2: Partitioning minimizers and routing super k-mers...");
    let partitioning = partition_minimizers(&histogram, params.num_bins)?;
    drop(histogram);
    route_superkmers(params, &partitioning, &budget)?;
    purge_allocator();
    info!(
        "RSS after P2: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    // Phase 3: Build global MPHF.
    info!("Phase 3: Building global MPHF...");
    let num_bins = partitioning.num_bins();
    let mphf = Mphf::<K>::build(&partitioning, &params.work_dir, params.memory_budget_bytes)?;
    purge_allocator();
    info!(
        "RSS after P3 MPHF: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );
    let states = AtomicStateVector::new(mphf.total_kmers() as usize);
    info!(
        "RSS after P3 StateVector: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    // Clean up bin temp files.
    cleanup_bin_files(&params.work_dir, num_bins);

    // Phase 4: DFA classification.
    info!("Phase 4: Classifying k-mer vertices...");
    let short_seqs = classify_vertices::<K>(params, &mphf, &states, &budget)?;
    purge_allocator();
    info!(
        "RSS after P4: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    // Phase 5: Output.
    info!("Phase 5: Extracting unitigs and writing output...");
    let (meta, untiled_seqs) = extract_and_output::<K>(params, &mphf, &states, &budget)?;
    purge_allocator();
    info!(
        "RSS after P5: current={} MB, peak={} MB",
        current_rss_mb(),
        peak_rss_mb()
    );

    // Write JSON.
    write_json(params, &meta, &short_seqs, &untiled_seqs)?;

    info!("Pipeline complete.");
    Ok((meta, short_seqs, untiled_seqs))
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::{current_rss_mb, peak_rss_mb};

    /// Touch `mb` MB so the resident set is well above rounding noise.
    fn resident_buffer(mb: usize) -> Vec<u8> {
        let mut buf = vec![0u8; mb << 20];
        for i in (0..buf.len()).step_by(4096) {
            buf[i] = 1;
        }
        std::hint::black_box(buf)
    }

    /// The `/proc/self/statm` reading (pages × page size) agrees with the
    /// kernel's own `VmRSS` (what `ps` reports). The two are read back to back:
    /// other tests in this process allocate concurrently, so a slow reference
    /// such as spawning `ps` would race them.
    #[cfg(target_os = "linux")]
    #[test]
    fn current_rss_matches_vmrss() {
        let buf = resident_buffer(64);
        let ours = current_rss_mb();
        let vmrss_mb = std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("VmRSS:"))
            .and_then(|v| v.split_whitespace().next()?.parse::<usize>().ok())
            .unwrap()
            / 1024;
        assert!(ours >= 64, "RSS {ours} MB with a 64 MB buffer resident");
        assert!(
            ours.abs_diff(vmrss_mb) <= 8,
            "statm {ours} MB vs VmRSS {vmrss_mb} MB"
        );
        drop(buf);
    }

    /// The current resident size counts a freshly touched buffer and never
    /// exceeds the high-water mark `getrusage` reports, read just after it.
    #[test]
    fn current_rss_counts_resident_memory_and_stays_below_peak() {
        let buf = resident_buffer(64);
        let ours = current_rss_mb();
        let peak = peak_rss_mb();
        assert!(ours >= 64, "RSS {ours} MB with a 64 MB buffer resident");
        assert!(ours <= peak + 1, "current {ours} MB above peak {peak} MB");
        drop(buf);
    }
}
