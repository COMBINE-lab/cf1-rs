use crate::budget::InFlightBudget;
use crate::params::Params;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// Number of histogram buckets for minimizer frequency estimation.
/// Each bucket covers an equal range of the minimizer hash space.
const NUM_HISTOGRAM_BUCKETS: usize = 65_536;

/// Target chunk size (bases) for splitting large sequences across threads.
/// Keeps simd_minimizers internal allocations small enough for the system
/// allocator to efficiently reuse freed pages between chromosomes.
const CHUNK_BASES: usize = 4_000_000;

/// Map a minimizer value to a histogram bucket via a fast hash.
/// Uses the same hash (xxh3) as bin_for_minimizer, ensuring consistency.
#[inline]
fn bucket_hash(val: u64) -> usize {
    let h = xxhash_rust::xxh3::xxh3_64(&val.to_le_bytes());
    (h >> 48) as usize // top 16 bits → 0..65535
}

/// Count minimizer frequencies into a fixed-size histogram.
/// Each minimizer value is hashed to one of `NUM_HISTOGRAM_BUCKETS` buckets.
/// This gives sufficient resolution to partition into ~128 balanced bins
/// while using only 512 KB instead of a multi-GB HashMap.
pub fn count_minimizer_histogram(
    params: &Params,
    budget: &InFlightBudget,
) -> anyhow::Result<Vec<u64>> {
    let m = params.minimizer_len();
    let k = params.k;
    let w = k - m + 1;

    info!(
        "Counting minimizer frequencies with m={}, w={}, k={}",
        m, w, k
    );

    let histogram: Vec<AtomicU64> = (0..NUM_HISTOGRAM_BUCKETS)
        .map(|_| AtomicU64::new(0))
        .collect();

    rayon::in_place_scope(|s| {
        for input_file in &params.input_files {
            let mut reader = needletail::parse_fastx_file(input_file)
                .expect("failed to open FASTA file");

            while let Some(result) = reader.next() {
                let record = result.expect("invalid FASTA record");
                let seq = record.seq();
                let seq_len = seq.len();
                if seq_len < k {
                    continue;
                }

                if seq_len < CHUNK_BASES {
                    // Small sequence: copy and spawn async for between-sequence parallelism.
                    budget.wait_and_acquire(seq_len);
                    let seq_owned = seq.to_vec();
                    let histogram = &histogram;
                    let budget = &budget;

                    s.spawn(move |_| {
                        count_minimizers_chunk(&seq_owned, m, w, histogram);
                        budget.release(seq_len);
                    });
                } else {
                    // Large sequence: process synchronously with borrowed data (zero copy).
                    // Use fixed-size chunks so simd_minimizers internal allocations stay
                    // small enough for the system allocator to reuse freed pages.
                    rayon::scope(|inner| {
                        let mut start = 0;
                        while start < seq_len {
                            let end = (start + CHUNK_BASES).min(seq_len);
                            let chunk = &seq[start..end];
                            let histogram = &histogram;

                            inner.spawn(move |_| {
                                count_minimizers_chunk(chunk, m, w, histogram);
                            });
                            start += CHUNK_BASES;
                        }
                    });
                }
            }
        }
    });

    let counts: Vec<u64> = histogram.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let total: u64 = counts.iter().sum();
    let nonzero = counts.iter().filter(|&&c| c > 0).count();
    info!(
        "Histogram: {} total minimizer occurrences across {} non-empty buckets",
        total, nonzero
    );
    Ok(counts)
}

/// Count minimizers in a sequence chunk into the shared histogram.
fn count_minimizers_chunk(seq: &[u8], m: usize, w: usize, histogram: &[AtomicU64]) {
    let mut positions: Vec<u32> = Vec::new();
    let ascii_seq = simd_minimizers::packed_seq::AsciiSeq(seq);
    let output = simd_minimizers::canonical_minimizers(m, w).run(ascii_seq, &mut positions);

    for (_pos, val) in output.pos_and_values_u64() {
        let bucket = bucket_hash(val);
        histogram[bucket].fetch_add(1, Ordering::Relaxed);
    }
}

/// Partition minimizers into bins based on the histogram.
/// Walks buckets left-to-right, placing bin boundaries when the accumulated
/// count reaches the target per bin. Boundaries are expressed as bucket indices;
/// `bin_for_minimizer` hashes values to buckets using the same hash function.
pub fn partition_minimizers(
    histogram: &[u64],
    num_bins: usize,
) -> anyhow::Result<Partitioning> {
    let total_count: u64 = histogram.iter().sum();
    let target_per_bin = total_count / num_bins as u64;

    let mut boundaries = Vec::new();
    let mut current_count = 0u64;

    for (bucket, &count) in histogram.iter().enumerate() {
        current_count += count;

        // Place a boundary when accumulated count reaches the target.
        // Carry forward excess so heavy buckets don't "waste" bin slots.
        if current_count >= target_per_bin && boundaries.len() < num_bins - 1 {
            boundaries.push(bucket as u16);
            current_count -= target_per_bin;
        }
    }

    info!(
        "Partitioned into {} bins (target {}/bin)",
        boundaries.len() + 1,
        target_per_bin
    );

    Ok(Partitioning::from_boundaries(&boundaries))
}

/// Minimizer partitioning: bucket-based bin assignment.
/// Uses a lookup table mapping each of 65,536 histogram buckets to a bin index.
pub struct Partitioning {
    /// For each histogram bucket, the bin it belongs to.
    bucket_to_bin: Vec<u16>,
    num_bins: usize,
}

impl Partitioning {
    /// Build the bucket→bin lookup table from sorted bucket boundaries.
    fn from_boundaries(boundaries: &[u16]) -> Self {
        let num_bins = boundaries.len() + 1;
        let mut bucket_to_bin = vec![0u16; NUM_HISTOGRAM_BUCKETS];
        let mut bin = 0u16;
        let mut boundary_idx = 0;
        for (bucket, entry) in bucket_to_bin.iter_mut().enumerate() {
            if boundary_idx < boundaries.len() && bucket as u16 > boundaries[boundary_idx] {
                bin += 1;
                boundary_idx += 1;
            }
            *entry = bin;
        }
        Partitioning {
            bucket_to_bin,
            num_bins,
        }
    }

    /// Look up which bin a minimizer value belongs to.
    #[inline]
    pub fn bin_for_minimizer(&self, val: u64) -> usize {
        let bucket = bucket_hash(val);
        self.bucket_to_bin[bucket] as usize
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

