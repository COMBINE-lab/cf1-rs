use crate::params::Params;
use std::collections::HashMap;
use std::io::Read;
use std::sync::Mutex;
use tracing::info;

/// Count minimizer frequencies across all input FASTA sequences.
/// Uses rayon::scope + spawn to overlap gzip decompression with computation.
pub fn count_minimizer_frequencies(params: &Params) -> anyhow::Result<HashMap<u64, u64>> {
    let m = params.minimizer_len();
    let k = params.k;
    let w = k - m + 1;

    info!(
        "Counting minimizer frequencies with m={}, w={}, k={}",
        m, w, k
    );

    let global_counts: Mutex<HashMap<u64, u64>> = Mutex::new(HashMap::new());

    rayon::in_place_scope(|s| {
        for input_file in &params.input_files {
            let reader = open_fasta(input_file).unwrap();
            let mut fasta_reader = paraseq::fasta::Reader::new(reader);
            let mut record_set = fasta_reader.new_record_set();

            while record_set.fill(&mut fasta_reader).unwrap() {
                for record_result in record_set.iter() {
                    let record = record_result.unwrap();
                    let seq = record.seq();
                    if seq.len() < k {
                        continue;
                    }

                    let seq_owned = seq.to_vec();
                    let global_counts = &global_counts;

                    s.spawn(move |_| {
                        let mut positions: Vec<u32> = Vec::new();
                        let mut local_counts: HashMap<u64, u64> = HashMap::new();

                        let ascii_seq = simd_minimizers::packed_seq::AsciiSeq(&seq_owned);
                        let output = simd_minimizers::canonical_minimizers(m, w)
                            .run(ascii_seq, &mut positions);

                        for (_pos, hash) in output.pos_and_values_u64() {
                            *local_counts.entry(hash).or_insert(0) += 1;
                        }

                        // Merge local counts into global.
                        let mut global = global_counts.lock().unwrap();
                        for (hash, count) in local_counts {
                            *global.entry(hash).or_insert(0) += count;
                        }
                    });
                }
            }
        }
    });

    let counts = global_counts.into_inner().unwrap();
    info!(
        "Found {} distinct minimizers",
        counts.len()
    );
    Ok(counts)
}

/// Partition minimizers into bins for balanced work distribution.
/// Returns sorted bin boundaries as (start_hash, end_hash) pairs.
pub fn partition_minimizers(
    counts: &HashMap<u64, u64>,
    num_bins: usize,
) -> anyhow::Result<Partitioning> {
    let mut sorted: Vec<(u64, u64)> = counts.iter().map(|(&h, &c)| (h, c)).collect();
    sorted.sort_unstable_by_key(|&(h, _)| h);

    let total_count: u64 = sorted.iter().map(|&(_, c)| c).sum();
    let target_per_bin = total_count / num_bins as u64;

    let mut boundaries = Vec::new();
    let mut current_count = 0u64;

    for (_i, &(hash, count)) in sorted.iter().enumerate() {
        current_count += count;

        if current_count >= target_per_bin && boundaries.len() < num_bins - 1 {
            boundaries.push(hash);
            current_count = 0;
        }
    }

    info!(
        "Partitioned {} minimizers into {} bins (target {}/bin)",
        sorted.len(),
        boundaries.len() + 1,
        target_per_bin
    );

    Ok(Partitioning { boundaries })
}

/// Minimizer partitioning: sorted hash boundaries for bin assignment.
pub struct Partitioning {
    /// Sorted boundary hashes. bin_for_minimizer uses binary search.
    pub boundaries: Vec<u64>,
}

impl Partitioning {
    /// Look up which bin a minimizer hash belongs to.
    pub fn bin_for_minimizer(&self, hash: u64) -> usize {
        match self.boundaries.binary_search(&hash) {
            Ok(idx) => idx,
            Err(idx) => idx,
        }
    }

    pub fn num_bins(&self) -> usize {
        self.boundaries.len() + 1
    }
}

/// Open a FASTA file, transparently handling .gz.
pub fn open_fasta(path: &std::path::Path) -> anyhow::Result<Box<dyn Read + Send>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        Ok(Box::new(flate2::read::GzDecoder::new(reader)))
    } else {
        Ok(Box::new(reader))
    }
}
