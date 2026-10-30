use crate::batch::{BATCH_BASES, SeqBatch};
use crate::budget::InFlightBudget;
use crate::dna::{Base, is_placeholder};
use crate::minimizer::Partitioning;
use crate::params::Params;
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::info;

/// Flush threshold for per-thread bin buffers (bytes).
const FLUSH_THRESHOLD: usize = 64 * 1024;

/// Target chunk size (bases) for splitting large sequences across threads.
/// Keeps simd_minimizers internal allocations small enough for the system
/// allocator to efficiently reuse freed pages between chromosomes.
const CHUNK_BASES: usize = 4_000_000;

/// Write super k-mers to per-bin temp files in 2-bit packed format.
/// Uses rayon::in_place_scope + spawn for parallel processing of sequences,
/// with per-thread bin buffers that flush to shared writers when full.
pub fn route_superkmers(
    params: &Params,
    partitioning: &Partitioning,
    budget: &InFlightBudget,
) -> anyhow::Result<()> {
    let m = params.minimizer_len();
    let k = params.k;
    let w = k - m + 1;
    let num_bins = partitioning.num_bins();

    info!(
        "Routing super k-mers to {} bins with m={}, w={}",
        num_bins, m, w
    );

    // Create one mutex-protected writer per bin.
    let writers: Vec<Mutex<BinWriter>> = (0..num_bins)
        .map(|bin| {
            let path = bin_file_path(&params.work_dir, bin);
            Mutex::new(BinWriter::new(&path).expect("Failed to create bin writer"))
        })
        .collect();

    let route = |seqs: &mut dyn Iterator<Item = &[u8]>| {
        route_chunk(
            seqs,
            m,
            w,
            k,
            num_bins,
            partitioning,
            &writers,
            params.poly_n_stretch,
        );
    };

    rayon::in_place_scope(|s| {
        let mut batch = SeqBatch::default();
        for input_file in &params.input_files {
            let mut reader =
                needletail::parse_fastx_file(input_file).expect("failed to open FASTA file");

            while let Some(result) = reader.next() {
                let record = result.expect("invalid FASTA record");
                let seq = record.seq();
                let seq_len = seq.len();
                if seq_len < k {
                    continue;
                }

                if seq_len < CHUNK_BASES {
                    // Small sequences are batched for between-sequence parallelism
                    // (and so that a task's per-bin buffers fill before flushing).
                    batch.push(&seq);
                    if batch.bases() >= BATCH_BASES {
                        std::mem::take(&mut batch)
                            .spawn(s, budget, |batch| route(&mut batch.iter()));
                    }
                } else {
                    // Large sequence: process synchronously with borrowed data (zero copy).
                    // Use fixed-size chunks so simd_minimizers internal allocations stay
                    // small enough for the system allocator to reuse freed pages.
                    let overlap = k - 1;
                    rayon::scope(|inner| {
                        let mut start = 0;
                        while start < seq_len {
                            let end = (start + CHUNK_BASES + overlap).min(seq_len);
                            let chunk = &seq[start..end];
                            let route = &route;

                            inner.spawn(move |_| route(&mut std::iter::once(chunk)));
                            start += CHUNK_BASES;
                        }
                    });
                }
            }
        }
        batch.spawn(s, budget, |batch| route(&mut batch.iter()));
    });

    // Finish all writers.
    for writer in &writers {
        writer.lock().unwrap().flush().unwrap();
    }
    drop(writers);

    info!("Finished routing super k-mers to bin files");
    Ok(())
}

/// Route the super k-mers of a run of sequence chunks to bin files.
#[allow(clippy::too_many_arguments)]
fn route_chunk(
    seqs: &mut dyn Iterator<Item = &[u8]>,
    m: usize,
    w: usize,
    k: usize,
    num_bins: usize,
    partitioning: &Partitioning,
    writers: &[Mutex<BinWriter>],
    poly_n: bool,
) {
    let mut positions: Vec<u32> = Vec::new();
    let mut skmer_positions: Vec<u32> = Vec::new();
    let mut buffers: Vec<Vec<u8>> = vec![Vec::new(); num_bins];
    let mut pack_buf: Vec<u8> = Vec::new();
    let mut routed = RoutedSuperkmers::default();

    // Route the super k-mers of one ACGT-only (sub)sequence. Only k-mer *content*
    // is packed (positions are not stored), so each segment is processed
    // independently with its own local positions — no global offset is needed.
    let mut route_segment =
        |seg: &[u8], positions: &mut Vec<u32>, skmer_positions: &mut Vec<u32>| {
            positions.clear();
            skmer_positions.clear();
            let ascii_seq = simd_minimizers::packed_seq::AsciiSeq(seg);
            let output = simd_minimizers::canonical_minimizers(m, w)
                .super_kmers(skmer_positions)
                .run(ascii_seq, positions);

            let pos_vals: Vec<(u32, u64)> = output.pos_and_values_u64().collect();
            let num_skmers = pos_vals.len();

            for sk_idx in 0..num_skmers {
                let start_kmer_pos = skmer_positions[sk_idx] as usize;
                let end_kmer_pos = if sk_idx + 1 < num_skmers {
                    skmer_positions[sk_idx + 1] as usize
                } else {
                    seg.len() - k + 1
                };

                if end_kmer_pos <= start_kmer_pos {
                    continue;
                }

                let seq_start = start_kmer_pos;
                let seq_end = end_kmer_pos - 1 + k;

                if seq_end > seg.len() {
                    continue;
                }

                let subseq = &seg[seq_start..seq_end];
                let hash = pos_vals[sk_idx].1;
                let bin = partitioning.bin_for_minimizer(hash);

                write_packed_segments(
                    subseq,
                    k,
                    bin,
                    &mut buffers,
                    writers,
                    &mut pack_buf,
                    &mut routed,
                );
            }
        };

    for seq in seqs {
        if poly_n {
            // Split on placeholder (`N`) bases so packed-seq only sees ACGT (it
            // panics otherwise); a super k-mer never spans a placeholder.
            crate::dna::for_each_acgt_segment(seq, k, |_seg_start, seg| {
                route_segment(seg, &mut positions, &mut skmer_positions)
            });
        } else {
            // No N handling requested: pass the raw chunk straight through. An
            // input containing `N` panics here (packed-seq), as before — see the
            // rationale in `minimizer::count_minimizers_chunk`.
            route_segment(seq, &mut positions, &mut skmer_positions);
        }
    }

    flush_all_buffers(&mut buffers, writers);
}

/// Split a super k-mer at N positions and write N-free segments as packed 2-bit.
fn write_packed_segments(
    subseq: &[u8],
    k: usize,
    bin: usize,
    buffers: &mut [Vec<u8>],
    writers: &[Mutex<BinWriter>],
    pack_buf: &mut Vec<u8>,
    routed: &mut RoutedSuperkmers,
) {
    // Find N-free segments.
    let mut seg_start = 0;
    for i in 0..subseq.len() {
        if is_placeholder(subseq[i]) {
            if i - seg_start >= k {
                pack_and_buffer(
                    &subseq[seg_start..i],
                    bin,
                    buffers,
                    writers,
                    pack_buf,
                    routed,
                );
            }
            seg_start = i + 1;
        }
    }
    // Final segment.
    if subseq.len() - seg_start >= k {
        pack_and_buffer(
            &subseq[seg_start..],
            bin,
            buffers,
            writers,
            pack_buf,
            routed,
        );
    }
}

/// The super k-mers a task has already routed, so exact repeats can be dropped.
///
/// Isoforms of a gene are usually adjacent in a transcriptome and share most
/// of their sequence, so a batch repeats many super k-mers verbatim. A repeat
/// adds nothing to the distinct k-mers its bin yields, only bytes to write and
/// k-mers to sort and deduplicate in phase 3.
#[derive(Default)]
struct RoutedSuperkmers {
    /// Distinct packed records (length prefix + packed bases), back to back.
    records: Vec<u8>,
    /// XXH3 digest of a record -> its offset in `records`.
    index: HashMap<u64, usize, BuildNoHashHasher<u64>>,
}

impl RoutedSuperkmers {
    /// Record `record` and report whether it is new. Repeats are recognized
    /// by content, never by digest alone: on a digest collision between
    /// different records the second is simply reported as new.
    fn insert(&mut self, record: &[u8]) -> bool {
        match self.index.entry(xxhash_rust::xxh3::xxh3_64(record)) {
            Entry::Occupied(seen) => {
                let start = *seen.get();
                self.records.get(start..start + record.len()) != Some(record)
            }
            Entry::Vacant(slot) => {
                slot.insert(self.records.len());
                self.records.extend_from_slice(record);
                true
            }
        }
    }
}

/// Pack an N-free ASCII segment to 2-bit and append to the bin buffer, unless
/// this task has already routed the identical super k-mer.
fn pack_and_buffer(
    ascii: &[u8],
    bin: usize,
    buffers: &mut [Vec<u8>],
    writers: &[Mutex<BinWriter>],
    pack_buf: &mut Vec<u8>,
    routed: &mut RoutedSuperkmers,
) {
    let base_len = ascii.len() as u16;
    let packed_bytes = ascii.len().div_ceil(4);

    // Pack the record (length + packed data) into the reusable buffer.
    pack_buf.clear();
    pack_buf.extend_from_slice(&base_len.to_le_bytes());
    pack_buf.resize(2 + packed_bytes, 0);
    let packed = &mut pack_buf[2..];
    for (i, &ch) in ascii.iter().enumerate() {
        let base = Base::map_base(ch) as u8;
        packed[i / 4] |= base << (6 - 2 * (i % 4));
    }
    if !routed.insert(pack_buf) {
        return;
    }

    // Append the record to the bin buffer.
    let buf = &mut buffers[bin];
    buf.extend_from_slice(pack_buf);

    // Flush if buffer exceeds threshold.
    if buf.len() >= FLUSH_THRESHOLD {
        let mut w = writers[bin].lock().unwrap();
        w.write_all(buf).unwrap();
        buf.clear();
    }
}

/// Flush all non-empty buffers to their writers.
fn flush_all_buffers(buffers: &mut [Vec<u8>], writers: &[Mutex<BinWriter>]) {
    for (bin, buf) in buffers.iter_mut().enumerate() {
        if !buf.is_empty() {
            let mut w = writers[bin].lock().unwrap();
            w.write_all(buf).unwrap();
            buf.clear();
        }
    }
}

/// Get the path for a bin's temp file.
pub fn bin_file_path(work_dir: &Path, bin: usize) -> PathBuf {
    work_dir.join(format!("cf1_bin_{:05}.bin", bin))
}

/// Clean up bin temp files.
pub fn cleanup_bin_files(work_dir: &Path, num_bins: usize) {
    for bin in 0..num_bins {
        let path = bin_file_path(work_dir, bin);
        let _ = std::fs::remove_file(path);
    }
}

/// Raw binary writer for a single bin's packed super k-mers.
pub struct BinWriter {
    writer: BufWriter<std::fs::File>,
}

impl BinWriter {
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::with_capacity(64 * 1024, file);
        Ok(BinWriter { writer })
    }

    fn write_all(&mut self, data: &[u8]) -> anyhow::Result<()> {
        self.writer.write_all(data)?;
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Iterate the packed super k-mers of a bin file's contents, as
/// `(base_count, packed_bytes)` (the format [`BinReader`] reads). A truncated
/// trailing record ends the iteration.
pub fn packed_superkmers(data: &[u8]) -> impl Iterator<Item = (usize, &[u8])> {
    let mut rest = data;
    std::iter::from_fn(move || {
        let (len, tail) = rest.split_first_chunk::<2>()?;
        let base_count = u16::from_le_bytes(*len) as usize;
        let packed = tail.get(..base_count.div_ceil(4))?;
        rest = &tail[packed.len()..];
        Some((base_count, packed))
    })
}

/// Reader for a single bin's packed super k-mers.
pub struct BinReader {
    reader: BufReader<std::fs::File>,
    buf: Vec<u8>,
}

impl BinReader {
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(256 * 1024, file);
        Ok(BinReader {
            reader,
            buf: Vec::new(),
        })
    }

    /// Read the next packed super k-mer. Returns (base_count, packed_bytes) or None at EOF.
    pub fn read_packed_superkmer(&mut self) -> anyhow::Result<Option<(usize, &[u8])>> {
        let mut len_buf = [0u8; 2];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let base_count = u16::from_le_bytes(len_buf) as usize;
        let packed_bytes = base_count.div_ceil(4);
        self.buf.resize(packed_bytes, 0);
        self.reader.read_exact(&mut self.buf)?;
        Ok(Some((base_count, &self.buf)))
    }
}
