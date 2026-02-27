use crate::budget::InFlightBudget;
use crate::dna::{is_placeholder, Base};
use crate::minimizer::Partitioning;
use crate::params::Params;
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
                    let partitioning = &partitioning;
                    let writers = &writers;
                    let budget = &budget;

                    s.spawn(move |_| {
                        route_chunk(&seq_owned, m, w, k, num_bins, partitioning, writers);
                        budget.release(seq_len);
                    });
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
                            let partitioning = &partitioning;
                            let writers = &writers;

                            inner.spawn(move |_| {
                                route_chunk(chunk, m, w, k, num_bins, partitioning, writers);
                            });
                            start += CHUNK_BASES;
                        }
                    });
                }
            }
        }
    });

    // Finish all writers.
    for writer in &writers {
        writer.lock().unwrap().flush().unwrap();
    }
    drop(writers);

    info!("Finished routing super k-mers to bin files");
    Ok(())
}

/// Route super k-mers from a sequence chunk to bin files.
fn route_chunk(
    seq: &[u8],
    m: usize,
    w: usize,
    k: usize,
    num_bins: usize,
    partitioning: &Partitioning,
    writers: &[Mutex<BinWriter>],
) {
    let mut positions: Vec<u32> = Vec::new();
    let mut skmer_positions: Vec<u32> = Vec::new();
    let mut buffers: Vec<Vec<u8>> = vec![Vec::new(); num_bins];
    let mut pack_buf: Vec<u8> = Vec::new();

    let ascii_seq = simd_minimizers::packed_seq::AsciiSeq(seq);
    let output = simd_minimizers::canonical_minimizers(m, w)
        .super_kmers(&mut skmer_positions)
        .run(ascii_seq, &mut positions);

    let pos_vals: Vec<(u32, u64)> = output.pos_and_values_u64().collect();
    let num_skmers = pos_vals.len();

    for sk_idx in 0..num_skmers {
        let start_kmer_pos = skmer_positions[sk_idx] as usize;
        let end_kmer_pos = if sk_idx + 1 < num_skmers {
            skmer_positions[sk_idx + 1] as usize
        } else {
            seq.len() - k + 1
        };

        if end_kmer_pos <= start_kmer_pos {
            continue;
        }

        let seq_start = start_kmer_pos;
        let seq_end = end_kmer_pos - 1 + k;

        if seq_end > seq.len() {
            continue;
        }

        let subseq = &seq[seq_start..seq_end];
        let hash = pos_vals[sk_idx].1;
        let bin = partitioning.bin_for_minimizer(hash);

        write_packed_segments(subseq, k, bin, &mut buffers, writers, &mut pack_buf);
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
) {
    // Find N-free segments.
    let mut seg_start = 0;
    for i in 0..subseq.len() {
        if is_placeholder(subseq[i]) {
            if i - seg_start >= k {
                pack_and_buffer(&subseq[seg_start..i], bin, buffers, writers, pack_buf);
            }
            seg_start = i + 1;
        }
    }
    // Final segment.
    if subseq.len() - seg_start >= k {
        pack_and_buffer(&subseq[seg_start..], bin, buffers, writers, pack_buf);
    }
}

/// Pack an N-free ASCII segment to 2-bit and append to the bin buffer.
fn pack_and_buffer(
    ascii: &[u8],
    bin: usize,
    buffers: &mut [Vec<u8>],
    writers: &[Mutex<BinWriter>],
    pack_buf: &mut Vec<u8>,
) {
    let base_len = ascii.len() as u16;
    let packed_bytes = ascii.len().div_ceil(4);

    // Pack into reusable buffer.
    pack_buf.clear();
    pack_buf.resize(packed_bytes, 0);
    for (i, &ch) in ascii.iter().enumerate() {
        let base = Base::map_base(ch) as u8;
        pack_buf[i / 4] |= base << (6 - 2 * (i % 4));
    }

    // Append length + packed data to bin buffer.
    let buf = &mut buffers[bin];
    buf.extend_from_slice(&base_len.to_le_bytes());
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
