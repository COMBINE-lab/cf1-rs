use crate::budget::InFlightBudget;
use crate::directed_kmer::{AnnotatedKmer, Direction, FWD};
use crate::dna::{complement_char, is_placeholder, to_upper};
use crate::kmer::{Kmer, KmerBits};
use crate::mphf::Mphf;
use crate::params::Params;
use crate::state::StateClass;
use crate::state_vector::AtomicStateVector;
use std::io::Write;
use std::sync::Mutex;
use tracing::info;

/// Oriented unitig: ID, direction, and k-mer index range.
#[derive(Clone, Copy, Debug)]
pub struct OrientedUnitig {
    pub unitig_id: u64,
    pub dir: Direction,
    pub start_kmer_idx: usize,
    pub end_kmer_idx: usize,
}

impl OrientedUnitig {
    pub fn is_valid(&self) -> bool {
        self.unitig_id != u64::MAX
    }

    pub fn length(&self, k: usize) -> usize {
        self.end_kmer_idx - self.start_kmer_idx + k
    }
}

/// Unitig extraction metadata.
pub struct UnipathsMeta {
    pub unipath_count: u64,
    pub kmer_count: u64,
    pub max_len: usize,
    pub min_len: usize,
    pub sum_len: u64,
}

impl Default for UnipathsMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl UnipathsMeta {
    pub fn new() -> Self {
        UnipathsMeta {
            unipath_count: 0,
            kmer_count: 0,
            max_len: 0,
            min_len: usize::MAX,
            sum_len: 0,
        }
    }

    pub fn add_unitig(&mut self, vertex_count: usize, k: usize) {
        self.unipath_count += 1;
        self.kmer_count += vertex_count as u64;
        let len = vertex_count + k - 1;
        self.max_len = self.max_len.max(len);
        self.min_len = self.min_len.min(len);
        self.sum_len += len as u64;
    }

    pub fn aggregate(&mut self, other: &UnipathsMeta) {
        self.unipath_count += other.unipath_count;
        self.kmer_count += other.kmer_count;
        self.max_len = self.max_len.max(other.max_len);
        self.min_len = self.min_len.min(other.min_len);
        self.sum_len += other.sum_len;
    }
}

/// Unitig boundary detection: is this k-mer the start of a unipath?
/// Matches C++ CdBG::is_unipath_start exactly.
fn is_unipath_start(
    state_class: StateClass,
    dir: Direction,
    prev_kmer_class: StateClass,
    prev_kmer_dir: Direction,
) -> bool {
    if state_class == StateClass::MultiInMultiOut {
        return true;
    }

    if dir == FWD {
        if state_class == StateClass::MultiInSingleOut {
            return true;
        }
    } else if state_class == StateClass::SingleInMultiOut {
        return true;
    }

    if prev_kmer_class == StateClass::MultiInMultiOut {
        return true;
    }

    if prev_kmer_dir == FWD {
        if prev_kmer_class == StateClass::SingleInMultiOut {
            return true;
        }
    } else if prev_kmer_class == StateClass::MultiInSingleOut {
        return true;
    }

    false
}

/// Unitig boundary detection: is this k-mer the end of a unipath?
/// Matches C++ CdBG::is_unipath_end exactly.
fn is_unipath_end(
    state_class: StateClass,
    dir: Direction,
    next_kmer_class: StateClass,
    next_kmer_dir: Direction,
) -> bool {
    if state_class == StateClass::MultiInMultiOut {
        return true;
    }

    if dir == FWD {
        if state_class == StateClass::SingleInMultiOut {
            return true;
        }
    } else if state_class == StateClass::MultiInSingleOut {
        return true;
    }

    if next_kmer_class == StateClass::MultiInMultiOut {
        return true;
    }

    if next_kmer_dir == FWD {
        if next_kmer_class == StateClass::MultiInSingleOut {
            return true;
        }
    } else if next_kmer_class == StateClass::SingleInMultiOut {
        return true;
    }

    false
}

/// Threshold for flushing per-thread segment buffer to global writer.
const SEG_BUFFER_THRESHOLD: usize = 100 * 1024; // 100 KB

/// Extract unitigs and write output for GFA-reduced format (format 3).
/// Uses rayon::scope + spawn for work-stealing scheduling across sequences.
pub fn extract_and_output<const K: usize>(
    params: &Params,
    mphf: &Mphf<K>,
    states: &AtomicStateVector,
    budget: &InFlightBudget,
) -> anyhow::Result<UnipathsMeta>
where
    Kmer<K>: KmerBits,
{
    info!("Phase 5: Extracting unitigs and writing output");

    // Open output files.
    let seg_path = params.segment_file_path();
    let seq_path = params.sequence_file_path();

    let seg_file = std::fs::File::create(&seg_path)?;
    let seg_writer = Mutex::new(std::io::BufWriter::new(seg_file));

    let seq_file = std::fs::File::create(&seq_path)?;
    let seq_writer = Mutex::new(std::io::BufWriter::new(seq_file));

    let global_meta = Mutex::new(UnipathsMeta::new());

    rayon::in_place_scope(|s| {
        for (file_idx, input_file) in params.input_files.iter().enumerate() {
            let ref_id = (file_idx + 1) as u64;
            let mut reader = needletail::parse_fastx_file(input_file)
                .expect("failed to open FASTA file");

            while let Some(result) = reader.next() {
                let record = result.expect("invalid FASTA record");
                let seq = record.seq();
                let seq_len = seq.len();

                if seq_len < K {
                    continue;
                }

                let seq_name = std::str::from_utf8(record.id())
                    .unwrap_or("")
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();

                let poly_n_stretch = params.poly_n_stretch;
                let seg_writer = &seg_writer;
                let seq_writer = &seq_writer;
                let global_meta = &global_meta;
                let budget = &budget;
                // Sequences under 4 MB are copied and spawned async for
                // between-sequence parallelism. Larger ones are processed
                // synchronously with zero copy.
                const LARGE_SEQ_THRESHOLD: usize = 4_000_000;

                // Closure to build tiling line and write output.
                let write_tiling = move |unitigs: &[OrientedUnitig],
                                        local_meta: &UnipathsMeta| {
                    let tiling_estimate = 64 + seq_name.len() + unitigs.len() * 14;
                    let mut tiling = Vec::with_capacity(tiling_estimate);
                    let mut ibuf = itoa::Buffer::new();

                    tiling.extend_from_slice(b"Reference:");
                    tiling.extend_from_slice(ibuf.format(ref_id).as_bytes());
                    tiling.extend_from_slice(b"_Sequence:");
                    tiling.extend_from_slice(seq_name.as_bytes());
                    tiling.push(b'\t');

                    if !unitigs.is_empty() {
                        let first = &unitigs[0];

                        if poly_n_stretch && first.start_kmer_idx > 0 {
                            tiling.push(b'N');
                            tiling.extend_from_slice(ibuf.format(first.start_kmer_idx).as_bytes());
                            tiling.push(b' ');
                        }

                        tiling.extend_from_slice(ibuf.format(first.unitig_id).as_bytes());
                        tiling.push(if first.dir == FWD { b'+' } else { b'-' });

                        for i in 1..unitigs.len() {
                            let left = &unitigs[i - 1];
                            let right = &unitigs[i];

                            if poly_n_stretch
                                && (left.end_kmer_idx + 1) != right.start_kmer_idx
                            {
                                let nuc_gap =
                                    right.start_kmer_idx as i64 - left.end_kmer_idx as i64;
                                if nuc_gap >= (K as i64 + 1) {
                                    let polyn_gap = nuc_gap as usize - K;
                                    tiling.push(b' ');
                                    tiling.push(b'N');
                                    tiling.extend_from_slice(ibuf.format(polyn_gap).as_bytes());
                                }
                            }

                            tiling.push(b' ');
                            tiling.extend_from_slice(ibuf.format(right.unitig_id).as_bytes());
                            tiling.push(if right.dir == FWD { b'+' } else { b'-' });
                        }
                    }

                    tiling.push(b'\n');

                    seq_writer.lock().unwrap().write_all(&tiling).ok();
                    global_meta.lock().unwrap().aggregate(local_meta);
                };

                if seq_len < LARGE_SEQ_THRESHOLD {
                    // Small sequence: copy and spawn async for between-sequence parallelism.
                    budget.wait_and_acquire(seq_len);
                    let seq_owned = seq.to_vec();

                    s.spawn(move |_| {
                        let mut seg_buf = Vec::with_capacity(SEG_BUFFER_THRESHOLD + 4096);
                        let mut local_meta = UnipathsMeta::new();

                        let unitigs = extract_unitigs_from_seq::<K>(
                            &seq_owned,
                            seq_len,
                            mphf,
                            states,
                            seg_writer,
                            &mut seg_buf,
                            K,
                            &mut local_meta,
                        );

                        if !seg_buf.is_empty() {
                            seg_writer.lock().unwrap().write_all(&seg_buf).ok();
                        }

                        write_tiling(&unitigs, &local_meta);
                        budget.release(seq_len);
                    });
                } else {
                    // Large sequence: process synchronously with borrowed data (zero copy).
                    let mut seg_buf = Vec::with_capacity(SEG_BUFFER_THRESHOLD + 4096);
                    let mut local_meta = UnipathsMeta::new();

                    let unitigs = extract_unitigs_from_seq::<K>(
                        &seq,
                        seq_len,
                        mphf,
                        states,
                        seg_writer,
                        &mut seg_buf,
                        K,
                        &mut local_meta,
                    );

                    if !seg_buf.is_empty() {
                        seg_writer.lock().unwrap().write_all(&seg_buf).ok();
                    }

                    write_tiling(&unitigs, &local_meta);
                }
            }
        }
    });

    // Flush.
    seg_writer.lock().unwrap().flush()?;
    seq_writer.lock().unwrap().flush()?;

    let result = global_meta.into_inner().unwrap();
    info!(
        "Extracted {} unitigs, {} vertices, avg length {}",
        result.unipath_count,
        result.kmer_count,
        if result.unipath_count > 0 {
            result.sum_len / result.unipath_count
        } else {
            0
        }
    );

    Ok(result)
}

/// Extract unitigs from a single sequence, buffering segment output.
#[allow(clippy::too_many_arguments)]
fn extract_unitigs_from_seq<const K: usize>(
    seq: &[u8],
    seq_len: usize,
    mphf: &Mphf<K>,
    states: &AtomicStateVector,
    seg_writer: &Mutex<std::io::BufWriter<std::fs::File>>,
    seg_buf: &mut Vec<u8>,
    k: usize,
    meta: &mut UnipathsMeta,
) -> Vec<OrientedUnitig>
where
    Kmer<K>: KmerBits,
{
    // Pre-allocate: ~1 unitig per 100 k-mers on average for genome data.
    let estimated = (seq_len.saturating_sub(K) + 100) / 100;
    let mut unitigs = Vec::with_capacity(estimated.max(64));
    let right_end = seq_len - K;

    let mut kmer_idx = 0usize;
    while kmer_idx <= right_end {
        kmer_idx = search_valid_kmer::<K>(seq, kmer_idx, right_end);
        if kmer_idx > right_end {
            break;
        }
        kmer_idx = output_unitigs_contiguous::<K>(
            seq,
            seq_len,
            right_end,
            kmer_idx,
            mphf,
            states,
            seg_writer,
            seg_buf,
            k,
            &mut unitigs,
            meta,
        );
    }

    unitigs
}

fn search_valid_kmer<const K: usize>(seq: &[u8], mut start: usize, right_end: usize) -> usize
where
    Kmer<K>: KmerBits,
{
    while start <= right_end {
        let mut valid = true;
        for j in 0..K {
            if is_placeholder(seq[start + j]) {
                start = start + j + 1;
                valid = false;
                break;
            }
        }
        if valid {
            return start;
        }
    }
    right_end + 1
}

#[allow(clippy::too_many_arguments)]
fn output_unitigs_contiguous<const K: usize>(
    seq: &[u8],
    seq_len: usize,
    right_end: usize,
    start_idx: usize,
    mphf: &Mphf<K>,
    states: &AtomicStateVector,
    seg_writer: &Mutex<std::io::BufWriter<std::fs::File>>,
    seg_buf: &mut Vec<u8>,
    k: usize,
    unitigs: &mut Vec<OrientedUnitig>,
    meta: &mut UnipathsMeta,
) -> usize
where
    Kmer<K>: KmerBits,
{
    let mut kmer_idx = start_idx;

    let mut curr_kmer =
        AnnotatedKmer::<K>::new(Kmer::<K>::from_ascii(seq, kmer_idx), kmer_idx, mphf, states);

    // Isolated k-mer.
    let no_left = kmer_idx == 0 || is_placeholder(seq[kmer_idx - 1]);
    let no_right = kmer_idx + K == seq_len || is_placeholder(seq[kmer_idx + K]);

    if no_left && no_right {
        output_gfa_unitig::<K>(
            seq,
            &curr_kmer,
            &curr_kmer,
            mphf,
            states,
            seg_writer,
            seg_buf,
            k,
            unitigs,
            meta,
        );
    } else {
        if no_right {
            if no_left {
                output_gfa_unitig::<K>(
                    seq,
                    &curr_kmer,
                    &curr_kmer,
                    mphf,
                    states,
                    seg_writer,
                    seg_buf,
                    k,
                    unitigs,
                    meta,
                );
            } else {
                let prev_kmer = AnnotatedKmer::<K>::new(
                    Kmer::<K>::from_ascii(seq, kmer_idx - 1),
                    kmer_idx,
                    mphf,
                    states,
                );
                if is_unipath_start(
                    curr_kmer.state_class,
                    curr_kmer.dir(),
                    prev_kmer.state_class,
                    prev_kmer.dir(),
                ) {
                    output_gfa_unitig::<K>(
                        seq,
                        &curr_kmer,
                        &curr_kmer,
                        mphf,
                        states,
                        seg_writer,
                        seg_buf,
                        k,
                        unitigs,
                        meta,
                    );
                }
            }
            return kmer_idx + K;
        }

        let mut next_kmer = curr_kmer;
        next_kmer.roll_to_next_kmer(seq[kmer_idx + K], mphf, states);

        let mut on_unipath = false;
        let mut unipath_start_kmer = curr_kmer;
        #[allow(unused_assignments)]
        let mut prev_kmer = curr_kmer;

        if no_left {
            on_unipath = true;
            unipath_start_kmer = curr_kmer;
        } else {
            prev_kmer = AnnotatedKmer::<K>::new(
                Kmer::<K>::from_ascii(seq, kmer_idx - 1),
                kmer_idx,
                mphf,
                states,
            );
            if is_unipath_start(
                curr_kmer.state_class,
                curr_kmer.dir(),
                prev_kmer.state_class,
                prev_kmer.dir(),
            ) {
                on_unipath = true;
                unipath_start_kmer = curr_kmer;
            }
        }

        if on_unipath
            && is_unipath_end(
                curr_kmer.state_class,
                curr_kmer.dir(),
                next_kmer.state_class,
                next_kmer.dir(),
            )
        {
            output_gfa_unitig::<K>(
                seq,
                &unipath_start_kmer,
                &curr_kmer,
                mphf,
                states,
                seg_writer,
                seg_buf,
                k,
                unitigs,
                meta,
            );
            on_unipath = false;
        }

        kmer_idx += 1;
        loop {
            if !on_unipath && kmer_idx > right_end {
                break;
            }

            prev_kmer = curr_kmer;
            curr_kmer = next_kmer;

            if is_unipath_start(
                curr_kmer.state_class,
                curr_kmer.dir(),
                prev_kmer.state_class,
                prev_kmer.dir(),
            ) {
                on_unipath = true;
                unipath_start_kmer = curr_kmer;
            }

            // No valid right neighbor.
            if kmer_idx + K == seq_len || is_placeholder(seq[kmer_idx + K]) {
                if on_unipath {
                    output_gfa_unitig::<K>(
                        seq,
                        &unipath_start_kmer,
                        &curr_kmer,
                        mphf,
                        states,
                        seg_writer,
                        seg_buf,
                        k,
                        unitigs,
                        meta,
                    );
                }
                return kmer_idx + K;
            }

            next_kmer.roll_to_next_kmer(seq[kmer_idx + K], mphf, states);

            if on_unipath
                && is_unipath_end(
                    curr_kmer.state_class,
                    curr_kmer.dir(),
                    next_kmer.state_class,
                    next_kmer.dir(),
                )
            {
                output_gfa_unitig::<K>(
                    seq,
                    &unipath_start_kmer,
                    &curr_kmer,
                    mphf,
                    states,
                    seg_writer,
                    seg_buf,
                    k,
                    unitigs,
                    meta,
                );
                on_unipath = false;
            }

            kmer_idx += 1;
        }
    }

    kmer_idx + K
}

/// Output a single GFA unitig (segment) and record it.
/// Writes to the per-thread seg_buf; flushes to seg_writer when threshold is exceeded.
#[allow(clippy::too_many_arguments)]
fn output_gfa_unitig<const K: usize>(
    seq: &[u8],
    start_kmer: &AnnotatedKmer<K>,
    end_kmer: &AnnotatedKmer<K>,
    mphf: &Mphf<K>,
    states: &AtomicStateVector,
    seg_writer: &Mutex<std::io::BufWriter<std::fs::File>>,
    seg_buf: &mut Vec<u8>,
    k: usize,
    unitigs: &mut Vec<OrientedUnitig>,
    meta: &mut UnipathsMeta,
) where
    Kmer<K>: KmerBits,
{
    let min_flanking_kmer = std::cmp::min(*start_kmer.canonical(), *end_kmer.canonical());
    let bucket_id = mphf.bucket_id(&min_flanking_kmer);

    let unitig_id = bucket_id;
    let unitig_dir = if *start_kmer.kmer() < *end_kmer.rev_compl() {
        FWD
    } else {
        !FWD
    };

    let current_unitig = OrientedUnitig {
        unitig_id,
        dir: unitig_dir,
        start_kmer_idx: start_kmer.idx,
        end_kmer_idx: end_kmer.idx,
    };

    // Try to claim this unitig for output via CAS.
    let idx = bucket_id as usize;
    let state = states.get(idx);

    if !state.is_outputted() {
        let outputted_state = state.outputted();
        if states.compare_and_swap(idx, state, outputted_state) {
            // We claimed it — write the segment to per-thread buffer.
            let segment_len = end_kmer.idx - start_kmer.idx + k;

            let mut ibuf = itoa::Buffer::new();
            seg_buf.extend_from_slice(ibuf.format(unitig_id).as_bytes());
            seg_buf.push(b'\t');
            if unitig_dir == FWD {
                for offset in 0..segment_len {
                    seg_buf.push(to_upper(seq[start_kmer.idx + offset]));
                }
            } else {
                for offset in 0..segment_len {
                    seg_buf.push(complement_char(seq[end_kmer.idx + k - 1 - offset]));
                }
            }
            seg_buf.push(b'\n');

            // Flush to global writer when buffer exceeds threshold.
            if seg_buf.len() >= SEG_BUFFER_THRESHOLD {
                let mut w = seg_writer.lock().unwrap();
                w.write_all(seg_buf).ok();
                seg_buf.clear();
            }

            let vertex_count = end_kmer.idx - start_kmer.idx + 1;
            meta.add_unitig(vertex_count, k);
        }
    }

    unitigs.push(current_unitig);
}

/// Write JSON output file.
pub fn write_json(
    params: &Params,
    meta: &UnipathsMeta,
    short_seqs: &[(String, usize)],
) -> anyhow::Result<()> {
    let json_path = params.json_file_path();

    let input_str = params
        .input_files
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let output_prefix = params.output.display().to_string();

    let mut json = serde_json::json!({
        "parameters info": {
            "input": input_str,
            "k": params.k,
            "output prefix": output_prefix,
        },
        "basic info": {
            "vertex count": meta.kmer_count,
        },
        "contigs info": {
            "maximal unitig count": meta.unipath_count,
            "vertex count in the maximal unitigs": meta.kmer_count,
            "shortest maximal unitig length": if meta.min_len == usize::MAX { 0 } else { meta.min_len },
            "longest maximal unitig length": meta.max_len,
            "sum maximal unitig length": meta.sum_len,
            "avg. maximal unitig length": if meta.unipath_count > 0 { meta.sum_len / meta.unipath_count } else { 0 },
            "_comment": "lengths are in bases",
        },
    });

    if !short_seqs.is_empty() {
        let short_seqs_json: Vec<serde_json::Value> = short_seqs
            .iter()
            .map(|(name, len)| serde_json::json!([name, len]))
            .collect();
        json["short seqs"] = serde_json::Value::Array(short_seqs_json);
    }

    let json_str = serde_json::to_string_pretty(&json)?;
    std::fs::write(&json_path, json_str)?;

    info!("Wrote JSON output to {}", json_path.display());
    Ok(())
}
