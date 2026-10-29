use crate::batch::{BATCH_BASES, SeqBatch};
use crate::budget::InFlightBudget;
use crate::directed_kmer::DirectedKmer;
use crate::directed_kmer::FWD;
use crate::dna::{Base, is_placeholder};
use crate::kmer::{Kmer, KmerBits};
use crate::mphf::Mphf;
use crate::state::{State, StateClass, Vertex};
use crate::state_vector::AtomicStateVector;
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

/// Target chunk size (k-mer positions) for splitting large sequences.
/// Keeps per-thread working set small for better allocator page reuse.
const CHUNK_KMERS: usize = 4_000_000;

/// Most consecutive k-mers whose MPHF lookups are issued together in
/// [`Classifier::process_contiguous_subseq`] (one window).
const LOOKUP_BATCH: usize = 32;

/// Whether `kmer` ends a classification window: a content-defined cut taken
/// by about one k-mer in 16 (and orientation-independent, as it looks at the
/// canonical form), so windows are typically 16 k-mers and the same sequence
/// is cut the same way wherever it occurs.
#[inline]
fn is_window_end<const K: usize>(kmer: &DirectedKmer<K>) -> bool
where
    Kmer<K>: KmerBits,
{
    let word = <Kmer<K> as KmerBits>::word(&kmer.canonical().bits, 0);
    (word ^ (word >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 == 0
}

/// Classification windows a task has already applied, by content.
///
/// Isoforms of a gene are usually adjacent in a transcriptome and share most
/// of their sequence. A window's updates are a function of its bases and of
/// its outer neighbors (or their absence), and applying the same update twice
/// changes nothing, so a task that meets a window it has already processed
/// can skip its MPHF lookups and state updates altogether. Windows are
/// recognized by content, never by digest alone: on a digest collision
/// between different windows the second is simply processed.
#[derive(Default)]
pub struct SeenWindows<'s> {
    index: HashMap<u64, (&'s [u8], u8), BuildNoHashHasher<u64>>,
}

impl<'s> SeenWindows<'s> {
    /// Record a window (its bases including the outer neighbors that exist,
    /// and which of them do) and report whether it is new.
    fn insert(&mut self, bases: &'s [u8], left_exists: bool, right_exists: bool) -> bool {
        let flags = left_exists as u8 | (right_exists as u8) << 1;
        let digest = xxhash_rust::xxh3::xxh3_64_with_seed(bases, flags as u64);
        match self.index.entry(digest) {
            Entry::Occupied(seen) => *seen.get() != (bases, flags),
            Entry::Vacant(slot) => {
                slot.insert((bases, flags));
                true
            }
        }
    }
}

/// Classifier: performs Phase 4 DFA classification.
/// Exact port of CdBG_Builder.cpp.
pub struct Classifier<'a, const K: usize>
where
    Kmer<K>: KmerBits,
{
    mphf: &'a Mphf<K>,
    states: &'a AtomicStateVector,
}

impl<'a, const K: usize> Classifier<'a, K>
where
    Kmer<K>: KmerBits,
{
    pub fn new(mphf: &'a Mphf<K>, states: &'a AtomicStateVector) -> Self {
        Classifier { mphf, states }
    }

    /// Process a substring of a sequence from left_end to right_end (k-mer indices).
    ///
    /// With `seen`, stretches of k-mers whose updates `seen` shows were already
    /// applied (see [`SeenWindows`]) are skipped.
    pub fn process_substring<'s>(
        &self,
        seq: &'s [u8],
        seq_len: usize,
        left_end: usize,
        right_end: usize,
        mut seen: Option<&mut SeenWindows<'s>>,
    ) {
        let mut kmer_idx = left_end;
        while kmer_idx <= right_end {
            kmer_idx = self.search_valid_kmer(seq, kmer_idx, right_end);
            if kmer_idx > right_end {
                break;
            }
            kmer_idx = self.process_contiguous_subseq(
                seq,
                seq_len,
                right_end,
                kmer_idx,
                seen.as_deref_mut(),
            );
        }
    }

    /// Find the next valid k-mer (no placeholders in the k-mer window).
    fn search_valid_kmer(&self, seq: &[u8], mut start: usize, right_end: usize) -> usize {
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

    /// Process the contiguous run of valid k-mers starting at `start_idx`: every
    /// k-mer up to the first one whose right neighbor is a placeholder (or the
    /// sequence end), or up to `right_end`, whichever comes first.
    /// Returns the non-inclusive ending index (last k-mer index + `K`).
    ///
    /// A k-mer's update depends only on its own state and its neighbors in the
    /// sequence, never on the update of the k-mer before it, so the run is handled
    /// in windows of at most [`LOOKUP_BATCH`] k-mers: every k-mer of a window is
    /// first rolled and resolved through the MPHF (independent lookups the core
    /// can overlap, with their state words prefetched), then the DFA updates are
    /// applied in sequence order. Interleaving each lookup with its dependent CAS
    /// loop instead leaves the core waiting on roughly one cache miss at a time.
    ///
    /// Windows end at content-defined k-mers ([`is_window_end`]), so a stretch of
    /// sequence shared by several references is cut into the same windows in
    /// each, which is what lets `seen` recognize a repeated window.
    fn process_contiguous_subseq<'s>(
        &self,
        seq: &'s [u8],
        seq_len: usize,
        right_end: usize,
        start_idx: usize,
        mut seen: Option<&mut SeenWindows<'s>>,
    ) -> usize {
        let mut last = start_idx;
        while last < right_end && !is_placeholder(seq[last + K]) {
            last += 1;
        }
        let has_left_first = start_idx > 0 && !is_placeholder(seq[start_idx - 1]);

        let first = DirectedKmer::<K>::new(Kmer::<K>::from_ascii(seq, start_idx));
        // kmers[i] is the k-mer at `pos + i`; kmers[n] its successor, if any.
        let mut kmers = [first; LOOKUP_BATCH + 1];
        let mut idxs = [0usize; LOOKUP_BATCH];
        let mut pos = start_idx;
        while pos <= last {
            let max_n = (last - pos + 1).min(LOOKUP_BATCH);
            let mut n = 1;
            while n < max_n && !is_window_end(&kmers[n - 1]) {
                kmers[n] = kmers[n - 1];
                kmers[n].roll_to_next_kmer(seq[pos + n - 1 + K]);
                n += 1;
            }
            // Inside the run every k-mer but `last` has a right neighbor; `last`
            // has one only when the run was cut at `right_end` (chunk boundary).
            let tail = pos + n - 1;
            let next_exists = tail < last || (tail + K < seq_len && !is_placeholder(seq[tail + K]));
            if next_exists {
                kmers[n] = kmers[n - 1];
                kmers[n].roll_to_next_kmer(seq[tail + K]);
            }
            let left_exists = pos > start_idx || has_left_first;

            // The window's updates are fully determined by its bases plus the
            // neighbor on each side, so if this task has already applied them,
            // applying them again would change nothing.
            let repeated = seen.as_deref_mut().is_some_and(|seen| {
                let lo = pos - left_exists as usize;
                let hi = tail + K + next_exists as usize;
                !seen.insert(&seq[lo..hi], left_exists, next_exists)
            });
            if !repeated {
                for (idx, kmer) in idxs[..n].iter_mut().zip(&kmers[..n]) {
                    *idx = self.mphf.hash(kmer.canonical()) as usize;
                    self.states.prefetch(*idx);
                }

                for i in 0..n {
                    let p = pos + i;
                    let idx = idxs[i];
                    let kmer = &kmers[i];
                    let has_left = i > 0 || left_exists;
                    let has_right = i + 1 < n || next_exists;
                    match (has_left, has_right) {
                        (false, false) => while !self.process_isolated_kmer(idx) {},
                        (true, false) => {
                            while !self.process_rightmost_kmer(idx, kmer, seq[p - 1]) {}
                        }
                        (false, true) => {
                            while !self.process_leftmost_kmer(idx, kmer, &kmers[i + 1], seq[p + K])
                            {
                            }
                        }
                        (true, true) => {
                            while !self.process_internal_kmer(
                                idx,
                                kmer,
                                &kmers[i + 1],
                                seq[p - 1],
                                seq[p + K],
                            ) {}
                        }
                    }
                }
            }

            // Read only when the loop continues, in which case it was rolled.
            kmers[0] = kmers[n];
            pos += n;
        }

        last + K
    }

    fn is_self_loop(&self, kmer_hat: &Kmer<K>, next_kmer_hat: &Kmer<K>) -> bool {
        kmer_hat == next_kmer_hat
    }

    fn process_isolated_kmer(&self, idx: usize) -> bool {
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        let new_state = State::from_vertex(&Vertex::mimo());
        self.states.compare_and_swap(idx, state, new_state)
    }

    fn process_loop(
        &self,
        idx: usize,
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        prev_char: u8, // 0 means leftmost (no prev)
    ) -> bool {
        // Leftmost (no prev_char) OR direct repeat -> MIMO.
        if prev_char == 0 || kmer.kmer() == next_kmer.kmer() {
            let state = self.states.get(idx);
            let new_state = State::from_vertex(&Vertex::mimo());
            return self.states.compare_and_swap(idx, state, new_state);
        }

        // Inverted repeat: treat as rightmost.
        self.process_rightmost_kmer(idx, kmer, prev_char)
    }

    fn process_leftmost_kmer(
        &self,
        idx: usize,
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        next_char: u8,
    ) -> bool {
        let kmer_hat = kmer.canonical();
        let dir = kmer.dir();
        let next_kmer_hat = next_kmer.canonical();
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        if self.is_self_loop(kmer_hat, next_kmer_hat) {
            return self.process_loop(idx, kmer, next_kmer, 0);
        }

        let old_state = state;
        let next_base = Base::map_base(next_char);

        let new_state = if dir == FWD {
            if !state.is_visited() {
                State::from_vertex(&Vertex::miso(next_base))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.back == next_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInSingleOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => {
                        if vertex.back != next_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state // no change
                        }
                    }
                    _ => {
                        // SIMO -> MIMO
                        State::from_vertex(&Vertex {
                            state_class: StateClass::MultiInMultiOut,
                            ..vertex
                        })
                    }
                }
            }
        } else {
            // BWD
            let compl_next = next_base.complement();
            if !state.is_visited() {
                State::from_vertex(&Vertex::simo(compl_next))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.front == compl_next {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::SingleInMultiOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => State::from_vertex(&Vertex {
                        state_class: StateClass::MultiInMultiOut,
                        ..vertex
                    }),
                    _ => {
                        // SIMO
                        if vertex.front != compl_next {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                }
            }
        };

        if new_state == old_state {
            true
        } else {
            self.states.compare_and_swap(idx, old_state, new_state)
        }
    }

    fn process_rightmost_kmer(&self, idx: usize, kmer: &DirectedKmer<K>, prev_char: u8) -> bool {
        let dir = kmer.dir();
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        let old_state = state;
        let prev_base = Base::map_base(prev_char);

        let new_state = if dir == FWD {
            if !state.is_visited() {
                State::from_vertex(&Vertex::simo(prev_base))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.front == prev_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::SingleInMultiOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => State::from_vertex(&Vertex {
                        state_class: StateClass::MultiInMultiOut,
                        ..vertex
                    }),
                    _ => {
                        // SIMO
                        if vertex.front != prev_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                }
            }
        } else {
            // BWD
            let compl_prev = prev_base.complement();
            if !state.is_visited() {
                State::from_vertex(&Vertex::miso(compl_prev))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.back == compl_prev {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInSingleOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => {
                        if vertex.back != compl_prev {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                    _ => {
                        // SIMO -> MIMO
                        State::from_vertex(&Vertex {
                            state_class: StateClass::MultiInMultiOut,
                            ..vertex
                        })
                    }
                }
            }
        };

        if new_state == old_state {
            true
        } else {
            self.states.compare_and_swap(idx, old_state, new_state)
        }
    }

    fn process_internal_kmer(
        &self,
        idx: usize,
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        prev_char: u8,
        next_char: u8,
    ) -> bool {
        let kmer_hat = kmer.canonical();
        let dir = kmer.dir();
        let next_kmer_hat = next_kmer.canonical();
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        if self.is_self_loop(kmer_hat, next_kmer_hat) {
            return self.process_loop(idx, kmer, next_kmer, prev_char);
        }

        let old_state = state;
        let prev_base = Base::map_base(prev_char);
        let next_base = Base::map_base(next_char);

        let new_state = if dir == FWD {
            if !state.is_visited() {
                State::from_vertex(&Vertex::siso(prev_base, next_base))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.front == prev_base && vertex.back == next_base {
                            return true; // no change needed
                        } else if vertex.front != prev_base && vertex.back != next_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else if vertex.front != prev_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInSingleOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::SingleInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => {
                        if vertex.back != next_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                    _ => {
                        // SIMO
                        if vertex.front != prev_base {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                }
            }
        } else {
            // BWD
            let compl_next = next_base.complement();
            let compl_prev = prev_base.complement();
            if !state.is_visited() {
                State::from_vertex(&Vertex::siso(compl_next, compl_prev))
            } else {
                let vertex = state.decode();
                match vertex.state_class {
                    StateClass::SingleInSingleOut => {
                        if vertex.front == compl_next && vertex.back == compl_prev {
                            return true;
                        } else if vertex.front != compl_next && vertex.back != compl_prev {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else if vertex.front != compl_next {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInSingleOut,
                                ..vertex
                            })
                        } else {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::SingleInMultiOut,
                                ..vertex
                            })
                        }
                    }
                    StateClass::MultiInSingleOut => {
                        if vertex.back != compl_prev {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                    _ => {
                        // SIMO
                        if vertex.front != compl_next {
                            State::from_vertex(&Vertex {
                                state_class: StateClass::MultiInMultiOut,
                                ..vertex
                            })
                        } else {
                            old_state
                        }
                    }
                }
            }
        };

        if new_state == old_state {
            true
        } else {
            self.states.compare_and_swap(idx, old_state, new_state)
        }
    }
}

/// Classify all vertices from input sequences.
/// Uses adaptive parallelism: small sequences are individual tasks,
/// large sequences are processed synchronously with borrowed data.
pub fn classify_vertices<const K: usize>(
    params: &crate::params::Params,
    mphf: &Mphf<K>,
    states: &AtomicStateVector,
    budget: &InFlightBudget,
) -> anyhow::Result<Vec<(String, usize)>>
where
    Kmer<K>: KmerBits,
{
    use tracing::info;

    info!("Phase 4: Classifying k-mer vertices");

    let classifier = Classifier::<K>::new(mphf, states);
    let short_seqs = Mutex::new(Vec::new());

    rayon::in_place_scope(|s| {
        let mut batch = SeqBatch::default();
        let classifier = &classifier;
        let classify_batch = move |batch: &SeqBatch| {
            let mut seen = SeenWindows::default();
            for seq in batch.iter() {
                classifier.process_substring(seq, seq.len(), 0, seq.len() - K, Some(&mut seen));
            }
        };
        for input_file in &params.input_files {
            let mut reader =
                needletail::parse_fastx_file(input_file).expect("failed to open FASTA file");

            while let Some(result) = reader.next() {
                let record = result.expect("invalid FASTA record");
                let seq = record.seq();
                let seq_len = seq.len();

                if seq_len < K {
                    if params.track_short_seqs {
                        let name = std::str::from_utf8(record.id())
                            .unwrap_or("")
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .to_string();
                        short_seqs.lock().unwrap().push((name, seq_len));
                    }
                    continue;
                }

                let right_end = seq_len - K;

                if seq_len < CHUNK_KMERS {
                    // Small sequences are batched for between-sequence parallelism,
                    // and so that shared stretches repeat within a task.
                    batch.push(&seq);
                    if batch.bases() >= BATCH_BASES {
                        std::mem::take(&mut batch).spawn(s, budget, classify_batch);
                    }
                } else {
                    // Large sequence: process synchronously with borrowed data (zero copy).
                    // Use fixed-size chunks for better allocator page reuse.
                    // process_substring uses the full seq buffer for boundary checks,
                    // so chunks get correct neighbor classification without overlap.
                    let seq_ref: &[u8] = &seq;
                    rayon::scope(|inner| {
                        let mut left = 0;
                        while left <= right_end {
                            let right = (left + CHUNK_KMERS - 1).min(right_end);

                            inner.spawn(move |_| {
                                classifier.process_substring(seq_ref, seq_len, left, right, None);
                            });
                            left += CHUNK_KMERS;
                        }
                    });
                }
            }
        }
        batch.spawn(s, budget, classify_batch);
    });

    info!("Classification complete");
    Ok(short_seqs.into_inner().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seen_windows_match_content_and_neighbors() {
        let a = b"ACGTACGTAC".to_vec();
        let b = b"ACGTACGTAC".to_vec();
        let c = b"ACGTACGTAG".to_vec();
        let mut seen = SeenWindows::default();
        assert!(seen.insert(&a, true, true));
        // Same bases and neighbors elsewhere: a repeat.
        assert!(!seen.insert(&b, true, true));
        // Same bases, different neighbor existence: a different window.
        assert!(seen.insert(&b, false, true));
        assert!(seen.insert(&b, true, false));
        // Different bases.
        assert!(seen.insert(&c, true, true));
        assert!(!seen.insert(&c, true, true));
    }
}
