use crate::budget::InFlightBudget;
use crate::directed_kmer::DirectedKmer;
use crate::dna::{is_placeholder, Base};
use crate::kmer::{Kmer, KmerBits};
use crate::mphf::Mphf;
use crate::state::{State, StateClass, Vertex};
use crate::state_vector::AtomicStateVector;
use crate::directed_kmer::FWD;
use std::sync::Mutex;

/// Target chunk size (k-mer positions) for splitting large sequences.
/// Keeps per-thread working set small for better allocator page reuse.
const CHUNK_KMERS: usize = 4_000_000;

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
    pub fn process_substring(&self, seq: &[u8], seq_len: usize, left_end: usize, right_end: usize) {
        let mut kmer_idx = left_end;
        while kmer_idx <= right_end {
            kmer_idx = self.search_valid_kmer(seq, kmer_idx, right_end);
            if kmer_idx > right_end {
                break;
            }
            kmer_idx = self.process_contiguous_subseq(seq, seq_len, right_end, kmer_idx);
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

    /// Process a contiguous subsequence starting at start_idx.
    /// Returns the non-inclusive ending index.
    fn process_contiguous_subseq(
        &self,
        seq: &[u8],
        seq_len: usize,
        right_end: usize,
        start_idx: usize,
    ) -> usize {
        let mut kmer_idx = start_idx;
        let mut curr_kmer = DirectedKmer::<K>::new(Kmer::<K>::from_ascii(seq, kmer_idx));

        // Check if isolated k-mer.
        let no_left = kmer_idx == 0 || is_placeholder(seq[kmer_idx - 1]);
        let no_right = kmer_idx + K == seq_len || is_placeholder(seq[kmer_idx + K]);

        if no_left && no_right {
            while !self.process_isolated_kmer(&curr_kmer) {}
        } else {
            // No valid right neighbor.
            if no_right {
                while !self.process_rightmost_kmer(&curr_kmer, seq[kmer_idx - 1]) {}
                return kmer_idx + K;
            }

            // A valid right neighbor exists.
            let mut next_kmer = curr_kmer;
            next_kmer.roll_to_next_kmer(seq[kmer_idx + K]);

            // No valid left neighbor.
            if no_left {
                while !self.process_leftmost_kmer(&curr_kmer, &next_kmer, seq[kmer_idx + K]) {}
            } else {
                while !self.process_internal_kmer(
                    &curr_kmer,
                    &next_kmer,
                    seq[kmer_idx - 1],
                    seq[kmer_idx + K],
                ) {}
            }

            // Process internal k-mers.
            kmer_idx += 1;
            while kmer_idx < right_end && !is_placeholder(seq[kmer_idx + K]) {
                curr_kmer = next_kmer;
                next_kmer.roll_to_next_kmer(seq[kmer_idx + K]);

                while !self.process_internal_kmer(
                    &curr_kmer,
                    &next_kmer,
                    seq[kmer_idx - 1],
                    seq[kmer_idx + K],
                ) {}

                kmer_idx += 1;
            }

            // Process rightmost k-mer.
            if kmer_idx <= right_end {
                curr_kmer = next_kmer;

                if kmer_idx + K == seq_len || is_placeholder(seq[kmer_idx + K]) {
                    while !self.process_rightmost_kmer(&curr_kmer, seq[kmer_idx - 1]) {}
                } else {
                    next_kmer.roll_to_next_kmer(seq[kmer_idx + K]);
                    while !self.process_internal_kmer(
                        &curr_kmer,
                        &next_kmer,
                        seq[kmer_idx - 1],
                        seq[kmer_idx + K],
                    ) {}
                }
            } else {
                kmer_idx -= 1;
            }
        }

        kmer_idx + K
    }

    fn is_self_loop(&self, kmer_hat: &Kmer<K>, next_kmer_hat: &Kmer<K>) -> bool {
        kmer_hat == next_kmer_hat
    }

    fn process_isolated_kmer(&self, kmer: &DirectedKmer<K>) -> bool {
        let kmer_hat = kmer.canonical();
        let idx = self.mphf.hash(kmer_hat) as usize;
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        let new_state = State::from_vertex(&Vertex::mimo());
        self.states.compare_and_swap(idx, state, new_state)
    }

    fn process_loop(
        &self,
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        prev_char: u8, // 0 means leftmost (no prev)
    ) -> bool {
        // Leftmost (no prev_char) OR direct repeat -> MIMO.
        if prev_char == 0 || kmer.kmer() == next_kmer.kmer() {
            let kmer_hat = kmer.canonical();
            let idx = self.mphf.hash(kmer_hat) as usize;
            let state = self.states.get(idx);
            let new_state = State::from_vertex(&Vertex::mimo());
            return self.states.compare_and_swap(idx, state, new_state);
        }

        // Inverted repeat: treat as rightmost.
        self.process_rightmost_kmer(kmer, prev_char)
    }

    fn process_leftmost_kmer(
        &self,
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        next_char: u8,
    ) -> bool {
        let kmer_hat = kmer.canonical();
        let dir = kmer.dir();
        let next_kmer_hat = next_kmer.canonical();
        let idx = self.mphf.hash(kmer_hat) as usize;
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        if self.is_self_loop(kmer_hat, next_kmer_hat) {
            return self.process_loop(kmer, next_kmer, 0);
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

    fn process_rightmost_kmer(&self, kmer: &DirectedKmer<K>, prev_char: u8) -> bool {
        let kmer_hat = kmer.canonical();
        let dir = kmer.dir();
        let idx = self.mphf.hash(kmer_hat) as usize;
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
        kmer: &DirectedKmer<K>,
        next_kmer: &DirectedKmer<K>,
        prev_char: u8,
        next_char: u8,
    ) -> bool {
        let kmer_hat = kmer.canonical();
        let dir = kmer.dir();
        let next_kmer_hat = next_kmer.canonical();
        let idx = self.mphf.hash(kmer_hat) as usize;
        let state = self.states.get(idx);

        if state.is_dead_end() {
            return true;
        }

        if self.is_self_loop(kmer_hat, next_kmer_hat) {
            return self.process_loop(kmer, next_kmer, prev_char);
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
        for input_file in &params.input_files {
            let mut reader = needletail::parse_fastx_file(input_file)
                .expect("failed to open FASTA file");

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
                    // Small sequence: copy and spawn async for between-sequence parallelism.
                    budget.wait_and_acquire(seq_len);
                    let seq_owned = seq.to_vec();
                    let classifier = &classifier;
                    let budget = &budget;

                    s.spawn(move |_| {
                        classifier.process_substring(&seq_owned, seq_len, 0, right_end);
                        budget.release(seq_len);
                    });
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
                            let classifier = &classifier;

                            inner.spawn(move |_| {
                                classifier.process_substring(seq_ref, seq_len, left, right);
                            });
                            left += CHUNK_KMERS;
                        }
                    });
                }
            }
        }
    });

    info!("Classification complete");
    Ok(short_seqs.into_inner().unwrap())
}
