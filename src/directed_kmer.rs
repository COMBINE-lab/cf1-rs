use crate::dna::Base;
use crate::kmer::{Kmer, KmerBits};

/// Direction of a k-mer relative to its canonical form.
/// FWD = true, BWD = false (matches C++ convention).
pub type Direction = bool;
pub const FWD: Direction = true;
pub const BWD: Direction = false;

/// A k-mer bundled with its reverse complement, canonical form, and direction.
#[derive(Clone, Copy, Debug)]
pub struct DirectedKmer<const K: usize>
where
    Kmer<K>: KmerBits,
{
    kmer: Kmer<K>,
    rev_compl: Kmer<K>,
    canonical: Kmer<K>,
    dir: Direction,
}

impl<const K: usize> DirectedKmer<K>
where
    Kmer<K>: KmerBits,
{
    pub fn new(kmer: Kmer<K>) -> Self {
        let rev_compl = kmer.reverse_complement();
        let canonical = kmer.canonical_with_rc(&rev_compl);
        let dir = kmer.in_forward(&canonical);
        DirectedKmer {
            kmer,
            rev_compl,
            canonical,
            dir,
        }
    }

    pub fn roll_to_next_kmer(&mut self, next_char: u8) {
        let base = Base::map_base(next_char);
        self.kmer.roll_to_next_kmer(base, &mut self.rev_compl);
        self.canonical = self.kmer.canonical_with_rc(&self.rev_compl);
        self.dir = self.kmer.in_forward(&self.canonical);
    }

    #[inline]
    pub fn kmer(&self) -> &Kmer<K> {
        &self.kmer
    }

    #[inline]
    pub fn rev_compl(&self) -> &Kmer<K> {
        &self.rev_compl
    }

    #[inline]
    pub fn canonical(&self) -> &Kmer<K> {
        &self.canonical
    }

    #[inline]
    pub fn dir(&self) -> Direction {
        self.dir
    }
}

/// Annotated k-mer: directed k-mer + position index + MPHF index + state class.
#[derive(Clone, Copy, Debug)]
pub struct AnnotatedKmer<const K: usize>
where
    Kmer<K>: KmerBits,
{
    pub directed: DirectedKmer<K>,
    pub idx: usize,
    /// MPHF index of the canonical k-mer.
    pub hash: u64,
    pub state_class: crate::state::StateClass,
}

impl<const K: usize> AnnotatedKmer<K>
where
    Kmer<K>: KmerBits,
{
    pub fn new(
        kmer: Kmer<K>,
        kmer_idx: usize,
        mphf: &crate::mphf::Mphf<K>,
        states: &crate::state_vector::AtomicStateVector,
    ) -> Self {
        let directed = DirectedKmer::new(kmer);
        let hash = mphf.hash(directed.canonical());
        let state = states.get(hash as usize);
        let state_class = state.state_class();
        AnnotatedKmer {
            directed,
            idx: kmer_idx,
            hash,
            state_class,
        }
    }

    #[inline]
    pub fn kmer(&self) -> &Kmer<K> {
        self.directed.kmer()
    }

    #[inline]
    pub fn rev_compl(&self) -> &Kmer<K> {
        self.directed.rev_compl()
    }

    #[inline]
    pub fn canonical(&self) -> &Kmer<K> {
        self.directed.canonical()
    }

    #[inline]
    pub fn dir(&self) -> Direction {
        self.directed.dir()
    }
}

/// Number of k-mers [`AnnotatedRun`] resolves per refill.
const RUN_BATCH: usize = 32;

/// The annotated k-mers of a sequence, in order, from a start position through
/// its placeholder-free run.
///
/// Annotating a k-mer costs an MPHF lookup plus a state read — cache misses
/// that do not depend on one another. Resolving them a batch at a time (rolling
/// the k-mers first, then issuing all the lookups) lets the core overlap those
/// misses instead of stalling on each in turn as a k-mer-at-a-time roll would.
/// States must not change class while a run is being read; that holds in
/// unitig extraction, which only ever sets the outputted flag.
pub struct AnnotatedRun<'a, const K: usize>
where
    Kmer<K>: KmerBits,
{
    seq: &'a [u8],
    mphf: &'a crate::mphf::Mphf<K>,
    states: &'a crate::state_vector::AtomicStateVector,
    buf: [AnnotatedKmer<K>; RUN_BATCH],
    /// Next buffered k-mer to hand out; `buf[head..len]` are still pending.
    head: usize,
    len: usize,
}

impl<'a, const K: usize> AnnotatedRun<'a, K>
where
    Kmer<K>: KmerBits,
{
    /// Start a run at k-mer position `start` (which must be a valid k-mer).
    pub fn new(
        seq: &'a [u8],
        start: usize,
        mphf: &'a crate::mphf::Mphf<K>,
        states: &'a crate::state_vector::AtomicStateVector,
    ) -> Self {
        let first = AnnotatedKmer {
            directed: DirectedKmer::new(Kmer::<K>::from_ascii(seq, start)),
            idx: start,
            hash: 0,
            state_class: crate::state::StateClass::SingleInSingleOut,
        };
        let mut run = AnnotatedRun {
            seq,
            mphf,
            states,
            buf: [first; RUN_BATCH],
            head: 0,
            len: 1,
        };
        run.fill(1);
        run
    }

    /// The next k-mer of the run. The caller must only ask for a successor
    /// that exists, i.e. when the base following the last returned k-mer is
    /// in the sequence and not a placeholder.
    #[inline]
    pub fn next_kmer(&mut self) -> AnnotatedKmer<K> {
        if self.head == self.len {
            let last = self.buf[self.len - 1];
            self.buf[0] = last;
            self.buf[0]
                .directed
                .roll_to_next_kmer(self.seq[last.idx + K]);
            self.buf[0].idx += 1;
            self.len = 1;
            self.head = 0;
            self.fill(1);
        }
        let kmer = self.buf[self.head];
        self.head += 1;
        kmer
    }

    /// Roll `buf[from..]` forward from `buf[from - 1]` for as long as the run
    /// continues, then resolve the MPHF index and state class of every k-mer
    /// from `buf[from - 1]` on.
    fn fill(&mut self, from: usize) {
        let resolve_from = from - 1;
        while self.len < RUN_BATCH {
            let prev = self.buf[self.len - 1];
            let next_pos = prev.idx + K;
            if next_pos >= self.seq.len() || crate::dna::is_placeholder(self.seq[next_pos]) {
                break;
            }
            let mut kmer = prev;
            kmer.directed.roll_to_next_kmer(self.seq[next_pos]);
            kmer.idx += 1;
            self.buf[self.len] = kmer;
            self.len += 1;
        }
        for kmer in &mut self.buf[resolve_from..self.len] {
            kmer.hash = self.mphf.hash(kmer.directed.canonical());
            self.states.prefetch(kmer.hash as usize);
        }
        for kmer in &mut self.buf[resolve_from..self.len] {
            kmer.state_class = self.states.get(kmer.hash as usize).state_class();
        }
    }
}
