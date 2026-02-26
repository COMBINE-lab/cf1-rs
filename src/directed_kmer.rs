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

/// Annotated k-mer: directed k-mer + position index + state class.
#[derive(Clone, Copy, Debug)]
pub struct AnnotatedKmer<const K: usize>
where
    Kmer<K>: KmerBits,
{
    pub directed: DirectedKmer<K>,
    pub idx: usize,
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
            state_class,
        }
    }

    pub fn roll_to_next_kmer(
        &mut self,
        next_char: u8,
        mphf: &crate::mphf::Mphf<K>,
        states: &crate::state_vector::AtomicStateVector,
    ) {
        self.directed.roll_to_next_kmer(next_char);
        self.idx += 1;
        let hash = mphf.hash(self.directed.canonical());
        let state = states.get(hash as usize);
        self.state_class = state.state_class();
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
