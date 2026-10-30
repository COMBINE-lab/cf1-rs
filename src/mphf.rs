use crate::dna::Base;
use crate::kmer::{Kmer, KmerBits};
use crate::minimizer::Partitioning;
use crate::superkmer::{bin_file_path, packed_superkmers};
use dsi_progress_logger::no_logging;
use lender::{FallibleLend, FallibleLender, FallibleLending};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use sux::func::{PHastR, PHastRBuilder};
use sux::utils::ToSig;
use tracing::info;

/// The MPHF: sux's PHastR with its default parameters (8-bit seeds, four
/// layouts, slices of length 1024, expected bucket size 4.25; about 1.94
/// bits per key).
///
/// Phases 4 and 5 evaluate the MPHF once per k-mer *occurrence* in the input
/// (on a transcriptome, several times the number of distinct k-mers), so
/// evaluation speed bounds their running time: PHastR answers almost every
/// query with a signature and a single seed read. It can be built either from
/// a slice of keys in memory or from keys streamed once, keeping 16-byte
/// per-key construction records in temporary files (offline mode), and both
/// constructions build the same function.
type Phf<const K: usize> = PHastR<KmerKey<K>>;

/// Number of chunks each PHastR level is swept in.
///
/// The function PHastR builds depends on the number of sweep chunks, which by
/// default is the size of the current rayon pool: fixing it makes the MPHF —
/// and hence unitig ids — independent of the thread count and of the
/// construction regime. A sweep uses at most this many threads.
const SWEEP_CHUNKS: usize = 16;

/// The PHastR builder used for every construction (see [`SWEEP_CHUNKS`]).
fn phf_builder() -> PHastRBuilder {
    PHastRBuilder::default().sweep_chunks(SWEEP_CHUNKS)
}

/// Global minimal perfect hash function over all canonical k-mers.
pub struct Mphf<const K: usize>
where
    Kmer<K>: KmerBits,
{
    mphf: Phf<K>,
    total_kmers: u64,
}

impl<const K: usize> Mphf<K>
where
    Kmer<K>: KmerBits,
{
    /// Build a global MPHF over the distinct canonical k-mers of all bins.
    ///
    /// Each bin is deduplicated independently, in parallel waves of bins
    /// (canonical minimizers guarantee no cross-bin duplicates); see
    /// [`build_with_clone_threshold`](Self::build_with_clone_threshold) for
    /// how the keys reach the MPHF construction.
    pub fn build(
        partitioning: &Partitioning,
        work_dir: &Path,
        _memory_budget_bytes: usize,
    ) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        Self::build_with_clone_threshold(partitioning, work_dir, DEFAULT_CLONE_THRESHOLD)
    }

    /// Build MPHF with an explicit `clone_threshold`.
    ///
    /// ## Construction strategy
    ///
    /// Bins are deduplicated a wave of bins at a time. Two regimes, chosen by
    /// the running total of distinct k-mers vs `clone_threshold`:
    ///
    /// **In-memory** (`total <= clone_threshold`): each wave's k-mers are
    /// appended to one key vector, and PHastR is built over that slice,
    /// computing key signatures in parallel. Peak RSS is the keys (8 or 16
    /// bytes each) plus about 10–12 bytes per key of construction memory.
    ///
    /// **Streaming** (`total > clone_threshold`): as soon as the threshold
    /// would be exceeded, the keys gathered so far, the wave that overflowed
    /// and then every later bin are streamed once into an offline PHastR
    /// construction, which keeps its 16-byte per-key records in temporary
    /// files in `work_dir` (about 16 bytes per key of disk) and needs well
    /// under a byte per key of memory besides ~256 MiB of buffers. Streaming
    /// is sequential, so later bins are deduplicated in the background, up to
    /// one bin per thread ahead. The keys themselves never touch the disk.
    ///
    /// Both regimes build the same function (see [`SWEEP_CHUNKS`]).
    pub fn build_with_clone_threshold(
        partitioning: &Partitioning,
        work_dir: &Path,
        clone_threshold: usize,
    ) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        Self::build_from_bins(partitioning.num_bins(), work_dir, clone_threshold)
    }

    /// [`build_with_clone_threshold`](Self::build_with_clone_threshold) on
    /// the bin files `0..num_bins` of `work_dir`.
    fn build_from_bins(
        num_bins: usize,
        work_dir: &Path,
        clone_threshold: usize,
    ) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        info!("Reading canonical k-mers from {} bins", num_bins);
        info!(
            "P3a: RSS before dedup: current={} MB, peak={} MB",
            crate::pipeline::current_rss_mb(),
            crate::pipeline::peak_rss_mb()
        );

        // Reserves address space only: pages are committed as keys are written.
        let mut keys: Vec<KmerKey<K>> = Vec::with_capacity(clone_threshold);
        let mut per_bin_counts: Vec<usize> = Vec::with_capacity(num_bins);
        let bins: Vec<usize> = (0..num_bins).collect();
        let mut waves = bins.chunks(2 * rayon::current_num_threads());
        let mut overflow = None;
        for wave in waves.by_ref() {
            let deduped = dedup_wave::<K>(work_dir, wave);
            per_bin_counts.extend(deduped.iter().map(Vec::len));
            if keys.len() + deduped.iter().map(Vec::len).sum::<usize>() > clone_threshold {
                overflow = Some(deduped);
                break;
            }
            append_par(&mut keys, &deduped);
        }

        let Some(first_wave) = overflow else {
            log_dedup_totals(&per_bin_counts);
            let total = keys.len();
            info!(
                "Building global MPHF for {} distinct canonical k-mers (in-memory, threads={})",
                total,
                rayon::current_num_threads()
            );
            let mphf = Phf::<K>::try_par_new_with_builder(&keys, phf_builder(), no_logging![])?;
            info!("MPHF construction complete");
            return Ok(Mphf {
                mphf,
                total_kmers: total as u64,
            });
        };

        // Streaming regime: stream the kept keys, the wave that overflowed
        // and every later bin into an offline construction. Later bins are
        // deduplicated in the background, a window of bins ahead of the one
        // being streamed, as streaming is sequential.
        info!(
            "Building global MPHF by streaming distinct canonical k-mers (offline, clone_threshold={}, threads={})",
            clone_threshold,
            rayon::current_num_threads()
        );
        let window = rayon::current_num_threads();
        let mut first_wave = first_wave.into_iter();
        let mut later =
            BinPrefetcher::<K>::new(work_dir, waves.flatten().copied().collect(), window);
        let lender = StreamingKeys::<K, _> {
            kept: keys,
            kept_pos: 0,
            bin: Vec::new(),
            pos: 0,
            next_bin: || {
                if let Some(bin) = first_wave.next() {
                    // Keep at most `window` deduplicated bins in memory.
                    later.prefetch(window.saturating_sub(first_wave.len()));
                    return Some(bin);
                }
                let bin = later.next()?;
                per_bin_counts.push(bin.len());
                Some(bin)
            },
            cur: KmerKey(Kmer::default()),
        };
        let builder = phf_builder().offline(true).temp_dir(work_dir);
        let mphf = Phf::<K>::try_new_with_builder(lender, builder, no_logging![])?;
        log_dedup_totals(&per_bin_counts);
        info!("MPHF construction complete");
        let total = mphf.len();
        Ok(Mphf {
            mphf,
            total_kmers: total as u64,
        })
    }

    /// Build MPHF from a pre-existing consolidated dedup file.
    /// Skips dedup and consolidation — just builds the MPHF.
    pub fn build_from_consolidated(
        consolidated_path: &Path,
        total: usize,
        clone_threshold: usize,
    ) -> anyhow::Result<Self> {
        let storage_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();
        let mphf = if total <= clone_threshold {
            info!(
                "Building global MPHF for {} distinct canonical k-mers (in-memory, threads={})",
                total,
                rayon::current_num_threads()
            );
            let keys = read_all_kmers::<K>(consolidated_path, storage_size);
            Phf::<K>::try_par_new_with_builder(&keys, phf_builder(), no_logging![])?
        } else {
            info!(
                "Building global MPHF for {} distinct canonical k-mers (streaming, offline, threads={})",
                total,
                rayon::current_num_threads()
            );
            let dir = consolidated_path.parent().unwrap_or(Path::new("."));
            let builder = phf_builder().offline(true).temp_dir(dir);
            let lender = IterLender {
                iter: KmerFileIterator::<K>::new(consolidated_path, storage_size),
                cur: None,
            };
            Phf::<K>::try_new_with_builder(lender, builder, no_logging![])?
        };
        info!("MPHF construction complete");
        Ok(Mphf {
            mphf,
            total_kmers: total as u64,
        })
    }

    /// Look up the global hash index for a canonical k-mer.
    #[inline(always)]
    pub fn hash(&self, canonical_kmer: &Kmer<K>) -> u64 {
        self.mphf.get(KmerKey(*canonical_kmer)) as u64
    }

    /// Get the bucket_id for a canonical k-mer (used for unitig IDs).
    #[inline]
    pub fn bucket_id(&self, canonical_kmer: &Kmer<K>) -> u64 {
        self.hash(canonical_kmer)
    }

    pub fn total_kmers(&self) -> u64 {
        self.total_kmers
    }
}

/// Default [`Mphf::build_with_clone_threshold`] threshold: up to this many
/// distinct k-mers the MPHF is built in memory.
pub const DEFAULT_CLONE_THRESHOLD: usize = 300_000_000;

/// Deduplicate the bins of `wave` in parallel, returning each bin's sorted
/// distinct canonical k-mers.
fn dedup_wave<const K: usize>(
    work_dir: &Path,
    wave: &[usize],
) -> Vec<Vec<<Kmer<K> as KmerBits>::Storage>>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    wave.par_iter()
        .map(|&bin| dedup_bin_kmers::<K>(&bin_file_path(work_dir, bin)))
        .collect()
}

fn log_dedup_totals(per_bin_counts: &[usize]) {
    let num_bins = per_bin_counts.len();
    let total: usize = per_bin_counts.iter().sum();
    info!(
        "Total {} distinct canonical k-mers across {} bins (per-bin: min={}, max={}, avg={})",
        total,
        num_bins,
        per_bin_counts.iter().min().unwrap_or(&0),
        per_bin_counts.iter().max().unwrap_or(&0),
        total / num_bins.max(1)
    );
    info!(
        "P3a: RSS after dedup: current={} MB, peak={} MB",
        crate::pipeline::current_rss_mb(),
        crate::pipeline::peak_rss_mb()
    );
}

/// Lends the keys of the streaming regime to PHastR: first the keys kept in
/// memory (freed once lent), then those of each bin returned by `next_bin`.
struct StreamingKeys<const K: usize, F>
where
    Kmer<K>: KmerBits,
{
    kept: Vec<KmerKey<K>>,
    kept_pos: usize,
    bin: Vec<<Kmer<K> as KmerBits>::Storage>,
    pos: usize,
    next_bin: F,
    cur: KmerKey<K>,
}

impl<'lend, const K: usize, F> FallibleLending<'lend> for StreamingKeys<K, F>
where
    Kmer<K>: KmerBits,
{
    type Lend = &'lend KmerKey<K>;
}

impl<const K: usize, F> FallibleLender for StreamingKeys<K, F>
where
    Kmer<K>: KmerBits,
    F: FnMut() -> Option<Vec<<Kmer<K> as KmerBits>::Storage>>,
{
    lender::check_covariance_fallible!();
    type Error = std::convert::Infallible;

    #[inline]
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        if let Some(&key) = self.kept.get(self.kept_pos) {
            self.kept_pos += 1;
            self.cur = key;
            return Ok(Some(&self.cur));
        }
        if self.kept.capacity() != 0 {
            self.kept = Vec::new();
        }
        loop {
            if let Some(&bits) = self.bin.get(self.pos) {
                self.pos += 1;
                self.cur = KmerKey(Kmer { bits });
                return Ok(Some(&self.cur));
            }
            // Free the exhausted bin before deduplicating the next one.
            self.bin = Vec::new();
            match (self.next_bin)() {
                Some(bin) => {
                    self.bin = bin;
                    self.pos = 0;
                }
                None => return Ok(None),
            }
        }
    }
}

/// Deduplicates a sequence of bins in the background on the current rayon
/// pool, keeping up to `window` bins in flight ahead of the one returned
/// last, and returns them in order. Nothing is started before the first call
/// to [`next`](Self::next) or [`prefetch`](Self::prefetch).
struct BinPrefetcher<const K: usize>
where
    Kmer<K>: KmerBits,
{
    work_dir: PathBuf,
    bins: Vec<usize>,
    results: Vec<crossbeam_channel::Receiver<Vec<<Kmer<K> as KmerBits>::Storage>>>,
    window: usize,
    next: usize,
}

impl<const K: usize> BinPrefetcher<K>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    fn new(work_dir: &Path, bins: Vec<usize>, window: usize) -> Self {
        BinPrefetcher {
            work_dir: work_dir.to_path_buf(),
            bins,
            results: Vec::new(),
            window: window.max(1),
            next: 0,
        }
    }

    /// Starts deduplicating the next `ahead` bins, if not started yet.
    fn prefetch(&mut self, ahead: usize) {
        self.spawn_up_to(self.next + ahead);
    }

    /// Starts deduplicating the bins before position `end` not started yet.
    fn spawn_up_to(&mut self, end: usize) {
        while self.results.len() < end.min(self.bins.len()) {
            let path = bin_file_path(&self.work_dir, self.bins[self.results.len()]);
            let (tx, rx) = crossbeam_channel::bounded(1);
            rayon::spawn(move || {
                // The receiver is gone only if construction was abandoned.
                let _ = tx.send(dedup_bin_kmers::<K>(&path));
            });
            self.results.push(rx);
        }
    }

    /// Returns the next bin's distinct canonical k-mers.
    fn next(&mut self) -> Option<Vec<<Kmer<K> as KmerBits>::Storage>> {
        if self.next == self.bins.len() {
            return None;
        }
        self.spawn_up_to(self.next + 1 + self.window);
        let rx = &self.results[self.next];
        let bin = loop {
            match rx.try_recv() {
                Ok(bin) => break bin,
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // Help the pool (with one thread, nothing else would
                    // deduplicate the bin), or wait for it.
                    if rayon::yield_now() != Some(rayon::Yield::Executed) {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    panic!("bin deduplication failed")
                }
            }
        };
        self.next += 1;
        Some(bin)
    }
}

/// Lends the keys of an iterator to PHastR.
struct IterLender<I: Iterator> {
    iter: I,
    cur: Option<I::Item>,
}

impl<'lend, I: Iterator> FallibleLending<'lend> for IterLender<I> {
    type Lend = &'lend I::Item;
}

impl<I: Iterator> FallibleLender for IterLender<I> {
    lender::check_covariance_fallible!();
    type Error = std::convert::Infallible;

    #[inline]
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        self.cur = self.iter.next();
        Ok(self.cur.as_ref())
    }
}

/// Bridge trait for sorting and deduplicating Storage types (u64, u128).
pub trait RadixSortDedup {
    fn radix_sort_dedup(vec: &mut Vec<Self>)
    where
        Self: Sized;
}

impl RadixSortDedup for u64 {
    fn radix_sort_dedup(vec: &mut Vec<Self>) {
        // Call the free function directly — it's the truly in-place radix sort.
        // The VoraciousSort trait impl for u64 dispatches to dlsd_radixsort
        // which allocates an O(n) auxiliary buffer.
        voracious_radix_sort::voracious_sort(vec, 8);
        vec.dedup();
    }
}

impl RadixSortDedup for u128 {
    fn radix_sort_dedup(vec: &mut Vec<Self>) {
        voracious_radix_sort::voracious_sort(vec, 8);
        vec.dedup();
    }
}

/// Read a bin's packed super k-mers, extract and deduplicate canonical k-mers
/// using radix sort, and write the unique canonical k-mers to a binary file.
/// Returns the count of distinct canonical k-mers.
pub fn dedup_bin_public<const K: usize>(src: &Path, dst: &Path) -> usize
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    dedup_bin::<K>(src, dst)
}

fn dedup_bin<const K: usize>(src: &Path, dst: &Path) -> usize
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    let kmers = dedup_bin_kmers::<K>(src);
    write_kmers::<K>(dst, kmers.iter().copied());
    kmers.len()
}

/// Append the k-mers of `parts` to `keys` in order, copying the parts in
/// parallel. `keys` must already have the capacity.
fn append_par<const K: usize>(
    keys: &mut Vec<KmerKey<K>>,
    parts: &[Vec<<Kmer<K> as KmerBits>::Storage>],
) where
    Kmer<K>: KmerBits,
{
    let added: usize = parts.iter().map(Vec::len).sum();
    assert!(keys.capacity() - keys.len() >= added, "key vector capacity");
    let mut spare = &mut keys.spare_capacity_mut()[..added];
    let mut slots = Vec::with_capacity(parts.len());
    for part in parts {
        let (slot, rest) = spare.split_at_mut(part.len());
        slots.push(slot);
        spare = rest;
    }
    slots.into_par_iter().zip(parts).for_each(|(slot, part)| {
        for (dst, &bits) in slot.iter_mut().zip(part) {
            dst.write(KmerKey(Kmer { bits }));
        }
    });
    // SAFETY: the `added` slots after the current length were all written above.
    unsafe { keys.set_len(keys.len() + added) };
}

/// Read a bin's packed super k-mers and return its distinct canonical k-mers,
/// sorted.
fn dedup_bin_kmers<const K: usize>(src: &Path) -> Vec<<Kmer<K> as KmerBits>::Storage>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    // A missing bin file is an empty bin.
    let data = std::fs::read(src).unwrap_or_default();

    // Collect all canonical k-mer storage values, sized up front so the
    // (occurrence-sized) buffer is never regrown.
    let num_kmers: usize = packed_superkmers(&data)
        .map(|(base_count, _)| (base_count + 1).saturating_sub(K))
        .sum();
    let mut kmer_vals: Vec<<Kmer<K> as KmerBits>::Storage> = Vec::with_capacity(num_kmers);
    for (base_count, packed) in packed_superkmers(&data) {
        if base_count >= K {
            // Roll through the super k-mer rather than decoding (and reverse-
            // complementing) every k-mer from scratch.
            let mut kmer = Kmer::<K>::from_packed_2bit(packed, 0);
            let mut rc = kmer.reverse_complement();
            kmer_vals.push(kmer.canonical_with_rc(&rc).bits);
            for j in K..base_count {
                let code = (packed[j / 4] >> (6 - 2 * (j % 4))) & 0b11;
                kmer.roll_to_next_kmer(Base::from_2bit(code), &mut rc);
                kmer_vals.push(kmer.canonical_with_rc(&rc).bits);
            }
        }
    }
    drop(data);

    // Radix sort and deduplicate. Distinct k-mers are typically a small
    // fraction of the occurrences, and the result may be held until every bin
    // is done, so give the excess capacity back.
    RadixSortDedup::radix_sort_dedup(&mut kmer_vals);
    kmer_vals.shrink_to_fit();
    kmer_vals
}

/// Write k-mers to `dst` as raw binary (native endian Storage bytes).
fn write_kmers<const K: usize>(
    dst: &Path,
    kmers: impl IntoIterator<Item = <Kmer<K> as KmerBits>::Storage>,
) where
    Kmer<K>: KmerBits,
{
    let storage_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();
    let file = std::fs::File::create(dst).expect("Failed to create dedup file");
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    for val in kmers {
        let bytes = <Kmer<K> as KmerBits>::as_bytes(&val);
        writer
            .write_all(&bytes[..storage_size])
            .expect("Failed to write k-mer");
    }
    writer.flush().expect("Failed to flush dedup file");
}

/// Iterator that reads KmerKey<K> values from a consolidated binary file via BufReader.
struct KmerFileIterator<const K: usize>
where
    Kmer<K>: KmerBits,
{
    reader: std::io::BufReader<std::fs::File>,
    storage_size: usize,
}

impl<const K: usize> KmerFileIterator<K>
where
    Kmer<K>: KmerBits,
{
    fn new(path: &Path, storage_size: usize) -> Self {
        let file = std::fs::File::open(path).expect("Failed to open consolidated dedup file");
        KmerFileIterator {
            reader: std::io::BufReader::with_capacity(256 * 1024, file),
            storage_size,
        }
    }
}

impl<const K: usize> Iterator for KmerFileIterator<K>
where
    Kmer<K>: KmerBits,
{
    type Item = KmerKey<K>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use std::io::Read;
        let mut kmer = Kmer::<K>::default();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut kmer.bits as *mut <Kmer<K> as KmerBits>::Storage as *mut u8,
                self.storage_size,
            )
        };
        match self.reader.read_exact(bytes) {
            Ok(()) => Some(KmerKey(kmer)),
            Err(_) => None,
        }
    }
}

/// Read all k-mers from a consolidated binary file into a Vec.
/// Used for small inputs where the entire keyset fits comfortably in RAM.
fn read_all_kmers<const K: usize>(path: &Path, storage_size: usize) -> Vec<KmerKey<K>>
where
    Kmer<K>: KmerBits,
{
    let file_len = std::fs::metadata(path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let mut keys = Vec::with_capacity(file_len / storage_size);
    keys.extend(KmerFileIterator::<K>::new(path, storage_size));
    keys
}

/// A canonical k-mer as an MPHF key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct KmerKey<const K: usize>(pub(crate) Kmer<K>)
where
    Kmer<K>: KmerBits;

impl<const K: usize> ToSig<[u64; 1]> for KmerKey<K>
where
    Kmer<K>: KmerBits,
{
    /// Seeded rapidhash v3 (a stable hash) of the k-mer's native-endian
    /// storage bytes, whose length is a constant of the k-mer type.
    #[inline(always)]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        use rapidhash::v3::{RapidSecrets, rapidhash_v3_inline};
        // Queries hash with the function's seed, zero: skip deriving its
        // secrets at every query.
        const SEED0: RapidSecrets = RapidSecrets::seed_cpp(0);
        let secrets = if seed == 0 {
            SEED0
        } else {
            RapidSecrets::seed_cpp(seed)
        };
        let bits = &key.borrow().0.bits;
        [rapidhash_v3_inline::<true, false, false>(
            <Kmer<K> as KmerBits>::as_bytes(bits),
            &secrets,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A tiny deterministic generator (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Writes `num_bins` bin files of random super k-mers to a fresh
    /// directory. Super k-mers repeat within a bin, so that deduplication
    /// matters, but not across bins (as with minimizer partitioning).
    fn write_bins(dir: &Path, num_bins: usize, seed: u64) {
        let mut rng = Rng(seed);
        for bin in 0..num_bins {
            let mut pool: Vec<Vec<u8>> = vec![];
            let mut data = vec![];
            for _ in 0..200 {
                let bases: Vec<u8> = if !pool.is_empty() && rng.next().is_multiple_of(4) {
                    pool[(rng.next() % pool.len() as u64) as usize].clone()
                } else {
                    let len = 20 + (rng.next() % 120) as usize;
                    (0..len).map(|_| (rng.next() & 3) as u8).collect()
                };
                data.extend_from_slice(&(bases.len() as u16).to_le_bytes());
                for chunk in bases.chunks(4) {
                    let mut byte = 0u8;
                    for (i, &b) in chunk.iter().enumerate() {
                        byte |= b << (6 - 2 * i);
                    }
                    data.push(byte);
                }
                pool.push(bases);
            }
            std::fs::write(bin_file_path(dir, bin), data).unwrap();
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cf1-mphf-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Every construction regime and thread count builds the same minimal
    /// perfect hash function on the distinct canonical k-mers of the bins.
    fn check_regimes<const K: usize>(name: &str)
    where
        Kmer<K>: KmerBits,
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        let num_bins = 13;
        let dir = scratch_dir(name);
        write_bins(&dir, num_bins, 0x9E37_79B9_7F4A_7C15 ^ K as u64);
        let keys: Vec<Kmer<K>> = (0..num_bins)
            .flat_map(|bin| dedup_bin_kmers::<K>(&bin_file_path(&dir, bin)))
            .map(|bits| Kmer { bits })
            .collect();
        let n = keys.len();
        assert!(n > 10_000, "too few keys ({n})");

        let mut reference: Option<Vec<u64>> = None;
        // In memory; streaming from the start; streaming after keeping some
        // waves in memory.
        for threshold in [n, 0, n / 3] {
            for threads in [1, 2, 5] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mphf = pool
                    .install(|| Mphf::<K>::build_from_bins(num_bins, &dir, threshold))
                    .unwrap();
                assert_eq!(mphf.total_kmers() as usize, n);
                let values: Vec<u64> = keys.iter().map(|k| mphf.hash(k)).collect();
                let distinct: HashSet<u64> = values.iter().copied().collect();
                assert_eq!(distinct.len(), n, "not injective");
                assert!(values.iter().all(|&v| (v as usize) < n), "not minimal");
                match &reference {
                    None => reference = Some(values),
                    Some(r) => assert!(
                        r == &values,
                        "threshold {threshold}, {threads} threads: different function"
                    ),
                }
            }
        }
        // Offline construction leaves no files behind besides the bins.
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, num_bins);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn regimes_build_the_same_function_u64() {
        check_regimes::<31>("k31");
    }

    #[test]
    fn regimes_build_the_same_function_u128() {
        check_regimes::<63>("k63");
    }
}
