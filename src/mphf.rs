use crate::kmer::{Kmer, KmerBits};
use crate::minimizer::Partitioning;
use crate::superkmer::{bin_file_path, BinReader};
use ph::fmph::keyset::CachedKeySet;
use rayon::prelude::*;
use voracious_radix_sort::RadixSort;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// Global minimal perfect hash function over all canonical k-mers.
pub struct Mphf<const K: usize>
where
    Kmer<K>: KmerBits,
{
    mphf: ph::fmph::Function,
    total_kmers: u64,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> Mphf<K>
where
    Kmer<K>: KmerBits,
{
    /// Build a global MPHF by reading canonical k-mers from all bin temp files.
    ///
    /// Phase 3a: Each bin is deduplicated independently in parallel (canonical minimizers
    /// guarantee no cross-bin duplicates). Deduplicated k-mers are written back to binary files.
    ///
    /// Phase 3b: The MPHF is built by streaming k-mers from the binary files using
    /// `CachedKeySet`, avoiding holding the entire keyset in memory.
    pub fn build(partitioning: &Partitioning, work_dir: &Path, memory_budget_bytes: usize) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        let num_bins = partitioning.num_bins();
        info!("Reading canonical k-mers from {} bins", num_bins);

        // Phase 3a: Dedup each bin's k-mers and write to binary files.
        let per_bin_counts: Vec<usize> = (0..num_bins)
            .into_par_iter()
            .map(|bin| {
                let src = bin_file_path(work_dir, bin);
                let dst = dedup_file_path(work_dir, bin);
                dedup_bin::<K>(&src, &dst)
            })
            .collect();

        let total: usize = per_bin_counts.iter().sum();
        info!(
            "Total {} distinct canonical k-mers across {} bins",
            total, num_bins
        );

        // Phase 3b: Consolidate per-bin dedup files into a single file for parallel streaming.
        let consolidated_path = work_dir.join("cf1_dedup_all.bin");
        {
            let out_file = std::fs::File::create(&consolidated_path)?;
            let mut writer = BufWriter::with_capacity(256 * 1024, out_file);
            for bin in 0..num_bins {
                let bin_path = dedup_file_path(work_dir, bin);
                if let Ok(mut f) = std::fs::File::open(&bin_path) {
                    std::io::copy(&mut f, &mut writer)?;
                }
                let _ = std::fs::remove_file(&bin_path);
            }
            writer.flush()?;
        }

        let kmer_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();
        let clone_threshold = memory_budget_bytes / kmer_size;
        info!(
            "Building global MPHF for {} distinct canonical k-mers (clone_threshold={}, threads={})",
            total, clone_threshold, rayon::current_num_threads()
        );

        let storage_size = kmer_size;
        let consolidated = Arc::new(consolidated_path.clone());
        let consolidated2 = Arc::clone(&consolidated);

        // Provide both sequential and parallel iterators via tuple.
        // CachedKeySet with (seq_closure, par_closure) enables ph::fmph's parallel
        // construction path (has_par_for_each_key() returns true).
        // The parallel iterator splits the file into chunks for multi-threaded reading.
        let keys = CachedKeySet::dynamic_with_len(
            (
                // Sequential iterator (used for retain_keys, into_vec).
                move || KmerFileIterator::<K>::new(Arc::clone(&consolidated), storage_size),
                // Parallel iterator (used for map_each_key, for_each_key).
                move || KmerFileParIter::<K>::new(Arc::clone(&consolidated2), total, storage_size),
            ),
            total,
            clone_threshold,
        );

        // Use relative_level_size=200 to match C++ BooPHF's gamma=2.0.
        let conf = ph::fmph::BuildConf {
            relative_level_size: 200,
            use_multiple_threads: rayon::current_num_threads() > 1,
            ..Default::default()
        };
        let mphf = ph::fmph::Function::with_conf(keys, conf);

        info!("MPHF construction complete");

        // Cleanup consolidated file.
        let _ = std::fs::remove_file(&consolidated_path);

        Ok(Mphf {
            mphf,
            total_kmers: total as u64,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Look up the global hash index for a canonical k-mer.
    #[inline]
    pub fn hash(&self, canonical_kmer: &Kmer<K>) -> u64 {
        self.mphf.get(&KmerKey(*canonical_kmer)).unwrap_or(0)
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

/// Bridge trait for radix-sorting Storage types (u64, u128).
pub trait RadixSortDedup {
    fn radix_sort_dedup(vec: &mut Vec<Self>) where Self: Sized;
}

impl RadixSortDedup for u64 {
    fn radix_sort_dedup(vec: &mut Vec<Self>) {
        vec.voracious_sort();
        vec.dedup();
    }
}

impl RadixSortDedup for u128 {
    fn radix_sort_dedup(vec: &mut Vec<Self>) {
        vec.voracious_sort();
        vec.dedup();
    }
}

/// Path for a deduplicated k-mer binary file.
fn dedup_file_path(work_dir: &Path, bin: usize) -> PathBuf {
    work_dir.join(format!("cf1_dedup_{:05}.bin", bin))
}

/// Read a bin's packed super k-mers, extract and deduplicate canonical k-mers
/// using radix sort, and write the unique canonical k-mers to a binary file.
/// Returns the count of distinct canonical k-mers.
fn dedup_bin<const K: usize>(src: &Path, dst: &Path) -> usize
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
{
    let storage_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();

    // Collect all canonical k-mer storage values.
    let mut kmer_vals: Vec<<Kmer<K> as KmerBits>::Storage> = Vec::new();

    if let Ok(mut reader) = BinReader::new(src) {
        while let Ok(Some((base_count, packed))) = reader.read_packed_superkmer() {
            if base_count >= K {
                let num_kmers = base_count - K + 1;
                for i in 0..num_kmers {
                    let kmer = Kmer::<K>::from_packed_2bit(packed, i);
                    let canonical = kmer.canonical();
                    kmer_vals.push(canonical.bits);
                }
            }
        }
    }

    // Radix sort and deduplicate.
    RadixSortDedup::radix_sort_dedup(&mut kmer_vals);

    let count = kmer_vals.len();

    // Write deduplicated canonical k-mers as raw binary (native endian Storage bytes).
    let file = std::fs::File::create(dst).expect("Failed to create dedup file");
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    for val in &kmer_vals {
        let bytes = <Kmer<K> as KmerBits>::as_bytes(val);
        writer
            .write_all(&bytes[..storage_size])
            .expect("Failed to write k-mer");
    }
    writer.flush().expect("Failed to flush dedup file");

    count
}

/// Iterator that reads KmerKey<K> values from the consolidated binary file via BufReader.
struct KmerFileIterator<const K: usize>
where
    Kmer<K>: KmerBits,
{
    reader: std::io::BufReader<std::fs::File>,
    storage_size: usize,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> KmerFileIterator<K>
where
    Kmer<K>: KmerBits,
{
    fn new(path: Arc<PathBuf>, storage_size: usize) -> Self {
        let file = std::fs::File::open(path.as_ref()).expect("Failed to open consolidated dedup file");
        KmerFileIterator {
            reader: std::io::BufReader::with_capacity(256 * 1024, file),
            storage_size,
            _phantom: std::marker::PhantomData,
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

/// Parallel iterator over k-mers in a consolidated binary file.
/// Splits the file into chunks that rayon processes on separate threads.
/// Each chunk reads a contiguous range of k-mers using its own BufReader.
struct KmerFileParIter<const K: usize>
where
    Kmer<K>: KmerBits,
{
    path: Arc<PathBuf>,
    total: usize,
    storage_size: usize,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> KmerFileParIter<K>
where
    Kmer<K>: KmerBits,
{
    fn new(path: Arc<PathBuf>, total: usize, storage_size: usize) -> Self {
        KmerFileParIter {
            path,
            total,
            storage_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<const K: usize> rayon::iter::ParallelIterator for KmerFileParIter<K>
where
    Kmer<K>: KmerBits,
    KmerKey<K>: Send,
{
    type Item = KmerKey<K>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.total)
    }
}

impl<const K: usize> rayon::iter::IndexedParallelIterator for KmerFileParIter<K>
where
    Kmer<K>: KmerBits,
    KmerKey<K>: Send,
{
    fn len(&self) -> usize {
        self.total
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(KmerFileProducer::<K> {
            path: self.path,
            start: 0,
            end: self.total,
            storage_size: self.storage_size,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Rayon Producer that splits a byte range of the k-mer file.
struct KmerFileProducer<const K: usize>
where
    Kmer<K>: KmerBits,
{
    path: Arc<PathBuf>,
    start: usize,
    end: usize,
    storage_size: usize,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> rayon::iter::plumbing::Producer for KmerFileProducer<K>
where
    Kmer<K>: KmerBits,
    KmerKey<K>: Send,
{
    type Item = KmerKey<K>;
    type IntoIter = KmerChunkIterator<K>;

    fn into_iter(self) -> Self::IntoIter {
        use std::io::Seek;
        let mut file =
            std::fs::File::open(self.path.as_ref()).expect("Failed to open consolidated dedup file");
        file.seek(std::io::SeekFrom::Start(
            (self.start * self.storage_size) as u64,
        ))
        .expect("Failed to seek in dedup file");
        KmerChunkIterator {
            reader: std::io::BufReader::with_capacity(256 * 1024, file),
            remaining: self.end - self.start,
            storage_size: self.storage_size,
            _phantom: std::marker::PhantomData,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.start + index;
        (
            KmerFileProducer {
                path: Arc::clone(&self.path),
                start: self.start,
                end: mid,
                storage_size: self.storage_size,
                _phantom: std::marker::PhantomData,
            },
            KmerFileProducer {
                path: self.path,
                start: mid,
                end: self.end,
                storage_size: self.storage_size,
                _phantom: std::marker::PhantomData,
            },
        )
    }
}

/// Sequential iterator over a chunk of k-mers in the file.
struct KmerChunkIterator<const K: usize>
where
    Kmer<K>: KmerBits,
{
    reader: std::io::BufReader<std::fs::File>,
    remaining: usize,
    storage_size: usize,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> Iterator for KmerChunkIterator<K>
where
    Kmer<K>: KmerBits,
{
    type Item = KmerKey<K>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        use std::io::Read;
        let mut kmer = Kmer::<K>::default();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut kmer.bits as *mut <Kmer<K> as KmerBits>::Storage as *mut u8,
                self.storage_size,
            )
        };
        match self.reader.read_exact(bytes) {
            Ok(()) => {
                self.remaining -= 1;
                Some(KmerKey(kmer))
            }
            Err(_) => {
                self.remaining = 0;
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const K: usize> ExactSizeIterator for KmerChunkIterator<K>
where
    Kmer<K>: KmerBits,
{
}

impl<const K: usize> DoubleEndedIterator for KmerChunkIterator<K>
where
    Kmer<K>: KmerBits,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // Required by rayon's bridge but we only iterate forward.
        // Decrement remaining and read the next item forward.
        // This is safe because rayon's bridge only calls next_back
        // to reduce the iterator length, never to actually reverse.
        self.next()
    }
}

/// Wrapper to implement Hash for Kmer in the context of MPHF.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct KmerKey<const K: usize>(pub(crate) Kmer<K>)
where
    Kmer<K>: KmerBits;

use std::hash::{Hash, Hasher};

impl<const K: usize> Hash for KmerKey<K>
where
    Kmer<K>: KmerBits,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use the Storage type's native Hash impl (write_u64 for u64, write_u128 for u128).
        // This is more efficient than going through bytes and works better with
        // the MPHF's seeded hash family.
        self.0.bits.hash(state);
    }
}
