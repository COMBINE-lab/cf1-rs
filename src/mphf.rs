use crate::dna::Base;
use crate::kmer::{Kmer, KmerBits};
use crate::minimizer::Partitioning;
use crate::superkmer::{bin_file_path, packed_superkmers};
use ph::fmph::keyset::CachedKeySet;
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// PHast+ (8-bit seeds, wrapped shifts): the in-memory MPHF.
type PhastFunction = ph::phast::Function2<ph::seeds::Bits8, ph::phast::ShiftOnlyWrapped>;

/// The MPHF backing [`Mphf`], chosen by construction regime.
///
/// Phases 4 and 5 evaluate the MPHF once per k-mer *occurrence* in the input
/// (on a transcriptome, several times the number of distinct k-mers), so
/// evaluation speed dominates their running time. PHast+ answers almost every
/// query with one seed read, where FMPH probes a bit array per level and then a
/// rank structure — about twice the cache misses. Its construction needs every
/// key in memory, though, so inputs too large for the in-memory regime keep
/// FMPH, which can be built by streaming the keys from disk.
enum MphfImpl {
    Fmph(ph::fmph::Function),
    Phast(PhastFunction),
}

/// Build the in-memory-regime MPHF over `keys`.
fn build_phast<const K: usize>(keys: Vec<KmerKey<K>>) -> MphfImpl
where
    Kmer<K>: KmerBits,
{
    MphfImpl::Phast(PhastFunction::with_vec_p_threads_hash_sc(
        keys,
        &ph::phast::Params::new(
            ph::seeds::Bits8,
            ph::phast::bits_per_seed_to_100_bucket_size(8),
        ),
        rayon::current_num_threads(),
        ph::BuildDefaultSeededHasher::default(),
        ph::phast::ShiftOnlyWrapped,
    ))
}

/// Global minimal perfect hash function over all canonical k-mers.
pub struct Mphf<const K: usize>
where
    Kmer<K>: KmerBits,
{
    mphf: MphfImpl,
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
    /// guarantee no cross-bin duplicates). In the in-memory regime the deduplicated
    /// k-mers stay in memory; otherwise they are written back to binary files.
    ///
    /// Phase 3b (streaming regime only): The MPHF is built by streaming k-mers from
    /// the binary files using `CachedKeySet`, avoiding holding the entire keyset in
    /// memory.
    pub fn build(
        partitioning: &Partitioning,
        work_dir: &Path,
        _memory_budget_bytes: usize,
    ) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        // clone_threshold=300M: good balance of speed and memory on genome-scale data.
        // Keys surviving early MPHF levels are cached in RAM when count drops below this.
        Self::build_with_clone_threshold(partitioning, work_dir, 300_000_000)
    }

    /// Build MPHF with an explicit `clone_threshold`.
    ///
    /// ## Construction strategy
    ///
    /// Two regimes, chosen automatically based on `total` vs `clone_threshold`:
    ///
    /// **In-memory** (`total <= clone_threshold`): All k-mers are read into a `Vec`
    /// and a PHast+ function is built over them (see [`MphfImpl`] for why). Peak
    /// RSS is dominated by the key Vec plus PHast's sorted level-0 hashes
    /// (`total × 8` bytes each). Suitable when `total` fits comfortably in RAM.
    ///
    /// **Streaming** (`total > clone_threshold`): Keys are read from the consolidated
    /// dedup file on every MPHF pass using a **SPMC iterator** (one dedicated I/O
    /// thread reading 16 MB chunks, `N` rayon workers pulling from per-worker channels).
    /// Once the surviving-key count after an early level drops below `clone_threshold`,
    /// `ph`'s `CachedKeySet` clones the survivors into a Vec and subsequent levels run
    /// in-memory. No `mmap` is used: RSS reflects actual heap usage, not mapped pages.
    ///
    /// ## Default threshold
    ///
    /// `build()` uses `clone_threshold = 300_000_000`. Benchmarks on a 2.5 B-key
    /// human genome dataset (k=31, 4 threads) showed:
    ///
    /// | clone_threshold | time  | peak RSS |
    /// |-----------------|-------|----------|
    /// | 400 M           | 97 s  | 9.1 GB   |
    /// | 300 M           | 130 s | 3.9 GB   |
    /// | 250 M           | 130 s | 3.8 GB   |
    pub fn build_with_clone_threshold(
        partitioning: &Partitioning,
        work_dir: &Path,
        clone_threshold: usize,
    ) -> anyhow::Result<Self>
    where
        <Kmer<K> as KmerBits>::Storage: RadixSortDedup,
    {
        let num_bins = partitioning.num_bins();
        info!("Reading canonical k-mers from {} bins", num_bins);

        // Phase 3a: Dedup each bin's k-mers, a wave of bins at a time. While the
        // running total of distinct k-mers stays within `clone_threshold` (the
        // in-memory regime), each wave's k-mers are appended, in bin order, to
        // one key vector that is handed to the MPHF build as is — so the keys
        // never touch the disk and never exist twice. Should the total exceed
        // the threshold, the keys gathered so far are written out per bin and
        // every later bin goes straight to its file, for streaming.
        info!(
            "P3a: RSS before dedup: current={} MB, peak={} MB",
            crate::pipeline::current_rss_mb(),
            crate::pipeline::peak_rss_mb()
        );
        // Reserves address space only: pages are committed as keys are written.
        let mut keys: Option<Vec<KmerKey<K>>> = Some(Vec::with_capacity(clone_threshold));
        let mut per_bin_counts: Vec<usize> = Vec::with_capacity(num_bins);
        let bins: Vec<usize> = (0..num_bins).collect();
        for wave in bins.chunks(2 * rayon::current_num_threads()) {
            let deduped: Vec<_> = wave
                .par_iter()
                .map(|&bin| dedup_bin_kmers::<K>(&bin_file_path(work_dir, bin)))
                .collect();
            let wave_total: usize = deduped.iter().map(Vec::len).sum();
            let first_bin = per_bin_counts.len();
            per_bin_counts.extend(deduped.iter().map(Vec::len));

            if let Some(kept) = &keys
                && kept.len() + wave_total > clone_threshold
            {
                // Leaving the in-memory regime: write out the bins kept so far.
                let mut rest = &kept[..];
                for (bin, &count) in per_bin_counts[..first_bin].iter().enumerate() {
                    let (bin_keys, tail) = rest.split_at(count);
                    write_kmers::<K>(
                        &dedup_file_path(work_dir, bin),
                        bin_keys.iter().map(|key| key.0.bits),
                    );
                    rest = tail;
                }
                keys = None;
            }
            match &mut keys {
                Some(kept) => append_par(kept, &deduped),
                None => wave.par_iter().zip(&deduped).for_each(|(&bin, kmers)| {
                    write_kmers::<K>(&dedup_file_path(work_dir, bin), kmers.iter().copied())
                }),
            }
        }

        let total: usize = per_bin_counts.iter().sum();
        let max_bin = per_bin_counts.iter().max().unwrap_or(&0);
        let min_bin = per_bin_counts.iter().min().unwrap_or(&0);
        info!(
            "Total {} distinct canonical k-mers across {} bins (per-bin: min={}, max={}, avg={})",
            total,
            num_bins,
            min_bin,
            max_bin,
            total / num_bins.max(1)
        );
        info!(
            "P3a: RSS after dedup: current={} MB, peak={} MB",
            crate::pipeline::current_rss_mb(),
            crate::pipeline::peak_rss_mb()
        );

        if let Some(keys) = keys {
            // Small input: build PHast+ over the in-memory keys; its evaluation
            // is what phases 4 and 5 are bound by (see `MphfImpl`).
            info!(
                "Building global MPHF for {} distinct canonical k-mers (in-memory, threads={})",
                total,
                rayon::current_num_threads()
            );
            let mphf = build_phast(keys);
            info!("MPHF construction complete");
            return Ok(Mphf {
                mphf,
                total_kmers: total as u64,
                _phantom: std::marker::PhantomData,
            });
        }

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

        info!(
            "P3b: RSS after consolidation: current={} MB, peak={} MB",
            crate::pipeline::current_rss_mb(),
            crate::pipeline::peak_rss_mb()
        );

        let storage_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();
        let mt = rayon::current_num_threads() > 1;

        // Stream k-mers from disk via SPMC parallel iterator.
        // CachedKeySet caches surviving keys once count drops below threshold.
        info!(
            "Building global MPHF for {} distinct canonical k-mers (streaming, clone_threshold={}, threads={})",
            total,
            clone_threshold,
            rayon::current_num_threads()
        );
        let conf = ph::fmph::BuildConf {
            relative_level_size: 200,
            use_multiple_threads: mt,
            ..Default::default()
        };
        let consolidated = Arc::new(consolidated_path.clone());
        let consolidated2 = Arc::clone(&consolidated);
        let keys = CachedKeySet::dynamic_with_len(
            (
                move || KmerFileIterator::<K>::new(Arc::clone(&consolidated), storage_size),
                move || KmerSpmcParIter::<K>::new(Arc::clone(&consolidated2), total, storage_size),
            ),
            total,
            clone_threshold,
        );
        let mphf = MphfImpl::Fmph(ph::fmph::Function::with_conf(keys, conf));

        info!("MPHF construction complete");

        // Cleanup consolidated file.
        std::fs::remove_file(&consolidated_path)?;

        Ok(Mphf {
            mphf,
            total_kmers: total as u64,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Build MPHF from a pre-existing consolidated dedup file.
    /// Skips dedup and consolidation — just builds the MPHF.
    pub fn build_from_consolidated(
        consolidated_path: &Path,
        total: usize,
        clone_threshold: usize,
    ) -> anyhow::Result<Self> {
        let kmer_size = std::mem::size_of::<<Kmer<K> as KmerBits>::Storage>();
        let storage_size = kmer_size;

        let mt = rayon::current_num_threads() > 1;

        let mphf = if total <= clone_threshold {
            info!(
                "Building global MPHF for {} distinct canonical k-mers (in-memory, threads={})",
                total,
                rayon::current_num_threads()
            );
            build_phast(read_all_kmers::<K>(consolidated_path, storage_size))
        } else {
            info!(
                "Building global MPHF for {} distinct canonical k-mers (streaming, clone_threshold={}, threads={})",
                total,
                clone_threshold,
                rayon::current_num_threads()
            );
            let conf = ph::fmph::BuildConf {
                relative_level_size: 200,
                use_multiple_threads: mt,
                ..Default::default()
            };
            let consolidated = Arc::new(consolidated_path.to_path_buf());
            let consolidated2 = Arc::clone(&consolidated);
            let keys = CachedKeySet::dynamic_with_len(
                (
                    move || KmerFileIterator::<K>::new(Arc::clone(&consolidated), storage_size),
                    move || {
                        KmerSpmcParIter::<K>::new(Arc::clone(&consolidated2), total, storage_size)
                    },
                ),
                total,
                clone_threshold,
            );
            MphfImpl::Fmph(ph::fmph::Function::with_conf(keys, conf))
        };

        info!("MPHF construction complete");

        Ok(Mphf {
            mphf,
            total_kmers: total as u64,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Look up the global hash index for a canonical k-mer.
    // Always inlined: it sits in the per-k-mer loops of phases 4 and 5, and the
    // dispatch on the MPHF kind would otherwise make it too big to inline there.
    #[inline(always)]
    pub fn hash(&self, canonical_kmer: &Kmer<K>) -> u64 {
        match &self.mphf {
            MphfImpl::Fmph(f) => f.get(&KmerKey(*canonical_kmer)).unwrap_or(0),
            MphfImpl::Phast(f) => f.get(&KmerKey(*canonical_kmer)) as u64,
        }
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

/// Path for a deduplicated k-mer binary file.
fn dedup_file_path(work_dir: &Path, bin: usize) -> PathBuf {
    work_dir.join(format!("cf1_dedup_{:05}.bin", bin))
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
        let file =
            std::fs::File::open(path.as_ref()).expect("Failed to open consolidated dedup file");
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

/// Read all k-mers from a consolidated binary file into a Vec.
/// Used for small inputs where the entire keyset fits comfortably in RAM.
fn read_all_kmers<const K: usize>(path: &Path, storage_size: usize) -> Vec<KmerKey<K>>
where
    Kmer<K>: KmerBits,
{
    use std::io::Read;
    let file = std::fs::File::open(path).expect("Failed to open consolidated dedup file");
    let file_len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let expected = file_len / storage_size;
    let mut reader = std::io::BufReader::with_capacity(256 * 1024, file);
    let mut keys = Vec::with_capacity(expected);
    loop {
        let mut kmer = Kmer::<K>::default();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut kmer.bits as *mut <Kmer<K> as KmerBits>::Storage as *mut u8,
                storage_size,
            )
        };
        match reader.read_exact(bytes) {
            Ok(()) => keys.push(KmerKey(kmer)),
            Err(_) => break,
        }
    }
    keys
}

// ── SPMC (Single-Producer, Multiple-Consumer) parallel iterator ──────────────
//
// Mirrors the C++ cuttlefish `Kmer_SPMC_Iterator` design:
// - One background thread reads the consolidated file sequentially in 16 MB chunks
// - Worker threads receive batches via per-worker bounded channels
// - Zero per-element syscalls from worker threads
// - No mmap — RSS reflects actual heap usage only

/// 16 MB read chunks — matches C++ cuttlefish's BUF_SZ_PER_CONSUMER.
const SPMC_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// A batch of raw bytes from the producer, to be parsed into k-mers by a worker.
/// Uses a pooled buffer to avoid allocation per batch.
struct RawBatch {
    buf: Vec<u8>,
    valid: usize,
}

/// SPMC parallel iterator. Created fresh for each pass over the k-mers.
struct KmerSpmcParIter<const K: usize>
where
    Kmer<K>: KmerBits,
{
    path: Arc<PathBuf>,
    total: usize,
    storage_size: usize,
    _phantom: std::marker::PhantomData<Kmer<K>>,
}

impl<const K: usize> KmerSpmcParIter<K>
where
    Kmer<K>: KmerBits,
{
    fn new(path: Arc<PathBuf>, total: usize, storage_size: usize) -> Self {
        KmerSpmcParIter {
            path,
            total,
            storage_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<const K: usize> rayon::iter::ParallelIterator for KmerSpmcParIter<K>
where
    Kmer<K>: KmerBits,
    KmerKey<K>: Send + Sync,
{
    type Item = KmerKey<K>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let num_workers = rayon::current_num_threads();
        let storage_size = self.storage_size;

        // Per-worker channels: producer sends RawBatch, workers receive.
        // Bound of 2 gives double-buffering per worker.
        let mut senders = Vec::with_capacity(num_workers);
        let mut receivers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, rx) = crossbeam_channel::bounded::<RawBatch>(2);
            senders.push(tx);
            receivers.push(rx);
        }

        // Return-path channel: workers send back empty buffers for reuse.
        let (buf_return_tx, buf_return_rx) = crossbeam_channel::bounded::<Vec<u8>>(num_workers * 3);

        // Pre-allocate buffer pool.
        for _ in 0..num_workers * 3 {
            let _ = buf_return_tx.send(vec![0u8; SPMC_CHUNK_BYTES]);
        }

        // Spawn producer thread (outside rayon pool — dedicated I/O thread).
        let path = self.path.as_ref().clone();
        let producer_handle = std::thread::spawn(move || {
            use std::io::Read;
            let mut file = std::fs::File::open(&path)
                .expect("SPMC producer: failed to open consolidated file");
            let mut worker_idx = 0;

            loop {
                // Get a buffer from the pool (or allocate if pool exhausted).
                let mut buf = buf_return_rx
                    .recv()
                    .unwrap_or_else(|_| vec![0u8; SPMC_CHUNK_BYTES]);

                let max_bytes = (buf.len() / storage_size) * storage_size;
                let mut total_read = 0;
                while total_read < max_bytes {
                    match file.read(&mut buf[total_read..max_bytes]) {
                        Ok(0) => break,
                        Ok(n) => total_read += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => panic!("SPMC producer: read error: {}", e),
                    }
                }
                total_read = (total_read / storage_size) * storage_size;

                if total_read == 0 {
                    // EOF — drop all senders to signal workers.
                    drop(senders);
                    break;
                }

                let batch = RawBatch {
                    buf,
                    valid: total_read,
                };
                // Send to current worker; if channel is full, this blocks
                // (backpressure from slow worker).
                if senders[worker_idx].send(batch).is_err() {
                    break; // Worker dropped — iteration was cut short.
                }
                worker_idx = (worker_idx + 1) % num_workers;
            }
        });

        // Split the rayon consumer into per-worker sub-consumers, run workers
        // in the rayon thread pool via `join` tree, then reduce results.
        let result = spmc_drive_workers::<K, C>(
            &receivers,
            &buf_return_tx,
            storage_size,
            consumer,
            0,
            num_workers,
        );

        producer_handle
            .join()
            .expect("SPMC producer thread panicked");
        result
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.total)
    }
}

/// Recursively split the rayon consumer and drive each worker.
/// This builds a binary join-tree so rayon can parallelize the workers.
fn spmc_drive_workers<const K: usize, C>(
    receivers: &[crossbeam_channel::Receiver<RawBatch>],
    buf_return: &crossbeam_channel::Sender<Vec<u8>>,
    storage_size: usize,
    consumer: C,
    start: usize,
    end: usize,
) -> C::Result
where
    Kmer<K>: KmerBits,
    KmerKey<K>: Send + Sync,
    C: rayon::iter::plumbing::UnindexedConsumer<KmerKey<K>>,
{
    use rayon::iter::plumbing::*;

    if end - start == 1 {
        // Leaf: this worker pulls batches from its channel and feeds the folder.
        let mut folder = consumer.into_folder();
        let rx = &receivers[start];
        while let Ok(batch) = rx.recv() {
            let buf = &batch.buf[..batch.valid];
            let mut pos = 0;
            while pos + storage_size <= batch.valid {
                let mut kmer = Kmer::<K>::default();
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut kmer.bits as *mut <Kmer<K> as KmerBits>::Storage as *mut u8,
                        storage_size,
                    )
                };
                dst.copy_from_slice(&buf[pos..pos + storage_size]);
                pos += storage_size;
                folder = folder.consume(KmerKey(kmer));
                if folder.full() {
                    // Return the buffer and stop.
                    let _ = buf_return.send(batch.buf);
                    return folder.complete();
                }
            }
            // Return buffer to pool for reuse.
            let _ = buf_return.send(batch.buf);
        }
        folder.complete()
    } else {
        // Internal node: split consumer and recurse with rayon::join.
        let mid = start + (end - start) / 2;
        let (left_consumer, right_consumer, reducer) = consumer.split_at(mid - start);
        let (left, right) = rayon::join(
            || {
                spmc_drive_workers::<K, _>(
                    receivers,
                    buf_return,
                    storage_size,
                    left_consumer,
                    start,
                    mid,
                )
            },
            || {
                spmc_drive_workers::<K, _>(
                    receivers,
                    buf_return,
                    storage_size,
                    right_consumer,
                    mid,
                    end,
                )
            },
        );
        reducer.reduce(left, right)
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
