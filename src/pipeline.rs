use crate::classify::classify_vertices;
use crate::kmer::{Kmer, KmerBits};
use crate::minimizer::{count_minimizer_frequencies, partition_minimizers};
use crate::mphf::Mphf;
use crate::output::{extract_and_output, write_json};
use crate::params::Params;
use crate::state_vector::AtomicStateVector;
use crate::superkmer::{cleanup_bin_files, route_superkmers};
use tracing::info;

pub fn run_pipeline<const K: usize>(params: &Params) -> anyhow::Result<()>
where
    Kmer<K>: KmerBits,
    <Kmer<K> as KmerBits>::Storage: crate::mphf::RadixSortDedup,
{
    // Phase 1: Count minimizer frequencies.
    info!("Phase 1: Counting minimizer frequencies...");
    let min_counts = count_minimizer_frequencies(params)?;

    // Phase 2: Partition + route super k-mers.
    info!("Phase 2: Partitioning minimizers and routing super k-mers...");
    let partitioning = partition_minimizers(&min_counts, params.num_bins)?;
    route_superkmers(params, &partitioning)?;

    // Phase 3: Build global MPHF.
    info!("Phase 3: Building global MPHF...");
    let num_bins = partitioning.num_bins();
    let mphf = Mphf::<K>::build(&partitioning, &params.work_dir, params.memory_budget_bytes)?;
    let states = AtomicStateVector::new(mphf.total_kmers() as usize);

    // Clean up bin temp files.
    cleanup_bin_files(&params.work_dir, num_bins);

    // Phase 4: DFA classification.
    info!("Phase 4: Classifying k-mer vertices...");
    let short_seqs = classify_vertices::<K>(params, &mphf, &states)?;

    // Phase 5: Output.
    info!("Phase 5: Extracting unitigs and writing output...");
    let meta = extract_and_output::<K>(params, &mphf, &states)?;

    // Write JSON.
    write_json(params, &meta, &short_seqs)?;

    info!("Pipeline complete.");
    Ok(())
}
