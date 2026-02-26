use cf1_rs::params::{Cli, Commands, Params};
use cf1_rs::pipeline::run_pipeline;
use clap::Parser;
use tracing::info;

fn main() -> anyhow::Result<()> {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            seq_file,
            list_file,
            dir_path,
            k,
            threads,
            output,
            format,
            work_dir,
            track_short_seqs,
            poly_n_stretch,
            collate_in_mem,
            num_bins,
            memory_budget_gb,
        } => {
            let params = Params::from_build_args(
                seq_file,
                list_file,
                dir_path,
                k,
                threads,
                output,
                format,
                work_dir,
                track_short_seqs,
                poly_n_stretch,
                collate_in_mem,
                num_bins,
                memory_budget_gb,
            )?;

            info!(
                "Building compacted de Bruijn graph with k={}, threads={}, format={}",
                params.k, params.threads, params.format
            );

            // Set up rayon thread pool.
            rayon::ThreadPoolBuilder::new()
                .num_threads(params.threads)
                .build_global()
                .ok();

            // Dispatch to the appropriate monomorphized pipeline.
            cf1_rs::dispatch_k!(params.k, run, &params)?;

            info!("Done.");
        }
    }

    Ok(())
}

fn run<const K: usize>(params: &Params) -> anyhow::Result<()>
where
    cf1_rs::kmer::Kmer<K>: cf1_rs::kmer::KmerBits,
    <cf1_rs::kmer::Kmer<K> as cf1_rs::kmer::KmerBits>::Storage: cf1_rs::mphf::RadixSortDedup,
{
    let _ = run_pipeline::<K>(params)?;
    Ok(())
}
