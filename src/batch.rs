use crate::budget::InFlightBudget;

/// Number of bases at which a [`SeqBatch`] is handed to a worker.
///
/// Large enough that per-task setup (local histograms, per-bin buffers) and
/// the merge into shared state are amortized over many sequences; small enough
/// that the batches of a transcriptome still spread over every thread.
pub const BATCH_BASES: usize = 4_000_000;

/// Sequences concatenated into one buffer, processed by a single parallel task.
///
/// A transcriptome has hundreds of thousands of short sequences. Spawning a
/// task per sequence makes every task pay for its own scratch state and its
/// own merge into shared structures; batching pays those once per batch.
#[derive(Default)]
pub struct SeqBatch {
    data: Vec<u8>,
    /// `ends[i]` is the end offset of the `i`-th sequence in `data`.
    ends: Vec<usize>,
}

impl SeqBatch {
    /// Append a sequence.
    pub fn push(&mut self, seq: &[u8]) {
        self.data.extend_from_slice(seq);
        self.ends.push(self.data.len());
    }

    /// Total bases held.
    pub fn bases(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The sequences, in the order they were pushed.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts
            .zip(&self.ends)
            .map(|(start, &end)| &self.data[start..end])
    }

    /// Spawn `process` on this batch in `scope`, holding the batch's bases of
    /// `budget` until it finishes. An empty batch is dropped.
    pub fn spawn<'scope>(
        self,
        scope: &rayon::Scope<'scope>,
        budget: &'scope InFlightBudget,
        process: impl FnOnce(&SeqBatch) + Send + 'scope,
    ) {
        if self.is_empty() {
            return;
        }
        let bases = self.bases();
        budget.wait_and_acquire(bases);
        scope.spawn(move |_| {
            process(&self);
            budget.release(bases);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterates_pushed_sequences_in_order() {
        let mut batch = SeqBatch::default();
        assert!(batch.is_empty());
        batch.push(b"ACGT");
        batch.push(b"");
        batch.push(b"GG");
        assert_eq!(batch.bases(), 6);
        let seqs: Vec<&[u8]> = batch.iter().collect();
        assert_eq!(seqs, [&b"ACGT"[..], b"", b"GG"]);
    }
}
