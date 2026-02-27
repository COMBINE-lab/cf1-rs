use std::sync::atomic::{AtomicUsize, Ordering};

/// Controls memory usage by tracking total in-flight sequence bytes.
///
/// The reader thread calls `wait_and_acquire` before copying a sequence for processing.
/// When a task finishes, it calls `release` to free its budget.
/// The `current == 0` check in `wait_and_acquire` guarantees progress: even a sequence
/// larger than `max_bytes` will be processed if nothing else is in flight.
pub struct InFlightBudget {
    bytes: AtomicUsize,
    max_bytes: usize,
    peak_bytes: AtomicUsize,
}

impl InFlightBudget {
    pub fn new(max_bytes: usize) -> Self {
        InFlightBudget {
            bytes: AtomicUsize::new(0),
            max_bytes,
            peak_bytes: AtomicUsize::new(0),
        }
    }

    /// Block until we have budget, OR nothing is in flight (guarantees progress).
    pub fn wait_and_acquire(&self, amount: usize) {
        loop {
            let current = self.bytes.load(Ordering::Relaxed);
            if current == 0 || current + amount <= self.max_bytes {
                let new = self.bytes.fetch_add(amount, Ordering::Relaxed) + amount;
                // Track peak in-flight bytes.
                self.peak_bytes.fetch_max(new, Ordering::Relaxed);
                return;
            }
            std::thread::yield_now();
        }
    }

    /// Release budget after a sequence (or chunk group) is done processing.
    pub fn release(&self, amount: usize) {
        self.bytes.fetch_sub(amount, Ordering::SeqCst);
    }

    /// Chunk threshold: sequences larger than this should be split across threads.
    pub fn chunk_threshold(&self) -> usize {
        self.max_bytes / rayon::current_num_threads().max(1)
    }

    /// Return and reset peak in-flight bytes (for diagnostics).
    pub fn take_peak_bytes(&self) -> usize {
        self.peak_bytes.swap(0, Ordering::Relaxed)
    }

    /// Return current in-flight bytes.
    pub fn current_bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }
}
