use crate::state::State;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic bit-packed vector storing 5-bit state values.
/// Packs 12 states per AtomicU64 (60 bits used, 4 wasted).
/// No element spans word boundaries.
pub struct AtomicStateVector {
    data: Vec<AtomicU64>,
    len: usize,
}

impl AtomicStateVector {
    const STATES_PER_WORD: usize = 12;
    const BITS_PER_STATE: usize = 5;
    const STATE_MASK: u64 = 0x1F;

    pub fn new(len: usize) -> Self {
        let num_words = (len + Self::STATES_PER_WORD - 1) / Self::STATES_PER_WORD;
        let mut data = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            data.push(AtomicU64::new(0));
        }
        AtomicStateVector { data, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Get the state at index i.
    #[inline]
    pub fn get(&self, i: usize) -> State {
        debug_assert!(i < self.len);
        let word_idx = i / Self::STATES_PER_WORD;
        let shift = (i % Self::STATES_PER_WORD) * Self::BITS_PER_STATE;
        let word = self.data[word_idx].load(Ordering::Acquire);
        State(((word >> shift) & Self::STATE_MASK) as u8)
    }

    /// Compare-and-swap: if the state at index i equals expected, replace with new_val.
    /// Returns true if the swap succeeded or was unnecessary.
    #[inline]
    pub fn compare_and_swap(&self, i: usize, expected: State, new_val: State) -> bool {
        debug_assert!(i < self.len);
        let word_idx = i / Self::STATES_PER_WORD;
        let shift = (i % Self::STATES_PER_WORD) * Self::BITS_PER_STATE;
        let mask = Self::STATE_MASK << shift;

        loop {
            let current = self.data[word_idx].load(Ordering::Acquire);
            let current_state = ((current >> shift) & Self::STATE_MASK) as u8;
            if current_state != expected.code() {
                return false;
            }
            let new_word = (current & !mask) | ((new_val.code() as u64) << shift);
            match self.data[word_idx].compare_exchange_weak(
                current,
                new_word,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }
}

// Safety: AtomicStateVector uses AtomicU64 internally and is safe to share.
unsafe impl Send for AtomicStateVector {}
unsafe impl Sync for AtomicStateVector {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let vec = AtomicStateVector::new(100);
        // Initially all unvisited (0)
        assert_eq!(vec.get(0).code(), 0);
        assert_eq!(vec.get(99).code(), 0);

        // CAS from 0 to 3 (MIMO)
        assert!(vec.compare_and_swap(0, State(0), State(3)));
        assert_eq!(vec.get(0).code(), 3);

        // CAS should fail if expected doesn't match
        assert!(!vec.compare_and_swap(0, State(0), State(5)));
        assert_eq!(vec.get(0).code(), 3);

        // CAS at different positions
        assert!(vec.compare_and_swap(11, State(0), State(16)));
        assert_eq!(vec.get(11).code(), 16);

        // Position 12 should be in the next word
        assert!(vec.compare_and_swap(12, State(0), State(31)));
        assert_eq!(vec.get(12).code(), 31);
    }

    #[test]
    fn test_all_positions_in_word() {
        let vec = AtomicStateVector::new(12);
        for i in 0..12 {
            let val = State((i as u8 % 32).max(3)); // Use valid state codes
            assert!(vec.compare_and_swap(i, State(0), val));
            assert_eq!(vec.get(i), val);
        }
    }
}
