use crate::dna::Base;
use std::hash::{Hash, Hasher};

/// Reverse complement lookup table for bytes (4 bases per byte).
/// Each byte encodes 4 bases in 2-bit format. The reverse complement reverses
/// the order and complements each base. Matches C++ Kmer_Utility::REVERSE_COMPLEMENT_BYTE.
pub const REVERSE_COMPLEMENT_BYTE: [u8; 256] = [
    255, 191, 127, 63, 239, 175, 111, 47, 223, 159, 95, 31, 207, 143, 79, 15, 251, 187, 123, 59,
    235, 171, 107, 43, 219, 155, 91, 27, 203, 139, 75, 11, 247, 183, 119, 55, 231, 167, 103, 39,
    215, 151, 87, 23, 199, 135, 71, 7, 243, 179, 115, 51, 227, 163, 99, 35, 211, 147, 83, 19,
    195, 131, 67, 3, 254, 190, 126, 62, 238, 174, 110, 46, 222, 158, 94, 30, 206, 142, 78, 14,
    250, 186, 122, 58, 234, 170, 106, 42, 218, 154, 90, 26, 202, 138, 74, 10, 246, 182, 118, 54,
    230, 166, 102, 38, 214, 150, 86, 22, 198, 134, 70, 6, 242, 178, 114, 50, 226, 162, 98, 34,
    210, 146, 82, 18, 194, 130, 66, 2, 253, 189, 125, 61, 237, 173, 109, 45, 221, 157, 93, 29,
    205, 141, 77, 13, 249, 185, 121, 57, 233, 169, 105, 41, 217, 153, 89, 25, 201, 137, 73, 9,
    245, 181, 117, 53, 229, 165, 101, 37, 213, 149, 85, 21, 197, 133, 69, 5, 241, 177, 113, 49,
    225, 161, 97, 33, 209, 145, 81, 17, 193, 129, 65, 1, 252, 188, 124, 60, 236, 172, 108, 44,
    220, 156, 92, 28, 204, 140, 76, 12, 248, 184, 120, 56, 232, 168, 104, 40, 216, 152, 88, 24,
    200, 136, 72, 8, 244, 180, 116, 52, 228, 164, 100, 36, 212, 148, 84, 20, 196, 132, 68, 4,
    240, 176, 112, 48, 224, 160, 96, 32, 208, 144, 80, 16, 192, 128, 64, 0,
];

/// Trait mapping K -> storage type. Implemented for each supported K value.
pub trait KmerBits: Sized {
    type Storage: Copy + Clone + Eq + Ord + Hash + Send + Sync + Default + std::fmt::Debug;
    const NUM_WORDS: usize;

    fn word(storage: &Self::Storage, idx: usize) -> u64;
    fn set_word(storage: &mut Self::Storage, idx: usize, val: u64);
    fn as_bytes(storage: &Self::Storage) -> &[u8];
}

#[derive(Clone, Copy, Debug)]
pub struct Kmer<const K: usize>
where
    Kmer<K>: KmerBits,
{
    pub(crate) bits: <Kmer<K> as KmerBits>::Storage,
}

impl<const K: usize> Default for Kmer<K>
where
    Kmer<K>: KmerBits,
{
    fn default() -> Self {
        Kmer {
            bits: Default::default(),
        }
    }
}

impl<const K: usize> PartialEq for Kmer<K>
where
    Kmer<K>: KmerBits,
{
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<const K: usize> Eq for Kmer<K> where Kmer<K>: KmerBits {}

impl<const K: usize> PartialOrd for Kmer<K>
where
    Kmer<K>: KmerBits,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Comparison: high word first (matches C++ operator<).
impl<const K: usize> Ord for Kmer<K>
where
    Kmer<K>: KmerBits,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let n = <Kmer<K> as KmerBits>::NUM_WORDS;
        for idx in (0..n).rev() {
            let a = <Kmer<K> as KmerBits>::word(&self.bits, idx);
            let b = <Kmer<K> as KmerBits>::word(&other.bits, idx);
            match a.cmp(&b) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl<const K: usize> Hash for Kmer<K>
where
    Kmer<K>: KmerBits,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

impl<const K: usize> Kmer<K>
where
    Kmer<K>: KmerBits,
{
    const NUM_WORDS: usize = K.div_ceil(32);
    const CLEAR_MSN_MASK: u64 = !(0b11u64 << (2 * ((K - 1) % 32)));
    const NUM_BYTES: usize = K.div_ceil(4);
    const MSN_SHIFT: usize = 2 * ((K - 1) % 32);

    /// Create a k-mer from ASCII sequence at the given offset.
    pub fn from_ascii(seq: &[u8], offset: usize) -> Self {
        let label = &seq[offset..offset + K];
        let mut kmer = Kmer::default();

        let packed_word_count = K / 32;

        // Get fully packed words' binary representations.
        for data_idx in 0..packed_word_count {
            let start = K - (data_idx << 5) - 32;
            let word = encode_word::<32>(&label[start..start + 32]);
            <Kmer<K> as KmerBits>::set_word(&mut kmer.bits, data_idx, word);
        }

        // Get the partially packed (highest index) word's binary representation.
        let rem = K & 31;
        if rem > 0 {
            let word = encode_word_dyn(rem, &label[0..rem]);
            <Kmer<K> as KmerBits>::set_word(&mut kmer.bits, Self::NUM_WORDS - 1, word);
        }

        kmer
    }

    /// Create a k-mer from 2-bit packed data at the given base offset.
    /// Packed format: 4 bases per byte, MSB-first (base j is at bits 6-2*(j%4) of byte j/4).
    /// The packed data must not contain placeholder (N) bases.
    pub fn from_packed_2bit(packed: &[u8], base_offset: usize) -> Self {
        let byte_start = base_offset / 4;
        let sub_offset = base_offset % 4;
        let total_bases = sub_offset + K;
        let bytes_needed = total_bases.div_ceil(4);

        // Read bytes into u128 accumulator (handles up to 64 bases = K≤61 with sub_offset≤3).
        let mut val = 0u128;
        for i in 0..bytes_needed {
            val = (val << 8) | packed[byte_start + i] as u128;
        }
        // Left-align to 128 bits, then skip sub_offset bases.
        val <<= (16 - bytes_needed) * 8 + 2 * sub_offset;
        // Extract top 2*K bits → right-align.
        let result = val >> (128 - 2 * K);

        let mut kmer = Kmer::default();
        <Kmer<K> as KmerBits>::set_word(&mut kmer.bits, 0, result as u64);
        if <Kmer<K> as KmerBits>::NUM_WORDS > 1 {
            <Kmer<K> as KmerBits>::set_word(&mut kmer.bits, 1, (result >> 64) as u64);
        }
        kmer
    }

    /// Reverse complement of this k-mer. Matches C++ byte-level algorithm.
    pub fn reverse_complement(&self) -> Self {
        let mut rc = Kmer::default();

        let data = <Kmer<K> as KmerBits>::as_bytes(&self.bits);

        // Use raw pointer to get mutable access to rc's bytes.
        let rc_ptr = &mut rc.bits as *mut <Kmer<K> as KmerBits>::Storage as *mut u8;

        let packed_byte_count = K / 4;
        for byte_idx in 0..packed_byte_count {
            unsafe {
                *rc_ptr.add(packed_byte_count - 1 - byte_idx) =
                    REVERSE_COMPLEMENT_BYTE[data[byte_idx] as usize];
            }
        }

        let rem_base_count = K % 4;
        if rem_base_count == 0 {
            return rc;
        }

        unsafe {
            *rc_ptr.add(packed_byte_count) = 0;
        }

        // Left shift by rem_base_count positions.
        rc.left_shift_by(rem_base_count);

        // Process remaining bases individually.
        let partial_byte = data[packed_byte_count];
        for i in 0..rem_base_count {
            let base = Base::from_2bit((partial_byte >> (2 * i)) & 0b11);
            let compl = base.complement();
            unsafe {
                *rc_ptr |= (compl as u8) << (2 * (rem_base_count - 1 - i));
            }
        }

        rc
    }

    /// Canonical form = min(self, self.reverse_complement()).
    pub fn canonical(&self) -> Self {
        let rc = self.reverse_complement();
        if *self <= rc {
            *self
        } else {
            rc
        }
    }

    /// Canonical form given a precomputed reverse complement.
    pub fn canonical_with_rc(&self, rev_compl: &Self) -> Self {
        if *self <= *rev_compl {
            *self
        } else {
            *rev_compl
        }
    }

    /// True if self is in forward direction relative to kmer_hat (the canonical form).
    pub fn in_forward(&self, kmer_hat: &Self) -> bool {
        self == kmer_hat
    }

    /// Extract front base (MSN = most significant nucleotide, n_{k-1}).
    pub fn front(&self) -> Base {
        let w = <Kmer<K> as KmerBits>::word(&self.bits, Self::NUM_WORDS - 1);
        Base::from_2bit(((w >> Self::MSN_SHIFT) & 0b11) as u8)
    }

    /// Extract back base (LSN = least significant nucleotide, n_0).
    pub fn back(&self) -> Base {
        let w = <Kmer<K> as KmerBits>::word(&self.bits, 0);
        Base::from_2bit((w & 0b11) as u8)
    }

    /// Roll forward: chop off front, append base at back.
    /// Also updates rev_compl accordingly.
    pub fn roll_to_next_kmer(&mut self, base: Base, rev_compl: &mut Self) {
        // Clear MSN.
        let w = <Kmer<K> as KmerBits>::word(&self.bits, Self::NUM_WORDS - 1);
        <Kmer<K> as KmerBits>::set_word(
            &mut self.bits,
            Self::NUM_WORDS - 1,
            w & Self::CLEAR_MSN_MASK,
        );
        // Left shift by 1.
        self.left_shift();
        // Insert new base at LSB.
        let w0 = <Kmer<K> as KmerBits>::word(&self.bits, 0);
        <Kmer<K> as KmerBits>::set_word(&mut self.bits, 0, w0 | base as u64);

        // Update reverse complement.
        rev_compl.right_shift();
        let rw = <Kmer<K> as KmerBits>::word(&rev_compl.bits, Self::NUM_WORDS - 1);
        <Kmer<K> as KmerBits>::set_word(
            &mut rev_compl.bits,
            Self::NUM_WORDS - 1,
            rw | ((base.complement() as u64) << Self::MSN_SHIFT),
        );
    }

    /// Roll forward from ASCII character.
    pub fn roll_to_next_kmer_char(&mut self, next_char: u8, rev_compl: &mut Self) {
        let base = Base::map_base(next_char);
        self.roll_to_next_kmer(base, rev_compl);
    }

    /// XXH3 64-bit hash matching C++ `to_u64()`.
    pub fn hash_xxh3(&self) -> u64 {
        let bytes = <Kmer<K> as KmerBits>::as_bytes(&self.bits);
        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes[..Self::NUM_BYTES], 0)
    }

    /// Get string label of the k-mer.
    pub fn string_label(&self) -> String {
        let mut copy = *self;
        let mut label = Vec::with_capacity(K);
        for _ in 0..K {
            let w = <Kmer<K> as KmerBits>::word(&copy.bits, 0);
            let base = Base::from_2bit((w & 0b11) as u8);
            label.push(base.to_char());
            copy.right_shift();
        }
        label.reverse();
        String::from_utf8(label).unwrap()
    }

    /// Left shift all words by 2 bits (one base).
    fn left_shift(&mut self) {
        let n = Self::NUM_WORDS;
        for idx in (1..n).rev() {
            let curr = <Kmer<K> as KmerBits>::word(&self.bits, idx);
            let prev = <Kmer<K> as KmerBits>::word(&self.bits, idx - 1);
            <Kmer<K> as KmerBits>::set_word(
                &mut self.bits,
                idx,
                (curr << 2) | (prev >> 62),
            );
        }
        let w0 = <Kmer<K> as KmerBits>::word(&self.bits, 0);
        <Kmer<K> as KmerBits>::set_word(&mut self.bits, 0, w0 << 2);
    }

    /// Left shift by B bases (2B bits).
    fn left_shift_by(&mut self, b: usize) {
        if b == 0 {
            return;
        }
        let num_bit_shift = 2 * b;
        let n = Self::NUM_WORDS;
        let mask_msns = ((1u64 << num_bit_shift) - 1) << (64 - num_bit_shift);
        for idx in (1..n).rev() {
            let curr = <Kmer<K> as KmerBits>::word(&self.bits, idx);
            let prev = <Kmer<K> as KmerBits>::word(&self.bits, idx - 1);
            <Kmer<K> as KmerBits>::set_word(
                &mut self.bits,
                idx,
                (curr << num_bit_shift) | ((prev & mask_msns) >> (64 - num_bit_shift)),
            );
        }
        let w0 = <Kmer<K> as KmerBits>::word(&self.bits, 0);
        <Kmer<K> as KmerBits>::set_word(&mut self.bits, 0, w0 << num_bit_shift);
    }

    /// Right shift all words by 2 bits (one base).
    fn right_shift(&mut self) {
        let n = Self::NUM_WORDS;
        for idx in 0..n - 1 {
            let curr = <Kmer<K> as KmerBits>::word(&self.bits, idx);
            let next = <Kmer<K> as KmerBits>::word(&self.bits, idx + 1);
            <Kmer<K> as KmerBits>::set_word(
                &mut self.bits,
                idx,
                (curr >> 2) | ((next & 0b11) << 62),
            );
        }
        let wn = <Kmer<K> as KmerBits>::word(&self.bits, n - 1);
        <Kmer<K> as KmerBits>::set_word(&mut self.bits, n - 1, wn >> 2);
    }

    /// Write the unitig sequence for this k-mer range to a buffer.
    pub fn write_label_to<W: std::io::Write>(
        &self,
        k: usize,
        seq: &[u8],
        start_kmer_idx: usize,
        end_kmer_idx: usize,
        dir: bool, // true = FWD
        writer: &mut W,
    ) -> std::io::Result<()> {
        let segment_len = end_kmer_idx - start_kmer_idx + k;
        if dir {
            for offset in 0..segment_len {
                writer.write_all(&[crate::dna::to_upper(seq[start_kmer_idx + offset])])?;
            }
        } else {
            for offset in 0..segment_len {
                writer.write_all(&[crate::dna::complement_char(
                    seq[end_kmer_idx + k - 1 - offset],
                )])?;
            }
        }
        Ok(())
    }
}

impl Base {
    #[inline]
    pub fn from_2bit(v: u8) -> Base {
        debug_assert!(v < 4);
        unsafe { std::mem::transmute(v) }
    }
}

/// Encode a fixed-size word from ASCII DNA characters.
/// First character occupies the most-significant bits, matching C++ cuttlefish.
fn encode_word<const N: usize>(label: &[u8]) -> u64 {
    debug_assert!(label.len() >= N);
    label.iter().take(N).fold(0u64, |acc, &b| (acc << 2) | (Base::map_base(b) as u64))
}

/// Encode a variable-size word.
fn encode_word_dyn(n: usize, label: &[u8]) -> u64 {
    debug_assert!(label.len() >= n);
    label.iter().take(n).fold(0u64, |acc, &b| (acc << 2) | (Base::map_base(b) as u64))
}

// Implementation for K=1..=32 (single u64 storage).
macro_rules! impl_kmer_bits_u64 {
    ($($k:literal),*) => {
        $(
            impl KmerBits for Kmer<$k> {
                type Storage = u64;
                const NUM_WORDS: usize = 1;

                #[inline]
                fn word(storage: &u64, _idx: usize) -> u64 {
                    *storage
                }

                #[inline]
                fn set_word(storage: &mut u64, _idx: usize, val: u64) {
                    *storage = val;
                }

                #[inline]
                fn as_bytes(storage: &u64) -> &[u8] {
                    unsafe { std::slice::from_raw_parts(storage as *const u64 as *const u8, 8) }
                }
            }
        )*
    };
}

// Implementation for K=33..=63 (u128 storage = 2 x u64 words).
macro_rules! impl_kmer_bits_u128 {
    ($($k:literal),*) => {
        $(
            impl KmerBits for Kmer<$k> {
                type Storage = u128;
                const NUM_WORDS: usize = 2;

                #[inline]
                fn word(storage: &u128, idx: usize) -> u64 {
                    if idx == 0 {
                        *storage as u64
                    } else {
                        (*storage >> 64) as u64
                    }
                }

                #[inline]
                fn set_word(storage: &mut u128, idx: usize, val: u64) {
                    if idx == 0 {
                        *storage = (*storage & (0xFFFF_FFFF_FFFF_FFFFu128 << 64)) | val as u128;
                    } else {
                        *storage = (*storage & 0xFFFF_FFFF_FFFF_FFFFu128) | ((val as u128) << 64);
                    }
                }

                #[inline]
                fn as_bytes(storage: &u128) -> &[u8] {
                    unsafe { std::slice::from_raw_parts(storage as *const u128 as *const u8, 16) }
                }
            }
        )*
    };
}

impl_kmer_bits_u64!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32
);

impl_kmer_bits_u128!(
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63
);

/// Dispatch macro: calls $func::<K>($($arg),*) for the given runtime k value.
/// Only odd values in [1, 63] are supported (de Bruijn graph convention).
#[macro_export]
macro_rules! dispatch_k {
    ($k:expr, $func:ident $(, $arg:expr)*) => {
        match $k {
            1 => $func::<1>($($arg),*),
            3 => $func::<3>($($arg),*),
            5 => $func::<5>($($arg),*),
            7 => $func::<7>($($arg),*),
            9 => $func::<9>($($arg),*),
            11 => $func::<11>($($arg),*),
            13 => $func::<13>($($arg),*),
            15 => $func::<15>($($arg),*),
            17 => $func::<17>($($arg),*),
            19 => $func::<19>($($arg),*),
            21 => $func::<21>($($arg),*),
            23 => $func::<23>($($arg),*),
            25 => $func::<25>($($arg),*),
            27 => $func::<27>($($arg),*),
            29 => $func::<29>($($arg),*),
            31 => $func::<31>($($arg),*),
            33 => $func::<33>($($arg),*),
            35 => $func::<35>($($arg),*),
            37 => $func::<37>($($arg),*),
            39 => $func::<39>($($arg),*),
            41 => $func::<41>($($arg),*),
            43 => $func::<43>($($arg),*),
            45 => $func::<45>($($arg),*),
            47 => $func::<47>($($arg),*),
            49 => $func::<49>($($arg),*),
            51 => $func::<51>($($arg),*),
            53 => $func::<53>($($arg),*),
            55 => $func::<55>($($arg),*),
            57 => $func::<57>($($arg),*),
            59 => $func::<59>($($arg),*),
            61 => $func::<61>($($arg),*),
            63 => $func::<63>($($arg),*),
            _ => panic!("k must be odd and in [1, 63], got {}", $k),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ascii_and_label() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACG"; // 31 bases
        let kmer = Kmer::<31>::from_ascii(seq, 0);
        let label = kmer.string_label();
        assert_eq!(label, "ACGTACGTACGTACGTACGTACGTACGTACG");
    }

    #[test]
    fn test_reverse_complement() {
        let seq = b"ACGT";
        let kmer = Kmer::<4>::from_ascii(seq, 0);
        let rc = kmer.reverse_complement();
        assert_eq!(rc.string_label(), "ACGT"); // ACGT is its own reverse complement
    }

    #[test]
    fn test_reverse_complement_asymmetric() {
        let seq = b"AAAC";
        let kmer = Kmer::<4>::from_ascii(seq, 0);
        let rc = kmer.reverse_complement();
        assert_eq!(rc.string_label(), "GTTT");
    }

    #[test]
    fn test_canonical() {
        let seq = b"AAAC";
        let kmer = Kmer::<4>::from_ascii(seq, 0);
        let can = kmer.canonical();
        // AAAC < GTTT so canonical should be AAAC
        assert_eq!(can.string_label(), "AAAC");
    }

    #[test]
    fn test_front_back() {
        let seq = b"ACGT";
        let kmer = Kmer::<4>::from_ascii(seq, 0);
        assert_eq!(kmer.front(), Base::A);
        assert_eq!(kmer.back(), Base::T);
    }

    #[test]
    fn test_roll() {
        let seq = b"ACGTG";
        let mut kmer = Kmer::<4>::from_ascii(seq, 0);
        let mut rc = kmer.reverse_complement();
        kmer.roll_to_next_kmer(Base::G, &mut rc);
        assert_eq!(kmer.string_label(), "CGTG");
        assert_eq!(rc.string_label(), "CACG");
    }

    #[test]
    fn test_hash_consistency() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACG";
        let kmer = Kmer::<31>::from_ascii(seq, 0);
        let h1 = kmer.hash_xxh3();
        let h2 = kmer.hash_xxh3();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_ordering() {
        let a = Kmer::<4>::from_ascii(b"AAAA", 0);
        let b = Kmer::<4>::from_ascii(b"AAAC", 0);
        let c = Kmer::<4>::from_ascii(b"TTTT", 0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_k33() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACG"; // 35 bases
        let kmer = Kmer::<33>::from_ascii(seq, 0);
        let label = kmer.string_label();
        assert_eq!(&label, "ACGTACGTACGTACGTACGTACGTACGTACGTA");

        let rc = kmer.reverse_complement();
        let rc_label = rc.string_label();
        // Verify round-trip
        let rc_rc = rc.reverse_complement();
        assert_eq!(rc_rc, kmer);
        let _ = rc_label;
    }
}
