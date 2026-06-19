/// DNA base encoding matching cuttlefish C++: A=0, C=1, G=2, T=3.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Base {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
}

impl Base {
    #[inline]
    pub fn complement(self) -> Base {
        COMPLEMENTED_BASE[self as usize]
    }

    #[inline]
    pub fn from_ascii(b: u8) -> Option<Base> {
        let mapped = MAPPED_BASE[b as usize];
        if mapped <= 3 {
            Some(unsafe { std::mem::transmute::<u8, Base>(mapped) })
        } else {
            None
        }
    }

    /// Map an ASCII byte to a Base, treating non-ACGT as A (caller's responsibility
    /// to check is_placeholder first).
    #[inline]
    pub fn map_base(b: u8) -> Base {
        unsafe { std::mem::transmute(MAPPED_BASE[b as usize]) }
    }

    #[inline]
    pub fn to_char(self) -> u8 {
        MAPPED_CHAR[self as usize]
    }
}

/// Returns true for anything not in ACGTacgt.
#[inline]
pub fn is_placeholder(b: u8) -> bool {
    IS_PLACEHOLDER[b as usize]
}

/// True iff `seq` contains any base outside `ACGTacgt`.
///
/// Faster than scanning with [`is_placeholder`] (a 256-entry table, which lowers
/// to a per-byte gather the compiler can't vectorize): each byte is upper-cased
/// with `& 0xDF` — which maps exactly the 8 bytes `ACGTacgt` (and no others) onto
/// `{A,C,G,T}` — then tested against 4 constants. That's branchless integer work
/// the compiler auto-vectorizes into wide SIMD compares. Use this for the boolean
/// "does it contain N?" check; fall back to [`is_placeholder`] only to locate the
/// offending position on the (rare) error path.
#[inline]
pub fn contains_non_acgt(seq: &[u8]) -> bool {
    seq.iter()
        .any(|&b| !matches!(b & 0xDF, b'A' | b'C' | b'G' | b'T'))
}

/// Invoke `f(seg_start, segment)` for each maximal run of non-placeholder
/// (ACGT) bases of length `>= k`. Cuttlefish handles ambiguous (`N`) bases by
/// splitting a sequence on them — no k-mer spans a placeholder — exactly as the
/// classification and super-k-mer packing phases already do. Runs shorter than `k`
/// carry no k-mers and are skipped. `seg_start` is the offset of the segment in
/// `seq` (callers that only need k-mer *content*, e.g. minimizer counting and
/// super-k-mer routing, can ignore it).
#[inline]
pub fn for_each_acgt_segment(seq: &[u8], k: usize, mut f: impl FnMut(usize, &[u8])) {
    let n = seq.len();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < n {
        if is_placeholder(seq[i]) {
            if i - start >= k {
                f(start, &seq[start..i]);
            }
            start = i + 1;
        }
        i += 1;
    }
    if n - start >= k {
        f(start, &seq[start..n]);
    }
}

/// Upper-case an ASCII DNA character matching C++ DNA_Utility::upper.
#[inline]
pub fn to_upper(b: u8) -> u8 {
    if b <= b'T' { b } else { b - (b'a' - b'A') }
}

/// Complement of an ASCII base character. Returns 'N' for non-ACGT.
#[inline]
pub fn complement_char(b: u8) -> u8 {
    COMPLEMENTED_CHAR[b as usize]
}

// Mapped DNA::Base for the ASCII characters in the range [0, 255].
// 4 = N (placeholder). Matches C++ DNA_Utility::MAPPED_BASE exactly.
const MAPPED_BASE: [u8; 256] = {
    let mut table = [4u8; 256];
    table[b'A' as usize] = 0;
    table[b'a' as usize] = 0;
    table[b'C' as usize] = 1;
    table[b'c' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'g' as usize] = 2;
    table[b'T' as usize] = 3;
    table[b't' as usize] = 3;
    table
};

const COMPLEMENTED_BASE: [Base; 4] = [Base::T, Base::G, Base::C, Base::A];

const MAPPED_CHAR: [u8; 4] = [b'A', b'C', b'G', b'T'];

const IS_PLACEHOLDER: [bool; 256] = {
    let mut table = [true; 256];
    table[b'A' as usize] = false;
    table[b'a' as usize] = false;
    table[b'C' as usize] = false;
    table[b'c' as usize] = false;
    table[b'G' as usize] = false;
    table[b'g' as usize] = false;
    table[b'T' as usize] = false;
    table[b't' as usize] = false;
    table
};

const COMPLEMENTED_CHAR: [u8; 256] = {
    let mut table = [b'N'; 256];
    table[b'A' as usize] = b'T';
    table[b'a' as usize] = b'T';
    table[b'C' as usize] = b'G';
    table[b'c' as usize] = b'G';
    table[b'G' as usize] = b'C';
    table[b'g' as usize] = b'C';
    table[b'T' as usize] = b'A';
    table[b't' as usize] = b'A';
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complement() {
        assert_eq!(Base::A.complement(), Base::T);
        assert_eq!(Base::C.complement(), Base::G);
        assert_eq!(Base::G.complement(), Base::C);
        assert_eq!(Base::T.complement(), Base::A);
    }

    #[test]
    fn test_from_ascii() {
        assert_eq!(Base::from_ascii(b'A'), Some(Base::A));
        assert_eq!(Base::from_ascii(b'a'), Some(Base::A));
        assert_eq!(Base::from_ascii(b'N'), None);
    }

    #[test]
    fn test_is_placeholder() {
        assert!(!is_placeholder(b'A'));
        assert!(!is_placeholder(b'a'));
        assert!(is_placeholder(b'N'));
        assert!(is_placeholder(b'n'));
        assert!(is_placeholder(b'X'));
    }
}
