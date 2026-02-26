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
            Some(unsafe { std::mem::transmute(mapped) })
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

/// Upper-case an ASCII DNA character matching C++ DNA_Utility::upper.
#[inline]
pub fn to_upper(b: u8) -> u8 {
    if b <= b'T' {
        b
    } else {
        b - (b'a' - b'A')
    }
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
