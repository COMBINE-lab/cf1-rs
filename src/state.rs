use crate::dna::Base;

/// DFA state class of a vertex.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum StateClass {
    SingleInSingleOut = 0,
    MultiInSingleOut = 1,
    SingleInMultiOut = 2,
    MultiInMultiOut = 3,
}

/// Decoded vertex information.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub state_class: StateClass,
    pub front: Base,
    pub back: Base,
    pub visited: bool,
    pub outputted: bool,
}

impl Vertex {
    /// Unvisited vertex.
    pub fn unvisited() -> Self {
        Vertex {
            state_class: StateClass::SingleInSingleOut,
            front: Base::A,
            back: Base::A,
            visited: false,
            outputted: false,
        }
    }

    /// SISO vertex with front and back bases.
    pub fn siso(front: Base, back: Base) -> Self {
        Vertex {
            state_class: StateClass::SingleInSingleOut,
            front,
            back,
            visited: true,
            outputted: false,
        }
    }

    /// MISO vertex with back base.
    pub fn miso(back: Base) -> Self {
        Vertex {
            state_class: StateClass::MultiInSingleOut,
            front: Base::A,
            back,
            visited: true,
            outputted: false,
        }
    }

    /// SIMO vertex with front base.
    pub fn simo(front: Base) -> Self {
        Vertex {
            state_class: StateClass::SingleInMultiOut,
            front,
            back: Base::A,
            visited: true,
            outputted: false,
        }
    }

    /// MIMO vertex.
    pub fn mimo() -> Self {
        Vertex {
            state_class: StateClass::MultiInMultiOut,
            front: Base::A,
            back: Base::A,
            visited: true,
            outputted: false,
        }
    }
}

/// 5-bit DFA state code. Matches C++ State exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct State(pub u8);

impl State {
    pub const UNVISITED: State = State(0);

    pub fn code(self) -> u8 {
        self.0
    }

    /// Encode a Vertex into a State. Matches C++ State::State(const Vertex&).
    pub fn from_vertex(v: &Vertex) -> State {
        if !v.visited {
            return State::UNVISITED;
        }
        if !v.outputted {
            match v.state_class {
                StateClass::SingleInSingleOut => {
                    State(0b10000 | ((v.front as u8) << 2) | (v.back as u8))
                }
                StateClass::MultiInSingleOut => State(0b00100 | (v.back as u8)),
                StateClass::SingleInMultiOut => State(0b01000 | (v.front as u8)),
                StateClass::MultiInMultiOut => State(0b00011),
            }
        } else {
            match v.state_class {
                StateClass::SingleInSingleOut => State(0b01100),
                StateClass::MultiInSingleOut => State(0b01101),
                StateClass::SingleInMultiOut => State(0b01110),
                StateClass::MultiInMultiOut => State(0b01111),
            }
        }
    }

    /// Decode state to Vertex. Matches C++ State::decode().
    pub fn decode(self) -> Vertex {
        DECODE_TABLE[self.0 as usize]
    }

    /// Is this state visited (non-zero)?
    #[inline]
    pub fn is_visited(self) -> bool {
        self.0 != 0
    }

    /// Is this a dead-end (visited + MIMO)?
    #[inline]
    pub fn is_dead_end(self) -> bool {
        self.is_visited() && self.state_class() == StateClass::MultiInMultiOut
    }

    /// Get the state class.
    pub fn state_class(self) -> StateClass {
        STATE_CLASS_TABLE[self.0 as usize]
    }

    /// Is this state outputted?
    pub fn is_outputted(self) -> bool {
        matches!(self.0, 12..=15)
    }

    /// Return the outputted version of this state.
    pub fn outputted(self) -> State {
        OUTPUTTED_TABLE[self.0 as usize]
    }
}

const fn build_decode_table() -> [Vertex; 32] {
    let mut table = [Vertex {
        state_class: StateClass::SingleInSingleOut,
        front: Base::A,
        back: Base::A,
        visited: false,
        outputted: false,
    }; 32];

    // 0 = unvisited (already set)
    // 1, 2 = invalid (leave as unvisited; shouldn't be accessed)

    // 3 = MIMO not outputted
    table[3] = Vertex {
        state_class: StateClass::MultiInMultiOut,
        front: Base::A,
        back: Base::A,
        visited: true,
        outputted: false,
    };

    // 4-7 = MISO back=A/C/G/T
    let bases = [Base::A, Base::C, Base::G, Base::T];
    let mut i = 0;
    while i < 4 {
        table[4 + i] = Vertex {
            state_class: StateClass::MultiInSingleOut,
            front: Base::A,
            back: bases[i],
            visited: true,
            outputted: false,
        };
        i += 1;
    }

    // 8-11 = SIMO front=A/C/G/T
    i = 0;
    while i < 4 {
        table[8 + i] = Vertex {
            state_class: StateClass::SingleInMultiOut,
            front: bases[i],
            back: Base::A,
            visited: true,
            outputted: false,
        };
        i += 1;
    }

    // 12 = SISO outputted
    table[12] = Vertex {
        state_class: StateClass::SingleInSingleOut,
        front: Base::A,
        back: Base::A,
        visited: true,
        outputted: true,
    };
    // 13 = MISO outputted
    table[13] = Vertex {
        state_class: StateClass::MultiInSingleOut,
        front: Base::A,
        back: Base::A,
        visited: true,
        outputted: true,
    };
    // 14 = SIMO outputted
    table[14] = Vertex {
        state_class: StateClass::SingleInMultiOut,
        front: Base::A,
        back: Base::A,
        visited: true,
        outputted: true,
    };
    // 15 = MIMO outputted
    table[15] = Vertex {
        state_class: StateClass::MultiInMultiOut,
        front: Base::A,
        back: Base::A,
        visited: true,
        outputted: true,
    };

    // 16-31 = SISO front(bits 3-2) + back(bits 1-0)
    let mut code: usize = 16;
    while code < 32 {
        let front_idx = (code >> 2) & 3;
        let back_idx = code & 3;
        table[code] = Vertex {
            state_class: StateClass::SingleInSingleOut,
            front: bases[front_idx],
            back: bases[back_idx],
            visited: true,
            outputted: false,
        };
        code += 1;
    }

    table
}

const DECODE_TABLE: [Vertex; 32] = build_decode_table();

const fn build_state_class_table() -> [StateClass; 32] {
    let mut table = [StateClass::SingleInSingleOut; 32];
    // 0 = unvisited (class is meaningless but we set SISO)
    // 1, 2 = invalid
    table[3] = StateClass::MultiInMultiOut;
    table[4] = StateClass::MultiInSingleOut;
    table[5] = StateClass::MultiInSingleOut;
    table[6] = StateClass::MultiInSingleOut;
    table[7] = StateClass::MultiInSingleOut;
    table[8] = StateClass::SingleInMultiOut;
    table[9] = StateClass::SingleInMultiOut;
    table[10] = StateClass::SingleInMultiOut;
    table[11] = StateClass::SingleInMultiOut;
    table[12] = StateClass::SingleInSingleOut;
    table[13] = StateClass::MultiInSingleOut;
    table[14] = StateClass::SingleInMultiOut;
    table[15] = StateClass::MultiInMultiOut;
    // 16-31 = SISO (already set)
    table
}

const STATE_CLASS_TABLE: [StateClass; 32] = build_state_class_table();

const fn build_outputted_table() -> [State; 32] {
    let mut table = [State(0); 32];
    // 0 = unvisited -> invalid (shouldn't happen)
    // 1, 2 = invalid
    table[3] = State(0b01111); // MIMO -> MIMO outputted
    table[4] = State(0b01101); // MISO -> MISO outputted
    table[5] = State(0b01101);
    table[6] = State(0b01101);
    table[7] = State(0b01101);
    table[8] = State(0b01110); // SIMO -> SIMO outputted
    table[9] = State(0b01110);
    table[10] = State(0b01110);
    table[11] = State(0b01110);
    table[12] = State(0b01100); // Already outputted
    table[13] = State(0b01101);
    table[14] = State(0b01110);
    table[15] = State(0b01111);
    // 16-31 = SISO -> SISO outputted
    let mut i: u8 = 16;
    while i < 32 {
        table[i as usize] = State(0b01100);
        i += 1;
    }
    table
}

const OUTPUTTED_TABLE: [State; 32] = build_outputted_table();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_round_trip() {
        // SISO with front=C, back=G
        let v = Vertex::siso(Base::C, Base::G);
        let s = State::from_vertex(&v);
        assert_eq!(s.code(), 0b10110); // 22
        let decoded = s.decode();
        assert_eq!(decoded.state_class, StateClass::SingleInSingleOut);
        assert_eq!(decoded.front, Base::C);
        assert_eq!(decoded.back, Base::G);
        assert!(decoded.visited);
        assert!(!decoded.outputted);
    }

    #[test]
    fn test_mimo_encoding() {
        let v = Vertex::mimo();
        let s = State::from_vertex(&v);
        assert_eq!(s.code(), 3);
        assert!(s.is_dead_end());
    }

    #[test]
    fn test_outputted() {
        let s = State(0b10110); // SISO(C, G)
        let o = s.outputted();
        assert_eq!(o.code(), 0b01100); // SISO outputted
        assert!(o.is_outputted());
    }

    #[test]
    fn test_miso_encoding() {
        let v = Vertex::miso(Base::T);
        let s = State::from_vertex(&v);
        assert_eq!(s.code(), 0b00111); // 7
        assert_eq!(s.state_class(), StateClass::MultiInSingleOut);
    }

    #[test]
    fn test_simo_encoding() {
        let v = Vertex::simo(Base::G);
        let s = State::from_vertex(&v);
        assert_eq!(s.code(), 0b01010); // 10
        assert_eq!(s.state_class(), StateClass::SingleInMultiOut);
    }
}
