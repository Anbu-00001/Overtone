//! Pauli strings as symplectic bitsets.
//!
//! Part III 3: the whole opportunity is that the Lie closure of Pauli generators needs no
//! matrices. A string is two bitsets `(x, z)`, two strings anticommute iff a symplectic
//! form is odd, and their product is an XOR. Everything below is that, plus honest phase
//! bookkeeping, because the phases are what the g-sim rotations' signs are made of.
//!
//! # Convention
//!
//! The Hermitian operator a `(x, z)` pair denotes is
//!
//! ```text
//! P(x, z) = i^|x & z| * X^x Z^z
//! ```
//!
//! which is the unique power of `i` making it Hermitian: on one qubit `x = z = 1` gives
//! `i * X * Z = i * (-i Y) = Y`. Qubit `q` is bit `q`, matching the little-endian indexing
//! the simulator uses everywhere else.

use overtone_sim::Pauli;

/// Qubits addressable by one string. Two `u128` planes, so 128 — comfortably past the
/// 100-qubit g-sim claim of Part III 4, and `Copy`, which the closure's hashing wants.
pub const MAX_QUBITS: usize = 128;

/// A Pauli string, phase-free.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct PauliString {
    /// Bit `q` set iff the string has an `X` or a `Y` on qubit `q`.
    pub x: u128,
    /// Bit `q` set iff the string has a `Z` or a `Y` on qubit `q`.
    pub z: u128,
}

impl PauliString {
    pub const IDENTITY: PauliString = PauliString { x: 0, z: 0 };

    pub fn new(x: u128, z: u128) -> Self {
        PauliString { x, z }
    }

    /// A single-qubit factor on qubit `q`, identity elsewhere.
    pub fn single(q: usize, p: Pauli) -> Self {
        assert!(q < MAX_QUBITS, "qubit {q} beyond MAX_QUBITS");
        let bit = 1u128 << q;
        match p {
            Pauli::I => PauliString::IDENTITY,
            Pauli::X => PauliString { x: bit, z: 0 },
            Pauli::Y => PauliString { x: bit, z: bit },
            Pauli::Z => PauliString { x: 0, z: bit },
        }
    }

    /// The product of single-qubit factors. Repeated qubits are multiplied, so
    /// `from_factors([(0, X), (0, Z)])` is `XZ` on qubit zero up to phase — pass distinct
    /// qubits if that is not what you meant.
    pub fn from_factors(factors: &[(usize, Pauli)]) -> Self {
        let mut s = PauliString::IDENTITY;
        for &(q, p) in factors {
            let t = PauliString::single(q, p);
            s.x ^= t.x;
            s.z ^= t.z;
        }
        s
    }

    /// Parse `"XYZI"`, leftmost character being qubit zero.
    pub fn parse(s: &str) -> Self {
        let mut out = PauliString::IDENTITY;
        for (q, c) in s.chars().enumerate() {
            let p = match c {
                'I' | 'i' | '.' => Pauli::I,
                'X' | 'x' => Pauli::X,
                'Y' | 'y' => Pauli::Y,
                'Z' | 'z' => Pauli::Z,
                other => panic!("not a Pauli character: {other}"),
            };
            out.x ^= PauliString::single(q, p).x;
            out.z ^= PauliString::single(q, p).z;
        }
        out
    }

    /// The factor on qubit `q`.
    pub fn at(&self, q: usize) -> Pauli {
        let bit = 1u128 << q;
        match (self.x & bit != 0, self.z & bit != 0) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
        }
    }

    /// `"XYZI"` over `n` qubits.
    pub fn render(&self, n: usize) -> String {
        (0..n)
            .map(|q| match self.at(q) {
                Pauli::I => 'I',
                Pauli::X => 'X',
                Pauli::Y => 'Y',
                Pauli::Z => 'Z',
            })
            .collect()
    }

    /// As `(qubit, Pauli)` pairs, identities dropped — the simulator's observable form.
    pub fn factors(&self, n: usize) -> Vec<(usize, Pauli)> {
        (0..n)
            .filter_map(|q| match self.at(q) {
                Pauli::I => None,
                p => Some((q, p)),
            })
            .collect()
    }

    pub fn is_identity(&self) -> bool {
        self.x == 0 && self.z == 0
    }

    /// Number of non-identity factors.
    pub fn weight(&self) -> u32 {
        (self.x | self.z).count_ones()
    }

    /// Highest qubit index carrying a non-identity factor, plus one.
    pub fn span(&self) -> usize {
        let bits = self.x | self.z;
        if bits == 0 {
            0
        } else {
            MAX_QUBITS - bits.leading_zeros() as usize
        }
    }

    /// The symplectic form. `true` when the two strings anticommute.
    ///
    /// This is the one bit the whole closure turns on: commuting strings contribute
    /// nothing, anticommuting ones contribute exactly one new string.
    #[inline]
    pub fn anticommutes(&self, other: &PauliString) -> bool {
        ((self.x & other.z).count_ones() + (other.x & self.z).count_ones()) & 1 == 1
    }

    #[inline]
    pub fn commutes(&self, other: &PauliString) -> bool {
        !self.anticommutes(other)
    }

    /// `self * other = i^e * product`, returning `(product, e mod 4)`.
    ///
    /// Derivation, with `|a|` the popcount: `X^a Z^b X^c Z^d = (-1)^|b & c| X^(a^c) Z^(b^d)`,
    /// and each Hermitian string carries its own `i^|x & z|`, so the exponents subtract.
    #[inline]
    pub fn mul(&self, other: &PauliString) -> (PauliString, u8) {
        let prod = PauliString {
            x: self.x ^ other.x,
            z: self.z ^ other.z,
        };
        let e = (self.x & self.z).count_ones() as i64 + (other.x & other.z).count_ones() as i64
            - (prod.x & prod.z).count_ones() as i64
            + 2 * (self.z & other.x).count_ones() as i64;
        (prod, e.rem_euclid(4) as u8)
    }

    /// The commutator, as `[self, other] = 2i * sign * string`.
    ///
    /// `None` when they commute, which is the overwhelmingly common case and the reason
    /// the closure is cheap. The `2i` is factored out because the result of a Pauli
    /// commutator is always anti-Hermitian and always a single string: there is no sum to
    /// accumulate and no coefficient to track beyond one sign bit.
    #[inline]
    pub fn commutator(&self, other: &PauliString) -> Option<(PauliString, i8)> {
        if self.commutes(other) {
            return None;
        }
        let (prod, e) = self.mul(other);
        // e is odd here: an anticommuting product is anti-Hermitian, so i^e is +-i.
        debug_assert!(e & 1 == 1, "anticommuting product had even phase {e}");
        let sign = if (e - 1) / 2 == 0 { 1 } else { -1 };
        Some((prod, sign))
    }

    /// `<0...0| P |0...0>`: one for a Z-type string, zero otherwise.
    #[inline]
    pub fn expectation_in_zero_state(&self) -> f64 {
        if self.x == 0 {
            1.0
        } else {
            0.0
        }
    }
}
