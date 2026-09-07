//! The worlds: one-line rules that move the transport exponent across its whole range.
//!
//! Part IV 3. Each substrate assigns one of two coin operators to every lattice site — the
//! *static* (position-dependent) case of Lo Gullo, Di Molfetta, Sanchez-Palencia and
//! Wilkowski, *Dynamics and energy spectra of aperiodic discrete-time quantum walks*,
//! Phys. Rev. E 96, 012111 (2017), arXiv:1611.04427. A site-dependent coin is what a
//! *world* means; a step-dependent one would be a world that changes under you.
//!
//! The three aperiodic words are distinguished by the nature of their spectrum, and that is
//! what sets the transport:
//!
//! | word | spectrum | behaviour reported in the paper |
//! |---|---|---|
//! | Fibonacci | pure point | diffusive, no signature of localization |
//! | Thue-Morse | singular continuous | a localized *and* a spreading component |
//! | Rudin-Shapiro | absolutely continuous | the most strongly localized |
//!
//! Part IV 3's table attaches those spectra the other way round — it calls Fibonacci
//! singular-continuous and Rudin-Shapiro discrete. It is the Rudin-Shapiro word whose
//! spectrum is absolutely continuous, which is precisely *why* it behaves like disorder,
//! and Thue-Morse that is singular continuous. The behaviour in that table is roughly
//! right; the labels are swapped, and the panel's caption depends on them.

/// A binary word on the non-negative integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// Every site the same: a clean lattice.
    Periodic,
    /// `ABABAB...`. Equivalent to a two-period walk in time, and still ballistic.
    TwoPeriodic,
    /// `A -> AB, B -> A`. Pure point spectrum.
    Fibonacci,
    /// `A -> AB, B -> BA`. Singular continuous spectrum.
    ThueMorse,
    /// Absolutely continuous spectrum; the closest deterministic word to noise.
    RudinShapiro,
    /// Not a word at all: an independent seeded draw per site, for Anderson localization.
    Disorder,
}

pub const ALL_WORDS: [Word; 6] = [
    Word::Periodic,
    Word::TwoPeriodic,
    Word::Fibonacci,
    Word::ThueMorse,
    Word::RudinShapiro,
    Word::Disorder,
];

impl Word {
    pub fn name(self) -> &'static str {
        match self {
            Word::Periodic => "periodic",
            Word::TwoPeriodic => "two-periodic",
            Word::Fibonacci => "Fibonacci",
            Word::ThueMorse => "Thue-Morse",
            Word::RudinShapiro => "Rudin-Shapiro",
            Word::Disorder => "static disorder",
        }
    }

    /// The regime the literature puts this substrate in, for the readout's caption.
    pub fn regime(self) -> &'static str {
        match self {
            Word::Periodic | Word::TwoPeriodic => "ballistic",
            Word::Fibonacci => "diffusive, no localization",
            Word::ThueMorse => "mixed: localized and spreading components",
            Word::RudinShapiro => "strongly localized",
            Word::Disorder => "Anderson localization",
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Word::Periodic => 0,
            Word::TwoPeriodic => 1,
            Word::Fibonacci => 2,
            Word::ThueMorse => 3,
            Word::RudinShapiro => 4,
            Word::Disorder => 5,
        }
    }

    pub fn from_code(c: u32) -> Word {
        ALL_WORDS[(c as usize).min(ALL_WORDS.len() - 1)]
    }
}

/// The Fibonacci word, built by substitution rather than by a floor-function identity, so
/// that it is the definition and not a claim about the definition.
fn fibonacci_word(len: usize) -> Vec<u8> {
    let mut w = vec![0u8];
    while w.len() < len {
        let mut next = Vec::with_capacity(w.len() * 2);
        for &c in &w {
            if c == 0 {
                next.push(0);
                next.push(1);
            } else {
                next.push(0);
            }
        }
        if next.len() == w.len() {
            break;
        }
        w = next;
    }
    w.truncate(len);
    w
}

/// `t_n` is the parity of the number of ones in `n`.
#[inline]
fn thue_morse(n: u64) -> u8 {
    (n.count_ones() & 1) as u8
}

/// `r_n` is the parity of the number of **overlapping** `11` pairs in the binary expansion
/// of `n`, which `n & (n >> 1)` counts in one instruction.
#[inline]
fn rudin_shapiro(n: u64) -> u8 {
    ((n & (n >> 1)).count_ones() & 1) as u8
}

/// A substrate placed on the integer line.
///
/// The word is indexed from an offset rather than mirrored about the origin: mirroring would
/// put a reflection symmetry at the walker's starting point, and then the symmetry rather
/// than the substrate would be setting the dynamics.
#[derive(Clone, Debug)]
pub struct Substrate {
    word: Word,
    offset: i64,
    fib: Vec<u8>,
    seed: u64,
}

impl Substrate {
    pub fn new(word: Word, span: usize, seed: u64) -> Self {
        let offset = span as i64 + 8;
        let fib = if word == Word::Fibonacci {
            fibonacci_word(2 * span + 32)
        } else {
            Vec::new()
        };
        Substrate {
            word,
            offset,
            fib,
            seed,
        }
    }

    pub fn word(&self) -> Word {
        self.word
    }

    /// The letter at index `n` of the underlying word.
    fn at_index(&self, n: u64) -> u8 {
        match self.word {
            Word::Periodic => 0,
            Word::TwoPeriodic => (n & 1) as u8,
            Word::Fibonacci => *self.fib.get(n as usize).unwrap_or(&0),
            Word::ThueMorse => thue_morse(n),
            Word::RudinShapiro => rudin_shapiro(n),
            Word::Disorder => {
                // A seeded hash, so a world is reproducible from its seed alone.
                let mut h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ self.seed;
                h ^= h >> 33;
                h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                ((h >> 60) & 1) as u8
            }
        }
    }

    /// Which of the two coins sits at site `x`: 0 for `A`, 1 for `B`.
    pub fn letter(&self, x: i64) -> u8 {
        self.at_index((x + self.offset).max(0) as u64)
    }

    /// The coin at site `x`.
    ///
    /// The binary words select one of the two given coins. Static disorder does not: a word
    /// over two letters is weak disorder, and a single realisation of it has a localisation
    /// length long enough that the fitted exponent swings by 0.2 between seeds. Anderson
    /// localization is a statement about a *continuum* of local parameters, so the disorder
    /// world draws a continuous coin angle per site instead.
    pub fn coin(&self, x: i64, a: &crate::coin::Coin, b: &crate::coin::Coin) -> crate::coin::Coin {
        if self.word != Word::Disorder {
            return if self.letter(x) == 0 { *a } else { *b };
        }
        let n = (x + self.offset).max(0) as u64;
        let mut h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ self.seed;
        h ^= h >> 29;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 32;
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        crate::coin::Coin::theta(u * std::f64::consts::PI)
    }

    /// The first `len` letters of the word itself, as `"ABAAB..."`. The tests pin each word
    /// against the sequence printed in the paper.
    pub fn render(&self, len: usize) -> String {
        (0..len as u64)
            .map(|n| if self.at_index(n) == 0 { 'A' } else { 'B' })
            .collect()
    }
}
