//! CPython-exact `random.Random` (MT19937) and CPython-exact `round()`.
//!
//! This module is a line-for-line transliteration of the CPython 3.12 sources:
//!
//! * `Modules/_randommodule.c`  — `init_genrand`, `init_by_array`, `genrand_uint32`,
//!   `random_seed`, `_random_Random_random_impl`, `_random_Random_getrandbits_impl`
//! * `Lib/random.py`            — `Random._randbelow_with_getrandbits`, `Random.choice`
//! * `Objects/floatobject.c`    — `float___round___impl` (the no-`ndigits` branch)
//!
//! Nothing in here may be "improved". Every expression is ordered the way CPython
//! orders it, because the whole point is bit-exactness with the Kaggle environment.
//!
//! Why the environment needs this:
//!
//! ```text
//! _end_of_day:  rng = random.Random((seed * 1_000_003) ^ day)
//!               player 0 _spawn_weeds  -> rng.random() per currently-None tile
//!               player 1 _spawn_weeds  -> same rng, stream continues
//!               maybe    rng.choice(sorted(SHOPS))   # 8 shops -> _randbelow(8)
//! market_price: max(PRICE_FLOOR, int(round(price)))  # round() is half-to-EVEN
//! ```

// ---------------------------------------------------------------------------
// MT19937 constants (CPython Modules/_randommodule.c)
// ---------------------------------------------------------------------------

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `static const uint32_t mag01[2] = {0x0U, MATRIX_A};`
const MAG01: [u32; 2] = [0, MATRIX_A];

/// CPython's `random.Random`, seeded exactly the way `random.Random(int)` seeds.
///
/// Cheap to clone (2.5 KiB of state, no allocation).
#[derive(Clone)]
pub struct PyRandom {
    /// `RandomObject.state`
    mt: [u32; N],
    /// `RandomObject.index`
    index: usize,
}

impl PyRandom {
    // -----------------------------------------------------------------------
    // Seeding
    // -----------------------------------------------------------------------

    /// `random.Random(seed)` for a Python `int` seed.
    ///
    /// CPython uses `abs(seed)` (`PyNumber_Absolute`), so `-7` and `7` seed
    /// identically. `i64::MIN` is handled via `unsigned_abs`.
    pub fn new(seed: i64) -> Self {
        Self::from_u64(seed.unsigned_abs())
    }

    /// `random.Random(seed)` for a non-negative seed that fits in 64 bits.
    pub fn from_u64(seed: u64) -> Self {
        // random_seed(): bits = _PyLong_NumBits(n); keyused = bits == 0 ? 1 : (bits - 1) / 32 + 1
        let bits = 64 - seed.leading_zeros();
        let keyused = if bits == 0 {
            1
        } else {
            ((bits - 1) / 32 + 1) as usize
        };
        // _PyLong_AsByteArray(..., PY_LITTLE_ENDIAN, unsigned): 32-bit words, LSW first.
        let words = [seed as u32, (seed >> 32) as u32];
        Self::from_words(&words[..keyused])
    }

    /// `random.Random(seed)` for a seed of up to 127 bits (`abs` is taken first).
    ///
    /// Needed because `(seed * 1_000_003) ^ day` is computed with Python's
    /// arbitrary-precision ints and can exceed 64 bits for large configured seeds.
    pub fn from_i128(seed: i128) -> Self {
        Self::from_u128(seed.unsigned_abs())
    }

    /// `random.Random(seed)` for a non-negative seed that fits in 128 bits.
    pub fn from_u128(seed: u128) -> Self {
        let bits = 128 - seed.leading_zeros();
        let keyused = if bits == 0 {
            1
        } else {
            ((bits - 1) / 32 + 1) as usize
        };
        let words = [
            seed as u32,
            (seed >> 32) as u32,
            (seed >> 64) as u32,
            (seed >> 96) as u32,
        ];
        Self::from_words(&words[..keyused])
    }

    /// The environment's per-day RNG: `random.Random((seed * 1_000_003) ^ day)`.
    ///
    /// Computed in `i128` so it matches Python's arbitrary-precision arithmetic for
    /// every `i64` seed (`i64::MAX * 1_000_003` needs 84 bits). Python's `^` on
    /// non-negative operands is plain bitwise xor, and on negative operands it is
    /// infinite-precision two's complement — both of which `i128` xor reproduces
    /// for any value representable in 128 bits.
    pub fn for_day(seed: i64, day: i64) -> Self {
        let key = (seed as i128 * 1_000_003i128) ^ (day as i128);
        Self::from_i128(key)
    }

    /// `init_by_array(self, key, key_length)` with `key` in CPython word order
    /// (little-endian 32-bit words of `abs(seed)`, least-significant word first).
    ///
    /// An empty slice is treated as `[0]`, matching `keyused = bits == 0 ? 1 : ...`.
    pub fn from_words(key: &[u32]) -> Self {
        let mut r = PyRandom {
            mt: [0; N],
            index: N,
        };
        let zero = [0u32];
        let key = if key.is_empty() { &zero[..] } else { key };
        r.init_by_array(key);
        r
    }

    /// `init_genrand(self, s)`
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for mti in 1..N {
            let prev = self.mt[mti - 1];
            self.mt[mti] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = N;
    }

    /// `init_by_array(self, init_key, key_length)`
    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let key_length = key.len();
        debug_assert!(key_length > 0);

        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k: usize = if N > key_length { N } else { key_length };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32); // non linear
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        let mut k: usize = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32); // non linear
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000; // MSB is 1; assuring non-zero initial array

        // NOTE: init_by_array does *not* touch self.index; it stays at N (set by
        // init_genrand), so the first genrand_uint32 call twists before drawing.
    }

    // -----------------------------------------------------------------------
    // Core generator
    // -----------------------------------------------------------------------

    /// `genrand_uint32(self)` — one raw 32-bit draw, tempered.
    #[inline]
    pub fn genrand_uint32(&mut self) -> u32 {
        if self.index >= N {
            // generate N words at one time
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                // kk + (M - N) in C, with the sum taken in signed arithmetic
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];

            self.index = 0;
        }

        let mut y = self.mt[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    // -----------------------------------------------------------------------
    // Public random.Random surface
    // -----------------------------------------------------------------------

    /// `random.random()` — a float in [0, 1) built from two 32-bit draws.
    ///
    /// `uint32_t a = genrand_uint32(self) >> 5, b = genrand_uint32(self) >> 6;`
    /// `return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);`
    ///
    /// Every operation here is exact in IEEE-754 double (`a < 2^27`, `b < 2^26`,
    /// `a * 2^26 + b < 2^53`, and the final factor is a power of two), so there is
    /// no rounding freedom and no fast-math hazard.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64;
        let b = (self.genrand_uint32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random.getrandbits(k)` for `0 <= k <= 64`.
    ///
    /// CPython fills the result's 32-bit words from least significant to most
    /// significant on a little-endian host, so the **first** draw becomes the
    /// **low** word and the last draw is the one that gets truncated.
    ///
    /// # Panics
    /// If `k > 64`. (The environment never needs more; `_randbelow` on a `u64`
    /// bound tops out at `k == 64`.)
    #[inline]
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k <= 64, "pyrng::getrandbits supports k <= 64, got {k}");
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            // Fast path, exactly as in CPython.
            return (self.genrand_uint32() >> (32 - k)) as u64;
        }

        let words = ((k - 1) / 32 + 1) as usize;
        // `k` is signed in CPython and is allowed to go negative; mirror that.
        let mut rem = k as i32;
        let mut out: u64 = 0;
        for i in 0..words {
            let mut r = self.genrand_uint32();
            if rem < 32 {
                r >>= 32 - rem as u32; // Drop least significant bits
            }
            out |= (r as u64) << (32 * i);
            rem -= 32;
        }
        out
    }

    /// `Random._randbelow_with_getrandbits(n)` — uniform in `[0, n)`, `n > 0`.
    ///
    /// Note `k = n.bit_length()`, **not** `(n - 1).bit_length()`. For `n == 8` that
    /// is `getrandbits(4)` with a ~50% rejection rate, not `getrandbits(3)`. The
    /// rejection loop is what makes the stream position state-dependent, so it must
    /// be reproduced draw-for-draw.
    ///
    /// # Panics
    /// If `n == 0` (CPython would loop forever there; `choice` guards it instead).
    #[inline]
    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0, "pyrng::randbelow is defined for n > 0");
        let k = 64 - n.leading_zeros(); // n.bit_length()
        let mut r = self.getrandbits(k); // 0 <= r < 2**k
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// Index chosen by `random.choice(seq)` for a sequence of length `n`.
    ///
    /// The environment calls `rng.choice(sorted(SHOPS))` with `n == 8`.
    ///
    /// # Panics
    /// If `n == 0` (CPython raises `IndexError` there).
    #[inline]
    pub fn choice_index(&mut self, n: usize) -> usize {
        assert!(n > 0, "Cannot choose from an empty sequence");
        self.randbelow(n as u64) as usize
    }
}

impl core::fmt::Debug for PyRandom {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PyRandom")
            .field("index", &self.index)
            .field("mt[0]", &self.mt[0])
            .field("mt[623]", &self.mt[N - 1])
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CPython round()
// ---------------------------------------------------------------------------

/// Python's builtin `round(x)` on a float with no `ndigits` — round-half-to-**even**.
///
/// Transliteration of `float___round___impl` (CPython `Objects/floatobject.c`):
///
/// ```c
/// rounded = round(x);                 /* C round(): half away from zero */
/// if (fabs(x - rounded) == 0.5)       /* halfway case: round to even   */
///     rounded = 2.0 * round(x / 2.0);
/// return PyLong_FromDouble(rounded);
/// ```
///
/// `f64::round` in Rust has exactly C `round`'s semantics (half away from zero),
/// which is why it is only ever used as the *first* step here. Using it alone would
/// give 67 for 66.5 where the Kaggle environment gives 66, silently shifting every
/// market price that lands on a half.
///
/// `x / 2.0` is exact for every non-subnormal double, and `2.0 * y` for an integral
/// `|y| <= 2^52` is exact, so the halfway branch introduces no rounding of its own.
///
/// Out-of-range behaviour differs from Python, which returns an arbitrary-precision
/// int: values beyond `i64` saturate and NaN maps to 0 (Rust `as` cast semantics).
/// Nothing in the environment rounds anything near `2^63`; `market_price` inputs are
/// bounded by a few thousand.
#[inline]
pub fn py_round(x: f64) -> i64 {
    let mut rounded = x.round();
    if (x - rounded).abs() == 0.5 {
        // halfway case: round to even
        rounded = 2.0 * (x / 2.0).round();
    }
    rounded as i64
}

// ---------------------------------------------------------------------------
// Unit tests (self-contained; the cross-checked parity suite lives in
// crates/kaggri-core/tests/pyrng_parity.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Values captured from CPython 3.12.13:
    ///   `random.Random(0).random()` etc.
    #[test]
    fn known_random_streams() {
        let cases: [(i64, [f64; 3]); 5] = [
            (
                0,
                [
                    0.8444218515250481,
                    0.7579544029403025,
                    0.420571580830845,
                ],
            ),
            (
                1,
                [
                    0.13436424411240122,
                    0.8474337369372327,
                    0.763774618976614,
                ],
            ),
            (
                12345,
                [
                    0.41661987254534116,
                    0.010169169457068361,
                    0.8252065092537432,
                ],
            ),
            (
                2147483647,
                [
                    0.3177580158172969,
                    0.8173550078299876,
                    0.1963472909507713,
                ],
            ),
            (
                2147483648,
                [
                    0.3387958510019964,
                    0.5050619906087787,
                    0.8891460738572615,
                ],
            ),
        ];
        for (seed, expect) in cases {
            let mut r = PyRandom::new(seed);
            for (i, e) in expect.iter().enumerate() {
                let got = r.random();
                assert_eq!(
                    got.to_bits(),
                    e.to_bits(),
                    "seed {seed} draw {i}: got {got:?} want {e:?}"
                );
            }
        }
    }

    /// A negative seed must behave like `abs(seed)`.
    #[test]
    fn negative_seed_uses_abs() {
        let mut a = PyRandom::new(-12345);
        let mut b = PyRandom::new(12345);
        for _ in 0..64 {
            assert_eq!(a.random().to_bits(), b.random().to_bits());
        }
        // i64::MIN must not panic on abs.
        let mut c = PyRandom::new(i64::MIN);
        let mut d = PyRandom::from_u64(1u64 << 63);
        assert_eq!(c.random().to_bits(), d.random().to_bits());
    }

    /// Seed 0 still produces a one-word key array `[0]`, which is *not* the same as
    /// an empty key array.
    #[test]
    fn zero_seed_is_one_word() {
        let mut a = PyRandom::new(0);
        let mut b = PyRandom::from_words(&[0]);
        let mut c = PyRandom::from_words(&[]);
        let (x, y, z) = (a.random(), b.random(), c.random());
        assert_eq!(x.to_bits(), y.to_bits());
        assert_eq!(x.to_bits(), z.to_bits(), "empty key must fall back to [0]");
        assert_eq!(x, 0.8444218515250481);
    }

    /// Word-count boundaries: 2**32-1 is one word, 2**32 is two.
    #[test]
    fn seed_word_boundaries() {
        let mut one = PyRandom::from_u64(0xffff_ffff);
        let mut also_one = PyRandom::from_words(&[0xffff_ffff]);
        assert_eq!(one.random().to_bits(), also_one.random().to_bits());

        let mut two = PyRandom::from_u64(1u64 << 32);
        let mut also_two = PyRandom::from_words(&[0, 1]);
        assert_eq!(two.random().to_bits(), also_two.random().to_bits());

        // The high zero word is significant: [0, 1] != [0].
        let mut one_word = PyRandom::from_words(&[0]);
        assert_ne!(two.random().to_bits(), one_word.random().to_bits());
    }

    #[test]
    fn getrandbits_fast_path_matches_raw_draw() {
        // getrandbits(32) is exactly one raw draw.
        let mut a = PyRandom::new(7);
        let mut b = PyRandom::new(7);
        for _ in 0..16 {
            assert_eq!(a.getrandbits(32), b.genrand_uint32() as u64);
        }
        // getrandbits(k<=32) is the top k bits of one raw draw.
        for k in 1..=32u32 {
            let mut a = PyRandom::new(k as i64);
            let mut b = PyRandom::new(k as i64);
            assert_eq!(a.getrandbits(k), (b.genrand_uint32() >> (32 - k)) as u64);
        }
        assert_eq!(PyRandom::new(3).getrandbits(0), 0);
    }

    #[test]
    fn getrandbits_word_order() {
        // k = 64: two full draws, first draw is the LOW word.
        let mut a = PyRandom::new(99);
        let mut b = PyRandom::new(99);
        let lo = b.genrand_uint32() as u64;
        let hi = b.genrand_uint32() as u64;
        assert_eq!(a.getrandbits(64), lo | (hi << 32));

        // k = 33: full low word, then 1 bit taken from the TOP of the second draw.
        let mut a = PyRandom::new(99);
        let mut b = PyRandom::new(99);
        let lo = b.genrand_uint32() as u64;
        let hi = (b.genrand_uint32() >> 31) as u64;
        assert_eq!(a.getrandbits(33), lo | (hi << 32));

        // k = 48
        let mut a = PyRandom::new(99);
        let mut b = PyRandom::new(99);
        let lo = b.genrand_uint32() as u64;
        let hi = (b.genrand_uint32() >> 16) as u64;
        assert_eq!(a.getrandbits(48), lo | (hi << 32));
    }

    #[test]
    fn randbelow_uses_bit_length_of_n_not_n_minus_1() {
        // n = 8 -> k = 4 (8.bit_length() == 4), values 8..15 rejected.
        let mut a = PyRandom::new(2024);
        let mut b = PyRandom::new(2024);
        let mut want = b.getrandbits(4);
        while want >= 8 {
            want = b.getrandbits(4);
        }
        assert_eq!(a.randbelow(8), want);

        // n = 1 -> k = 1, so it still consumes a draw (and rejects 1s).
        let mut a = PyRandom::new(5);
        let mut b = PyRandom::new(5);
        assert_eq!(a.randbelow(1), 0);
        let mut got = b.getrandbits(1);
        while got >= 1 {
            got = b.getrandbits(1);
        }
        assert_eq!(got, 0);
        // Both consumed the same number of draws.
        assert_eq!(a.genrand_uint32(), b.genrand_uint32());
    }

    /// The exact `.5` market-price cases confirmed against the Python env.
    #[test]
    fn py_round_confirmed_cases() {
        assert_eq!(py_round(157.5), 158);
        assert_eq!(py_round(66.5), 66);
        assert_eq!(py_round(59.5), 60);
        assert_eq!(py_round(52.5), 52);
        assert_eq!(py_round(45.5), 46);
        assert_eq!(py_round(38.5), 38);
        // Rust's own round() disagrees on exactly the even-target cases.
        assert_ne!(py_round(66.5), 66.5f64.round() as i64);
        assert_ne!(py_round(52.5), 52.5f64.round() as i64);
        assert_ne!(py_round(38.5), 38.5f64.round() as i64);
        // ...and agrees where the away-from-zero target happens to be even.
        assert_eq!(py_round(157.5), 157.5f64.round() as i64);
        assert_eq!(py_round(59.5), 59.5f64.round() as i64);
        assert_eq!(py_round(45.5), 45.5f64.round() as i64);
    }

    #[test]
    fn py_round_basics() {
        assert_eq!(py_round(0.0), 0);
        assert_eq!(py_round(-0.0), 0);
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(-0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(-1.5), -2);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(-2.5), -2);
        assert_eq!(py_round(0.49999999999999994), 0); // nextafter(0.5, 0)
        assert_eq!(py_round(2.675), 3);
        assert_eq!(py_round(1e17), 100000000000000000);
    }
}
