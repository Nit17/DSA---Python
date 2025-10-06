"""Mathematical Algorithms – Number Theory Essentials
=====================================================
Core helpers used throughout algorithmic problems.

Included:
- GCD, LCM, Extended GCD
- Fast exponentiation (pow), Modular exponentiation
- Modular inverse (with existence check)
- Modular add/mul helpers (safe with large ints)
- Sieve of Eratosthenes (primes up to n)
"""

from __future__ import annotations
from typing import List, Optional, Tuple


class MathAlgorithms:
    """Number-theoretic algorithms and modular arithmetic helpers."""

    # ---------------------- GCD / LCM / Extended GCD ----------------------
    def gcd(self, a: int, b: int) -> int:
        """Euclidean algorithm for greatest common divisor."""
        while b:
            a, b = b, a % b
        return abs(a)

    def lcm(self, a: int, b: int) -> int:
        """Least common multiple using gcd."""
        if a == 0 or b == 0:
            return 0
        return abs(a // self.gcd(a, b) * b)

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclid: returns (g, x, y) with ax + by = g = gcd(a,b)."""
        if b == 0:
            return (abs(a), 1 if a >= 0 else -1, 0)
        g, x1, y1 = self.extended_gcd(b, a % b)
        x, y = y1, x1 - (a // b) * y1
        return (g, x, y)

    # -------------------------- Fast exponentiation -----------------------
    def fast_pow(self, a: int, n: int) -> int:
        """Compute a**n in O(log n) using binary exponentiation."""
        if n < 0:
            raise ValueError("fast_pow expects non-negative exponent")
        res = 1
        base = a
        e = n
        while e > 0:
            if e & 1:
                res *= base
            base *= base
            e >>= 1
        return res

    def mod_pow(self, a: int, n: int, mod: int) -> int:
        """Compute a**n % mod in O(log n). mod must be > 0."""
        if mod <= 0:
            raise ValueError("mod must be positive")
        a %= mod
        res = 1
        e = n
        if e < 0:
            # use modular inverse of a when gcd(a, mod) == 1
            g = self.gcd(a, mod)
            if g != 1:
                raise ValueError("negative exponent requires invertible base modulo mod")
            a = self.mod_inverse(a, mod)
            e = -e
        while e > 0:
            if e & 1:
                res = (res * a) % mod
            a = (a * a) % mod
            e >>= 1
        return res

    # -------------------------- Modular arithmetic ------------------------
    def mod_add(self, a: int, b: int, mod: int) -> int:
        if mod <= 0:
            raise ValueError("mod must be positive")
        return (a % mod + b % mod) % mod

    def mod_mul(self, a: int, b: int, mod: int) -> int:
        if mod <= 0:
            raise ValueError("mod must be positive")
        return (a % mod) * (b % mod) % mod

    def mod_inverse(self, a: int, mod: int) -> int:
        """Return modular inverse of a modulo mod, if it exists.

        Raises ValueError if inverse doesn't exist (i.e., gcd(a, mod) != 1).
        For prime mod, use Fermat's little theorem: a^(mod-2) % mod.
        """
        a %= mod
        g, x, _y = self.extended_gcd(a, mod)
        if g != 1:
            raise ValueError("inverse does not exist")
        return x % mod

    # -------------------------- Sieve of Eratosthenes ---------------------
    def sieve_of_eratosthenes(self, n: int) -> List[int]:
        """Return list of primes <= n using classic sieve. O(n log log n)."""
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        p = 2
        while p * p <= n:
            if sieve[p]:
                step = p
                start = p * p
                sieve[start : n + 1 : step] = [False] * ((n - start) // step + 1)
            p += 1
        return [i for i, is_prime in enumerate(sieve) if is_prime]


if __name__ == "__main__":
    m = MathAlgorithms()
    print(m.gcd(54, 24), m.lcm(21, 6))
    print(m.fast_pow(2, 10), m.mod_pow(2, 10, 1_000_000_007))
    print(m.mod_inverse(3, 11))
    print(m.sieve_of_eratosthenes(30))
