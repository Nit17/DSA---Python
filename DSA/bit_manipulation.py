"""Bit Manipulation – Handy Tricks and Utilities
================================================
This module collects foundational bit hacks and helpers used across problems.

Highlights:
- Single-bit operations: get/set/clear/toggle
- Counting set bits (Kernighan), parity
- Power of two checks, lowest/highest set bit
- Iterate all subsets of a bitmask

All operations are O(1) unless otherwise noted.
"""

from __future__ import annotations
from typing import Iterator, List, Tuple


class BitAlgorithms:
    """Utility methods for common bit operations."""

    # ----------------------- Single-bit operations -----------------------
    def get_bit(self, x: int, i: int) -> int:
        """Return the i-th bit of x (0 or 1)."""
        return (x >> i) & 1

    def set_bit(self, x: int, i: int) -> int:
        """Set the i-th bit of x to 1."""
        return x | (1 << i)

    def clear_bit(self, x: int, i: int) -> int:
        """Clear the i-th bit of x to 0."""
        return x & ~(1 << i)

    def toggle_bit(self, x: int, i: int) -> int:
        """Flip the i-th bit of x."""
        return x ^ (1 << i)

    # -------------------- Aggregate bit properties ----------------------
    def count_set_bits(self, x: int) -> int:
        """Count set bits using Kernighan's trick: O(k), k = set bits."""
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    def is_power_of_two(self, x: int) -> bool:
        """Return True iff x is a power of two (>0)."""
        return x > 0 and (x & (x - 1)) == 0

    def parity(self, x: int) -> int:
        """Return parity (0 even, 1 odd) of number of set bits."""
        p = 0
        while x:
            p ^= 1
            x &= x - 1
        return p

    def lowest_set_bit(self, x: int) -> int:
        """Return value with only lowest set bit kept (x & -x)."""
        return x & -x

    def highest_set_bit_index(self, x: int) -> int:
        """Return index of highest set bit (0-based), or -1 if x==0."""
        if x == 0:
            return -1
        return x.bit_length() - 1

    # ------------------------- Subset generation ------------------------
    def iterate_subsets(self, mask: int) -> Iterator[int]:
        """Generate all submasks of mask (including 0 and mask)."""
        sub = mask
        while True:
            yield sub
            if sub == 0:
                break
            sub = (sub - 1) & mask


if __name__ == "__main__":
    b = BitAlgorithms()
    x = 0b10110
    print(b.get_bit(x, 1), b.set_bit(x, 0), b.clear_bit(x, 4))
    print(b.count_set_bits(x), b.is_power_of_two(16), b.parity(x))
    print(bin(b.lowest_set_bit(x)), b.highest_set_bit_index(x))
    print(list(map(bin, b.iterate_subsets(0b1011))))
