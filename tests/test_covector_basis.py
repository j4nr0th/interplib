"""Check that covector basis work as expected."""

from itertools import combinations

import pytest
from interplib._interp import CovectorBasis


def test_1d():
    """Testing in 1D is very simple."""
    b00 = CovectorBasis(1)
    b01 = CovectorBasis(1, 0)

    assert ~b01 == b00
    assert b01 == ~b00
    assert b00 ^ b01 == b01
    assert b01 ^ b00 == b01
    assert not b01 ^ b01
    assert b00 ^ b00 == b00


@pytest.mark.parametrize("n", range(2, 5))
def test_nd(n: int):
    """Testing in n-D is a bit more involved."""
    b_full = CovectorBasis(n, *range(n))
    b_empty = CovectorBasis(n)
    for k in range(1, n):
        for c in combinations(range(n), k):
            b1 = CovectorBasis(n, *c)
            assert b1 ^ ~b1 == b_full
            assert ~b1 ^ b1 == -b_full
            assert b1 ^ b_empty == b1
            assert not (b1 ^ b1)
