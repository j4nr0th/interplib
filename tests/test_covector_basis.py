"""Check that covector basis work as expected."""

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


if __name__ == "__main__":
    test_1d()
