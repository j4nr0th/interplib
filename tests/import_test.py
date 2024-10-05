"""Test very basic functionality."""
import pytest

import numpy as np

from interplib import _interp

from interplib import lagrange1d

def test_internal_import():
    assert _interp.test() == "Test successful!\n"

@pytest.mark.parametrize("n", np.logspace(1, 6, base=2, num=10, dtype=int))
def test_lagrange1d(n):
    np.random.seed(0)
    xvals = np.sort(np.random.random_sample(n))
    yvals = np.random.random_sample(n)
    xvals = np.array(xvals, dtype=np.float64)
    yvals = np.array(yvals, dtype=np.float64)

    interpolated = _interp.lagrange1d(xvals, xvals, yvals)
    assert np.all(np.abs(interpolated - yvals) < 1e-15 * n)







