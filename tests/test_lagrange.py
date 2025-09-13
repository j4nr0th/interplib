"""Tests for Lagrange interpolation."""

import numpy as np
import pytest
from interplib.lagrange import (
    interp1d_2derivative_samples,
    interp1d_derivative_samples,
    interp1d_function_samples,
)


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_funciton_interpolation(order: int, test_samples: int) -> None:
    """Check a polynomial of some order is interpolated with enough samples."""
    np.random.seed(1512)
    coeffs = np.random.random_sample(order + 1)
    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = sum(nodes**i * c for i, c in enumerate(coeffs))
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_function_samples(test_nodes, nodes, values)
    assert pytest.approx(test_results) == sum(
        test_nodes**i * c for i, c in enumerate(coeffs)
    )


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_derivative_interpolation(order: int, test_samples: int) -> None:
    """Check a derivative of some order is interpolated with enough samples."""
    np.random.seed(1512)
    coeffs = np.random.random_sample(order + 1)

    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = sum(nodes**i * c for i, c in enumerate(coeffs))
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_derivative_samples(test_nodes, nodes, values)
    assert pytest.approx(test_results) == sum(
        i * test_nodes ** (i - 1) * c for i, c in enumerate(coeffs)
    )


@pytest.mark.parametrize("order,test_samples", ((1, 20), (3, 100), (10, 410)))
def test_2derivative_interpolation(order: int, test_samples: int) -> None:
    """Check a derivative of some order is interpolated with enough samples."""
    np.random.seed(1512)
    coeffs = np.random.random_sample(order + 1)
    nodes = np.sort(
        np.random.random_sample(order + 1)
    )  # Let's just assume we won't have duplicates
    values = sum(nodes**i * c for i, c in enumerate(coeffs))
    test_nodes = np.linspace(nodes[0], nodes[-1], test_samples)
    test_results = interp1d_2derivative_samples(test_nodes, nodes, values)
    assert pytest.approx(test_results) == sum(
        i * (i - 1) * test_nodes ** (i - 2) * c for i, c in enumerate(coeffs)
    )
