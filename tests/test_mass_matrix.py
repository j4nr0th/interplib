"""Check that mass matrix computations are correct."""

import numpy as np
import pytest
from interplib import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    compute_mass_matrix,
)


@pytest.mark.parametrize(
    ("order_1,basis_type_1,order_2,basis_type_2"),
    (
        (1, BasisType.LEGENDRE, 1, BasisType.LEGENDRE),
        (3, BasisType.LEGENDRE, 3, BasisType.LEGENDRE),
        (2, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 2, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (2, BasisType.LEGENDRE, 3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS),
        (3, BasisType.LAGRANGE_CHEBYSHEV_GAUSS, 2, BasisType.LEGENDRE),
        (4, BasisType.LEGENDRE, 5, BasisType.BERNSTEIN),
    ),
)
def test_mass_matrix_1d(
    order_1: int, basis_type_1: BasisType, order_2: int, basis_type_2: BasisType
) -> None:
    """Check mass matrix computation in 1D."""
    int_space = IntegrationSpace(
        IntegrationSpecs(max(order_1, order_2) + 1, IntegrationMethod.GAUSS)
    )
    func_space_1 = FunctionSpace(BasisSpecs(basis_type_1, order_1))
    func_space_2 = FunctionSpace(BasisSpecs(basis_type_2, order_2))

    mass_matrix = compute_mass_matrix(func_space_1, func_space_2, int_space)

    int_nodes = int_space.nodes()[0, ...]
    int_weights = int_space.weights()

    vals_1 = func_space_1.evaluate(int_nodes)
    vals_2 = func_space_2.evaluate(int_nodes)

    expected_mass_matrix = np.sum(
        vals_1[..., None, :] * vals_2[..., :, None] * int_weights[:, None, None], axis=0
    )
    assert pytest.approx(mass_matrix) == expected_mass_matrix


if __name__ == "__main__":
    test_mass_matrix_1d(2, BasisType.LEGENDRE, 3, BasisType.LEGENDRE)
