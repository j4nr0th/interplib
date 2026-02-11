"""Real k-form mass matrix this time."""

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    compute_kform_mass_matrix,
)
from interplib.domains import Line
from interplib.enum_type import BasisType
from interplib.mass_matrix import compute_inner_prod_mass_matrix


@pytest.mark.parametrize(
    ("order_1", "order_2", "order_left", "order_right", "btype_1", "btype_2", "m"),
    (
        (6, 7, 2, 3, BasisType.BERNSTEIN, BasisType.BERNSTEIN, 1),
        (4, 5, 3, 3, BasisType.LEGENDRE, BasisType.LEGENDRE, 2),
        (4, 5, 2, 1, BasisType.BERNSTEIN, BasisType.LEGENDRE, 4),
        (6, 8, 6, 7, BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS, 5),
    ),
)
def test_1d_to_md(
    order_1: int,
    order_2: int,
    order_left: int,
    order_right: int,
    btype_1: BasisType,
    btype_2: BasisType,
    m: int,
) -> None:
    """Check that 1D to mD inner product is correct."""
    assert order_1 > 0 and order_2 > 0 and m > 0
    rng = np.random.default_rng(order_1**2 + 2 * order_2**2 + m)
    line = Line(
        *(
            0.1 * rng.random((order_1, m))
            + np.stack(m * [np.linspace(-1, +1, order_1)], axis=-1)
        )
    )

    fn_space_left = FunctionSpace(BasisSpecs(btype_1, order_left))
    fn_space_right = FunctionSpace(BasisSpecs(btype_2, order_right))

    int_space = IntegrationSpace(IntegrationSpecs(order_2))
    space_map = line(int_space)

    mat_computed = compute_kform_mass_matrix(space_map, 0, fn_space_left, fn_space_right)
    mat_expected = compute_inner_prod_mass_matrix(
        space_map, 0, fn_space_left, fn_space_right
    )
    assert pytest.approx(mat_expected) == mat_computed

    mat_computed = compute_kform_mass_matrix(space_map, 1, fn_space_left, fn_space_right)
    mat_expected = compute_inner_prod_mass_matrix(
        space_map, 1, fn_space_left, fn_space_right
    )
    assert pytest.approx(mat_expected) == mat_computed
