"""Check coordinate mappings work as expected."""

import numpy as np
import pytest
from interplib._interp import (
    BasisSpecs,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
)
from interplib.enum_type import BasisType

_TEST_ORDERS = (1, 2, 5, 10)


@pytest.mark.parametrize("int_order", _TEST_ORDERS)
@pytest.mark.parametrize("basis_order", _TEST_ORDERS)
@pytest.mark.parametrize("basis_type", BasisType)
def test_coord_1d(int_order: int, basis_order: int, basis_type: BasisType) -> None:
    """Check that coordinate as a function of 1 variable works."""
    rng = np.random.default_rng(2198)

    int_spec = IntegrationSpecs(int_order, method="gauss-lobatto")
    int_space = IntegrationSpace(int_spec)

    b_spec = BasisSpecs(basis_type, basis_order)
    b_space = FunctionSpace(b_spec)

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )


_TEST_ORDERS_2D = ((1, 1), (2, 3), (10, 3), (10, 10))
_TEST_BASIS_2D = (
    (BasisType.BERNSTEIN, BasisType.BERNSTEIN),
    (BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS),
    (BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS_LOBATTO),
)


@pytest.mark.parametrize(("int_order_1", "int_order_2"), _TEST_ORDERS_2D)
@pytest.mark.parametrize(("basis_order_1", "basis_order_2"), _TEST_ORDERS_2D)
@pytest.mark.parametrize(("basis_type_1", "basis_type_2"), _TEST_BASIS_2D)
def test_coord_2d(
    int_order_1: int,
    basis_order_1: int,
    basis_type_1: BasisType,
    int_order_2: int,
    basis_order_2: int,
    basis_type_2: BasisType,
) -> None:
    """Check that coordinate as a function of 1 variable works."""
    rng = np.random.default_rng(2198)

    int_space = IntegrationSpace(
        IntegrationSpecs(int_order_1, method="gauss-lobatto"),
        IntegrationSpecs(int_order_2, method="gauss-lobatto"),
    )

    b_space = FunctionSpace(
        BasisSpecs(basis_type_1, basis_order_1), BasisSpecs(basis_type_2, basis_order_2)
    )

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )
    assert np.all(
        coord_map.gradient(1)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[1])
    )


_TEST_ORDERS_3D = (
    (1, 1, 2),
    (2, 3, 1),
    (10, 3, 4),
)
_TEST_BASIS_3D = (
    (BasisType.BERNSTEIN, BasisType.BERNSTEIN, BasisType.LAGRNAGE_GAUSS_LOBATTO),
    (BasisType.LAGRANGE_UNIFORM, BasisType.LAGRNAGE_GAUSS, BasisType.LEGENDRE),
    (BasisType.LEGENDRE, BasisType.LAGRNAGE_GAUSS_LOBATTO, BasisType.LAGRANGE_UNIFORM),
)


@pytest.mark.parametrize(("int_order_1", "int_order_2", "int_order_3"), _TEST_ORDERS_3D)
@pytest.mark.parametrize(
    ("basis_order_1", "basis_order_2", "basis_order_3"), _TEST_ORDERS_3D
)
@pytest.mark.parametrize(("basis_type_1", "basis_type_2", "basis_type_3"), _TEST_BASIS_3D)
def test_coord_3d(
    int_order_1: int,
    basis_order_1: int,
    basis_type_1: BasisType,
    int_order_2: int,
    basis_order_2: int,
    basis_type_2: BasisType,
    int_order_3: int,
    basis_order_3: int,
    basis_type_3: BasisType,
) -> None:
    """Check that coordinate as a function of 3 variable works."""
    rng = np.random.default_rng(2198)

    int_space = IntegrationSpace(
        IntegrationSpecs(int_order_1, method="gauss-lobatto"),
        IntegrationSpecs(int_order_2, method="gauss-lobatto"),
        IntegrationSpecs(int_order_3, method="gauss-lobatto"),
    )

    b_space = FunctionSpace(
        BasisSpecs(basis_type_1, basis_order_1),
        BasisSpecs(basis_type_2, basis_order_2),
        BasisSpecs(basis_type_3, basis_order_3),
    )

    dofs = DegreesOfFreedom(b_space)
    dofs.values = rng.random(dofs.values.shape)

    coord_map = CoordinateMap(dofs, int_space)

    assert np.all(coord_map.values == dofs.reconstruct_at_integration_points(int_space))
    assert np.all(
        coord_map.gradient(0)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[0])
    )
    assert np.all(
        coord_map.gradient(1)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[1])
    )
    assert np.all(
        coord_map.gradient(2)
        == dofs.reconstruct_derivative_at_integration_points(int_space, idim=[2])
    )


# def test_space_map_2_to_2(n_in: int, n_out: int, btype: BasisType) -> None:
#     """Test that a 2D -> 2D space map works."""
#     # Create the integration space
#     int_space = IntegrationSpace(IntegrationSpecs(n_in + 1), IntegrationSpecs(n_in - 1))
#     # Create the function space
#     func_space = FunctionSpace(BasisSpecs(btype, n_out - 1),
# BasisSpecs(btype, n_out + 1))
#     # Create the


if __name__ == "__main__":
    test_coord_1d(6, 6, BasisType.BERNSTEIN)
