"""Check that the gradien mass matrix works as one would expect."""

import pytest
from interplib import (
    BasisSpecs,
    BasisType,
    FunctionSpace,
    IntegrationMethod,
    IntegrationSpace,
    IntegrationSpecs,
    Line,
)
from interplib._interp import compute_gradient_mass_matrix
from interplib.domains import Quad
from interplib.integration import projection_l2_dual, projection_l2_primal


@pytest.mark.parametrize("order", (1, 2, 4))
def test_gradient_1d(order: int):
    """Check that gradient on a line works."""

    def function(*args):
        (x,) = args
        return x**order

    def grad_x(*args):
        (x,) = args
        return order * x ** (order - 1)

    domain = Line((0,), (+1,), (+5,))
    k = 4
    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, order + k))
    test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, order + k - 1))
    int_space = IntegrationSpace(IntegrationSpecs(order + k + 2, IntegrationMethod.GAUSS))

    smap = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, smap)
    grad_x_dofs = projection_l2_dual(grad_x, test_space, smap)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space, smap, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x


# NOTE: this test can mathematically not work! If you have a 1D object in 2D space,
# you can only recover the tangential gradient along it (duh!)
def _check_gradient_1d_to_2d():
    """Check that gradient on a line works."""

    def function(*args):
        x, y = args
        return 3 * (x + y) ** 2 + 2 * (x + y) + 1

    def grad_x(*args):
        x, y = args
        return 6 * (x + y) + 2 * x

    def grad_y(*args):
        x, y = args
        return 6 * (x + y) + 2 * y

    domain = Line((0, 0), (1, 1))

    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 3))
    test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 2))
    int_space = IntegrationSpace(IntegrationSpecs(4, IntegrationMethod.GAUSS))

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)
    grad_x_dofs = projection_l2_dual(grad_x, test_space, domain_map)
    grad_y_dofs = projection_l2_dual(grad_y, test_space, domain_map)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space, domain_map, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values

    grad_matrix_y = compute_gradient_mass_matrix(func_space, test_space, domain_map, 1)
    comp_dual_y = grad_matrix_y @ base_dofs.values
    print(f"{base_dofs.values=}")
    print(f"{grad_matrix_x=}")
    print(f"{grad_matrix_y=}")
    print("Real x:", grad_x_dofs.values)
    print("Comp x:", comp_dual_x)
    print("Real y:", grad_y_dofs.values)
    print("Comp y:", comp_dual_y)
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x
    assert pytest.approx(grad_y_dofs.values) == comp_dual_y


@pytest.mark.parametrize(("order1", "order2"), ((6, 6), (8, 6), (6, 9)))
def test_gradient_2d_to_2d(order1: int, order2: int):
    """Check that gradient on a surface works."""

    def function(*args):
        x, y = args
        return 2 * (x + 1) ** 2 + 2 * y**2 + 1

    def grad_x(*args):
        x, y = args
        return 4 * (x + 1) + 0 * y

    def grad_y(*args):
        x, y = args
        return 4 * y + 0 * x + 0 * y

    domain = Quad(
        bottom=Line((-1.5, -1), (+1, -2)),
        right=Line((+1, -2), (0, +1), (+1, +1)),
        top=Line((+1, +1), (0, +0.5), (0, +1.5), (-1, +1)),
        left=Line((-1, +1), (-0.7, 0.2), (-1.5, -1)),
    )

    func_space = FunctionSpace(
        BasisSpecs(BasisType.BERNSTEIN, order1), BasisSpecs(BasisType.BERNSTEIN, order2)
    )
    test_space_x = FunctionSpace(
        BasisSpecs(BasisType.BERNSTEIN, order1 - 1),
        BasisSpecs(BasisType.BERNSTEIN, order2),
    )
    test_space_y = FunctionSpace(
        BasisSpecs(BasisType.BERNSTEIN, order1),
        BasisSpecs(BasisType.BERNSTEIN, order2 - 1),
    )
    int_space = IntegrationSpace(
        IntegrationSpecs(order1 + 1, IntegrationMethod.GAUSS),
        IntegrationSpecs(order2 + 1, IntegrationMethod.GAUSS),
    )

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)
    grad_x_dofs = projection_l2_dual(grad_x, test_space_x, domain_map)
    grad_y_dofs = projection_l2_dual(grad_y, test_space_y, domain_map)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space_x, domain_map, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values.flatten()

    grad_matrix_y = compute_gradient_mass_matrix(func_space, test_space_y, domain_map, 1)
    comp_dual_y = grad_matrix_y @ base_dofs.values.flatten()

    assert pytest.approx(grad_x_dofs.values.flatten()) == comp_dual_x
    assert pytest.approx(grad_y_dofs.values.flatten()) == comp_dual_y


if __name__ == "__main__":
    test_gradient_1d(4)
