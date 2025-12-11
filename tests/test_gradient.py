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
from interplib.integration import projection_l2_dual, projection_l2_primal


def check_gradient_1d():
    """Check that gradient on a line works."""

    def function(*args):
        (x,) = args
        return x**2 + 2 * x + 1

    def grad_x(*args):
        (x,) = args
        return 2 * x + 2

    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 4))
    # test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 3))
    int_space = IntegrationSpace(IntegrationSpecs(5, IntegrationMethod.GAUSS))

    base_dofs = projection_l2_primal(function, func_space, int_space)
    grad_x_dofs = projection_l2_dual(grad_x, func_space, int_space)

    # TODO: fix gradient case when no space map is given!
    grad_matrix_x = compute_gradient_mass_matrix(func_space, func_space, int_space, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x


def check_gradient_1d_to_2d():
    """Check that gradient on a line works."""

    def function(*args):
        x, y = args
        return x**2 + 2 * x * y - y + 1

    def grad_x(*args):
        x, y = args
        return 2 * x + 2 * y

    def grad_y(*args):
        x, y = args
        return 2 * x - 1 + 0 * y

    domain = Line((-1, -1), (+1, +1))

    func_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 5))
    test_space = FunctionSpace(BasisSpecs(BasisType.BERNSTEIN, 4))
    int_space = IntegrationSpace(IntegrationSpecs(5, IntegrationMethod.GAUSS))

    domain_map = domain(int_space)

    base_dofs = projection_l2_primal(function, func_space, domain_map)
    grad_x_dofs = projection_l2_dual(grad_x, test_space, domain_map)
    grad_y_dofs = projection_l2_dual(grad_y, test_space, domain_map)

    grad_matrix_x = compute_gradient_mass_matrix(func_space, test_space, domain_map, 0)
    comp_dual_x = grad_matrix_x @ base_dofs.values
    assert pytest.approx(grad_x_dofs.values) == comp_dual_x

    grad_matrix_y = compute_gradient_mass_matrix(func_space, test_space, domain_map, 1)
    comp_dual_y = grad_matrix_y @ base_dofs.values
    assert pytest.approx(grad_y_dofs.values) == comp_dual_y


if __name__ == "__main__":
    check_gradient_1d()
