"""Check that indicence matrices work correctly."""

import pytest
from interplib._interp import (
    BasisSpecs,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    incidence_matrix,
)
from interplib.enum_type import BasisType, IntegrationMethod
from interplib.integration import projection_l2_primal


@pytest.mark.parametrize("btype", BasisType)
@pytest.mark.parametrize("order", (1, 2, 5, 10))
def test_incidence(order: int, btype: BasisType) -> None:
    """Check that applying an incidence matrix returns correct DoFs."""
    specs = BasisSpecs(btype, order)
    int_space = IntegrationSpace(IntegrationSpecs(order + 1, IntegrationMethod.GAUSS))
    func_space = FunctionSpace(specs)

    def real_function(*args):
        (x,) = args
        return x**order

    def real_derivative(*args):
        (x,) = args
        return order * x ** (order - 1)

    fn_dofs = projection_l2_primal(real_function, func_space, int_space)
    d_dofs = projection_l2_primal(real_derivative, func_space.lower_order(0), int_space)

    incidence = incidence_matrix(specs)
    computed = incidence @ fn_dofs.values
    expected = d_dofs.values

    assert pytest.approx(expected) == computed


if __name__ == "__main__":
    for btype in BasisType:
        print(f"Started {btype.value}")
        for order in reversed([1, 2, 4, 10]):
            test_incidence(order, btype)
        print(f"Finished {btype.value}")
