"""Tests for finite-element visualization helpers."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from fdg import (
    BasisSpecs,
    BasisType,
    CoordinateMap,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationSpace,
    IntegrationSpecs,
    SpaceMap,
)
from fdg.visualization import lagrange_hexahedral_grid, lagrange_quadrilateral_grid


def test_lagrange_quadrilateral_grid_interpolates_nonlinear_data() -> None:
    """High-order quadrilateral ordering preserves a polynomial field."""
    integration = IntegrationSpace(IntegrationSpecs(3), IntegrationSpecs(3))
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    space_map = SpaceMap(
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 0.0, 1.0, 1.0]), integration),
        CoordinateMap(DegreesOfFreedom(basis, [0.0, 1.0, 0.0, 1.0]), integration),
    )
    nodes = np.linspace(0.0, 1.0, 7)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    field = x**2 + 2.0 * y**2 + 3.0 * x * y
    grid = lagrange_quadrilateral_grid([space_map], 6, {"field": [field]})

    query_nodes = np.linspace(0.05, 0.95, 11)
    query_x, query_y = np.meshgrid(query_nodes, query_nodes, indexing="ij")
    query = pv.PolyData(
        np.column_stack(
            [
                query_x.ravel(),
                query_y.ravel(),
                np.zeros(query_x.size),
            ]
        )
    )
    sampled = query.sample(grid)
    expected = (query_x**2 + 2.0 * query_y**2 + 3.0 * query_x * query_y).ravel()
    np.testing.assert_allclose(sampled["field"], expected, atol=1.0e-12)


def test_lagrange_hexahedral_grid_accepts_anisotropic_orders(tmp_path) -> None:
    """Anisotropic orders survive high-order VTK serialization."""
    integration = IntegrationSpace(
        IntegrationSpecs(3), IntegrationSpecs(3), IntegrationSpecs(3)
    )
    basis = FunctionSpace(
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
        BasisSpecs(BasisType.LAGRANGE_UNIFORM, 1),
    )
    coordinates = np.meshgrid(
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 1.0)),
        indexing="ij",
    )
    space_map = SpaceMap(
        *(
            CoordinateMap(DegreesOfFreedom(basis, coordinate.ravel()), integration)
            for coordinate in coordinates
        )
    )
    orders = (1, 2, 3)
    shape = tuple(order + 1 for order in orders)
    field = np.arange(np.prod(shape), dtype=float).reshape(shape)

    grid = lagrange_hexahedral_grid([space_map], orders, {"field": [field]})
    assert grid.n_cells == 1
    assert grid.n_points == np.prod(shape)
    assert grid.celltypes[0] == pv.CellType.LAGRANGE_HEXAHEDRON
    np.testing.assert_array_equal(grid.point_data["field"], field.ravel())
    output = tmp_path / "anisotropic.vtu"
    grid.save(output)
    loaded = pv.read(output)
    degrees = loaded.GetCellData().GetHigherOrderDegrees()
    assert degrees is not None
    assert degrees.GetTuple(0) == (1.0, 2.0, 3.0)
    assert [loaded.GetCell(0).GetOrder(axis) for axis in range(3)] == [1, 2, 3]
