"""Spline auxiliary functions and classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from interplib._interp import Polynomial1D, Spline1D, Spline1Di

__all__ = [
    "SplineBC",
    "element_interpolating_splinei",
    "nodal_interpolating_splinei",
    "element_interpolating_spline",
    "nodal_interpolating_spline",
]


@dataclass(init=False, frozen=True)
class SplineBC:
    r"""Represents a boundary condition for a spline.

    This represents the boundary condition expressed as:

    :math:`\sum\limits_{n=0}^N k_n \frac{d^n}{{dt}^n}(y) = v`,

    where the values :math:`k_n` correspond to entries in ``coeffs[n]``.

    Parameters
    ----------
    coeffs : (N,) array_like
        Represents coefficients of the spline's derivatives.
    value : float
        Value of the weighted sum for the boundary condition.
    """

    coefficients: npt.NDArray[np.float64]
    value: float

    def __init__(self, coeffs: npt.ArrayLike, value: float) -> None:
        k = np.array(coeffs, np.float64)
        if len(k.shape) != 1:
            raise ValueError("Coefficient array must be 1D")
        v = float(value)
        if np.all(k == 0.0):
            raise ValueError("All coefficients were zero.")
        object.__setattr__(self, "coefficients", k)
        object.__setattr__(self, "value", v)


def _ensure_boundary_conditions(
    n: int, bc_left: Iterable[SplineBC] | None, bc_right: Iterable[SplineBC] | None
) -> tuple[list[SplineBC], list[SplineBC]]:
    """Create natural boundary conditions for n order spline if more are needed.

    Parameters
    ----------
    n : int
        Order of the spline for which these should be prepared. It can be either odd or
        even.
    bc_left : Iterable of SplineBC or None
        Iterable of boundary conditions on the left side. If it is instead ``None``, then
        these will be created.
    bc_right : Iterable of SplineBC or None
        Iterable of boundary conditions on the left side. If it is instead ``None``, then
        these will be created.

    Returns
    -------
    (list of SplineBC, list of SplineBC)
        Tuple of two lists of boundary conditions to use for the spline.
    """
    bcs_left: list[SplineBC] = []
    bcs_right: list[SplineBC] = []

    expected_order = n  # if (n & 1) else n - 1
    expected_count = n - 1 if (n & 1) else n

    if bc_left is not None:
        for ib, bc in enumerate(bc_left):
            if bc.coefficients.shape[0] != expected_order:
                raise ValueError(
                    f"Boundary condition {ib} on left boundary has the wrong degree (got"
                    f" {bc.coefficients.shape[0]} when expecting {expected_order})."
                )
            bcs_left.append(bc)

    if bc_right is not None:
        for ib, bc in enumerate(bc_right):
            if bc.coefficients.shape[0] != expected_order:
                raise ValueError(
                    f"Boundary condition {ib} on right boundary has the wrong degree (got"
                    f" {bc.coefficients.shape[0]} when expecting {expected_order})."
                )
            bcs_right.append(bc)

    if len(bcs_left) + len(bcs_right) > expected_count:
        raise ValueError(
            "Number of left and right boundary conditions exceeds the required amount "
            f"(expected {expected_count}, got {len(bcs_left)} on the left and"
            f"{len(bcs_right)} on the right)."
        )

    if bc_left is None:
        n_needed = n // 2 if bc_right is None else expected_count - len(bcs_right)
        k = np.empty(expected_order, np.float64)
        for i in range(n_needed):
            k[:] = 0
            k[expected_order - 1 - i] = 1.0
            bcs_left.append(SplineBC(k, 0.0))

    if bc_right is None:
        n_needed = expected_count - len(bcs_left)
        k = np.empty(expected_order, np.float64)
        for i in range(n_needed):
            k[:] = 0
            k[expected_order - 1 - i] = 1.0
            bcs_right.append(SplineBC(k, 0.0))

    return (bcs_left, bcs_right)


def _element_polynomial_coefficients(n: int) -> npt.NDArray[np.floating]:
    """Return coefficients for element polynomials.

    Parameters
    ----------
    n : int
        Order of polynomials. Number of terms will be one higher.

    Returns
    -------
    (n + 1, n + 1) array
        Matrix, where each row has the coefficients of the polynomial
        basis in from the constant term, to the :math:`t^n` term.
    """
    assert n & 1 == 0
    a = np.zeros((n + 1, n + 1), np.float64)
    # First row is anti-derivative
    k = np.arange(n + 1)
    a[0, :] = 1 / (k + 1)

    f = 1.0
    k = np.ones(n + 1, int)
    for r in range(n // 2):
        a[2 * r + 1, r] = f
        a[2 * r + 2, :] = k
        f *= r + 1
        k[r:] *= np.arange(n + 1 - r)
    return np.linalg.inv(a).T


def _element_interpolating_basis(n: int) -> tuple[Polynomial1D, ...]:
    """Create interpolating basis for element spline with degree n.

    Parameters
    ----------
    n : int
        Order of polynomials to use. Number of terms will be one more.

    Returns
    -------
    tuple of Polynomial1D
        Tuple of basis polynomials for element-centered interpolation.
    """
    assert n & 1 == 0
    m = _element_polynomial_coefficients(n)

    # re-order rows of m
    def _permute_gen(n: int) -> Generator[int, None, None]:
        """Correctly order the rows."""
        yield 0
        for i in range(n // 2):
            yield 2 * i + 1
        for i in range(n // 2):
            yield 2 * i + 2

    rows = _permute_gen(n)
    return tuple(Polynomial1D(m[i, :]) for i in rows)


def _element_interpolation_system(
    n: int,
    basis: Iterable[Polynomial1D],
    avg: npt.ArrayLike,
    bc_left: Iterable[SplineBC],
    bc_right: Iterable[SplineBC],
) -> tuple[sp.csr_array, npt.NDArray[np.float64]]:
    """Create interpolation matrix for natural element.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    avg : (M,) array_like
        Averages on the elements which are to be interpolated.
    bc_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    ((M + 1) * n//2, (M + 1) * n//2) csr_array
        System matrix which gives coefficients required to construct the spine.
    ((M + 1) * n//2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 0
    values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.array(values)  # , axis=0)
    averages = np.array(avg, np.float64)
    assert len(averages.shape) == 1
    nelem = int(averages.shape[0])
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)

    n_node = n // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = sp.lil_array(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n)
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n)

        for j in range(n):
            coeffs += bc.coefficients[j] * v[j, 1:, 0]
            k[p] -= bc.coefficients[j] * v[j, 0, 0] * averages[0]
        m[p, inode] = coeffs
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = np.arange(n)
        iright = n_node + np.arange(n)
        for j in range(n_node, n):
            coeffs = np.zeros(3 * n_node)
            coeffs[ileft] -= v[j, 1:, 1]
            coeffs[iright] += v[j, 1:, 0]
            m[p, n_node * i + np.arange(coeffs.shape[0])] = coeffs
            k[p] = v[j, 0, 1] * averages[i] - v[j, 0, 0] * averages[i + 1]
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        ind = np.arange(n) + (nelem - 1) * n_node
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n)

        for j in range(n):
            coeffs += bc.coefficients[j] * v[j, 1:, 1]
            k[p] -= bc.coefficients[j] * v[j, 0, 1] * averages[-1]
        m[p, ind] = coeffs
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m.tocsr(), k


def element_interpolating_splinei(
    n: int,
    avgs: npt.ArrayLike,
    bcs_left: Iterable[SplineBC] | None = None,
    bcs_right: Iterable[SplineBC] | None = None,
) -> Spline1Di:
    """Create interpolating spline, which has specified averages.

    Parameters
    ----------
    n : int
        Order of the spline. Must be even, otherwise a ``ValueError`` will be raised.
    avgs : (N,) array_like
        Averages of elements to hit.
    bcs_left : Iterable of SplineBC, optional
        Boundary conditions of the left side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.
    bcs_right : Iterable of SplineBC, optional
        Boundary conditions of the right side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.

    Returns
    -------
    Spline1Di
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    bcs_left, bcs_right = _ensure_boundary_conditions(n, bcs_left, bcs_right)
    if (n & 1) != 0:
        raise ValueError(f"Spline order must be even (instead it was {n}).")
    averages = np.array(avgs, np.float64)
    if len(averages.shape) != 1:
        raise ValueError(
            f"Averages should be a 1D array, instead they have the shape {averages.shape}"
        )
    basis = _element_interpolating_basis(n)
    m, k = _element_interpolation_system(n, basis, averages, bcs_left, bcs_right)
    r = sla.spsolve(m, k)
    poly: list[Polynomial1D] = []
    for i, a in enumerate(averages[:]):
        p = basis[0] * float(a) + sum(basis[j + 1] * r[n // 2 * i + j] for j in range(n))
        poly.append(p)
    spl = Spline1Di(tuple(p.coefficients for p in poly))
    return spl


def _element_interpolation_system2(
    n: int,
    basis: Iterable[Polynomial1D],
    avg: npt.NDArray[np.float64],
    nds: npt.NDArray[np.float64],
    bc_left: list[SplineBC],
    bc_right: list[SplineBC],
) -> tuple[sp.csr_array, npt.NDArray[np.float64]]:
    """Create interpolation matrix for natural element.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    avg : (M,) array of float64
        Averages on the elements which are to be interpolated.
    nds : (M,) array of float64
        Nodes of the elements.
    bc_left : list of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : list of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    ((M + 1) * n//2, (M + 1) * n//2) csr_array
        System matrix which gives coefficients required to construct the spine.
    ((M + 1) * n//2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 0
    values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.array(values)  # , axis=0)
    averages = np.array(avg, np.float64)
    assert len(averages.shape) == 1
    nelem = int(averages.shape[0])
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)

    dx = nds[1:] - nds[:-1]

    n_node = n // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = sp.lil_array(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n)
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n)

        for j in range(n):
            coeffs += (
                bc.coefficients[j]
                * v[j, 1:, 0]
                * (dx[0] ** np.tile(np.arange(n // 2), 2))
                / dx[0] ** j
            )
            k[p] -= bc.coefficients[j] * v[j, 0, 0] * averages[0] / dx[0] ** j
        m[p, inode] = coeffs
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = np.arange(n)
        iright = n_node + np.arange(n)
        for j in range(n_node, n):
            coeffs = np.zeros(3 * n_node)
            coeffs[ileft] -= (
                v[j, 1:, 1] * (dx[i] ** np.tile(np.arange(n // 2), 2)) / dx[i] ** j
            )
            coeffs[iright] += (
                v[j, 1:, 0]
                * (dx[i + 1] ** np.tile(np.arange(n // 2), 2))
                / dx[i + 1] ** j
            )
            m[p, n_node * i + np.arange(coeffs.shape[0])] = coeffs
            k[p] = (v[j, 0, 1] * averages[i]) / dx[i] ** (j) - (
                v[j, 0, 0] * averages[i + 1]
            ) / dx[i + 1] ** (j)
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        ind = np.arange(n) + (nelem - 1) * n_node
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n)

        for j in range(n):
            coeffs += (
                bc.coefficients[j]
                * v[j, 1:, 1]
                * (dx[-1] ** np.tile(np.arange(n // 2), 2))
                / dx[-1] ** j
            )
            k[p] -= bc.coefficients[j] * v[j, 0, 1] * averages[-1] / dx[-1] ** j
        m[p, ind] = coeffs
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m.tocsr(), k


def element_interpolating_spline(
    n: int,
    avgs: npt.ArrayLike,
    nds: npt.ArrayLike,
    bcs_left: Iterable[SplineBC] | None = None,
    bcs_right: Iterable[SplineBC] | None = None,
) -> Spline1D:
    """Create interpolating spline, which has specified averages.

    Parameters
    ----------
    n : int
        Order of the spline. Must be even, otherwise a ``ValueError`` will be raised.
    integrals : (N,) array_like
        Integrals over each element. By definition equal to element length times the
        average.
    nds : (N + 1,) array_like
        Nodes of the elements.
    bcs_left : Iterable of SplineBC, optional
        Boundary conditions of the left side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.
    bcs_right : Iterable of SplineBC, optional
        Boundary conditions of the right side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.

    Returns
    -------
    Spline1D
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    bcs_left, bcs_right = _ensure_boundary_conditions(n, bcs_left, bcs_right)
    if (n & 1) != 0:
        raise ValueError(f"Spline order must be even (instead it was {n}).")
    averages = np.array(avgs, np.float64)
    if len(averages.shape) != 1:
        raise ValueError(
            f"Averages should be a 1D array, instead they have the shape {averages.shape}"
        )
    nodes = np.array(nds, np.float64)
    if len(nodes.shape) != 1 or nodes.shape[0] != averages.shape[0] + 1:
        raise ValueError(
            f"Nodes should be a 1D array with one element more than averages,"
            f" instead averages have the shape {averages.shape} and the nodes"
            f" have the shape {nodes.shape}."
        )
    dx = nodes[1:] - nodes[:-1]
    averages /= dx
    basis = _element_interpolating_basis(n)
    m, k = _element_interpolation_system2(n, basis, averages, nodes, bcs_left, bcs_right)
    r = sla.spsolve(m, k)
    poly: list[Polynomial1D] = []
    for i, a in enumerate(averages[:]):
        d = np.tile(dx[i] ** (np.arange(n // 2)), 2)
        # Make into local basis
        local = tuple(
            Polynomial1D(p.coefficients / dx[i] ** np.arange(p.order + 1)) for p in basis
        )
        p = (
            local[0] * float(a)
            # add this
            + sum(local[j + 1] * (r[n // 2 * i + j] * d[j]) for j in range(n))
        )
        poly.append(p)
    spl = Spline1D(nodes, tuple(p.coefficients for p in poly))
    return spl


def _nodal_polynomial_coefficients(n: int) -> npt.NDArray[np.floating]:
    """Return coefficients for nodal polynomials.

    Parameters
    ----------
    n : int
        Order of polynomials. Number of terms will be one higher.

    Returns
    -------
    (n + 1, n + 1) array
        Matrix, where each row has the coefficients of the polynomial
        basis in from the constant term, to the :math:`t^n` term.
    """
    assert n & 1 == 1
    a = np.zeros((n + 1, n + 1), np.float64)
    f = 1.0
    k = np.ones(n + 1)
    for r in range((n + 1) // 2):
        a[2 * r + 0, r] = f
        a[2 * r + 1, :] = k
        f *= r + 1
        k[r:] *= np.arange(n + 1 - r)
    return np.linalg.inv(a).T


def _nodal_interpolating_basis(n: int) -> tuple[Polynomial1D, ...]:
    """Create interpolating basis for nodal spline with degree n.

    Parameters
    ----------
    n : int
        Order of polynomials to use. Number of terms will be one more.

    Returns
    -------
    tuple of Polynomial1D
        Tuple of basis polynomials for nodal interpolation.
    """
    assert n & 1 == 1
    m = _nodal_polynomial_coefficients(n)

    # re-order rows of m
    def _permute_gen(n: int) -> Generator[int, None, None]:
        """Correctly order the rows."""
        yield 0  # Left nodal
        yield 1  # Right nodal
        for i in range(1, (n + 1) // 2):
            yield 2 * i  # Remaining left basis
        for i in range(1, (n + 1) // 2):
            yield 2 * i + 1  # Remaining right basis

    return tuple(Polynomial1D(m[i, :]) for i in _permute_gen(n))


def _nodal_interpolation_system(
    n: int,
    basis: Iterable[Polynomial1D],
    nds: npt.ArrayLike,
    bc_left: Iterable[SplineBC],
    bc_right: Iterable[SplineBC],
) -> tuple[sp.csr_array, npt.NDArray[np.float64]]:
    """Create interpolation matrix for natural nodal spline.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    nds : (M,) array_like
        Nodal values which are to be interpolated.
    bc_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    (M * (n - 1)/2, (M + 1) * (n - 1)/2) csr_array
        System matrix which gives coefficients required to construct the spine.
    (M * (n - 1)/2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 1
    values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        basis_values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        values.append(basis_values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.stack(values, axis=0)
    nodes = np.array(nds, np.float64)
    assert len(nodes.shape) == 1
    nelem = int(nodes.shape[0]) - 1
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)

    n_node = (n - 1) // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = sp.lil_array(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n - 1)
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f" coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n - 1)

        for j in range(1, n):
            coeffs += bc.coefficients[j] * v[j, 2:, 0]
            k[p] -= bc.coefficients[j] * (v[j, 0, 0] * nodes[0] + v[j, 1, 0] * nodes[1])
        m[p, inode] = coeffs
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = np.arange(n - 1)
        iright = n_node + np.arange(n - 1)
        for j in range(n_node + 1, n):
            coeffs = np.zeros(3 * n_node)
            coeffs[ileft] -= v[j, 2:, 1]
            coeffs[iright] += v[j, 2:, 0]
            m[p, n_node * i + np.arange(coeffs.shape[0])] = coeffs
            k[p] = (
                v[j, 0, 1] * nodes[i]
                + v[j, 1, 1] * nodes[i + 1]
                - v[j, 0, 0] * nodes[i + 1]
                - v[j, 1, 0] * nodes[i + 2]
            )
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        ind = np.arange(n - 1) + (nelem - 1) * n_node
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n - 1)

        for j in range(1, n):
            coeffs += bc.coefficients[j] * v[j, 2:, 1]
            k[p] -= bc.coefficients[j] * (v[j, 0, 1] * nodes[-2] + v[j, 1, 1] * nodes[-1])
        m[p, ind] = coeffs
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m.tocsr(), k


def nodal_interpolating_splinei(
    n: int,
    nds: npt.ArrayLike,
    bcs_left: Iterable[SplineBC] | None = None,
    bcs_right: Iterable[SplineBC] | None = None,
) -> Spline1Di:
    """Create interpolating spline, which has specified nodal values.

    Parameters
    ----------
    n : int
        Order of the spline. Must be odd, otherwise a ``ValueError`` will be raised.
    nds : (N,) array_like
        Nodes to hit.
    bcs_left : Iterable of SplineBC, optional
        Boundary conditions of the left side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.
    bcs_right : Iterable of SplineBC, optional
        Boundary conditions of the right side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.

    Returns
    -------
    Spline1Di
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    bcs_left, bcs_right = _ensure_boundary_conditions(n, bcs_left, bcs_right)
    if (n & 1) != 1:
        raise ValueError(f"Spline order must be odd (instead it was {n}).")
    nodes = np.array(nds, np.float64)
    if len(nodes.shape) != 1:
        raise ValueError(
            f"Nodes should be a 1D array, instead they have the shape {nodes.shape}"
        )
    basis = _nodal_interpolating_basis(n)
    m, k = _nodal_interpolation_system(n, basis, nodes, bcs_left, bcs_right)
    r = sla.spsolve(m, k)
    poly: list[Polynomial1D] = []
    for i in range(nodes.shape[0] - 1):
        p = (
            basis[0] * float(nodes[i])
            + basis[1] * float(nodes[i + 1])
            + sum(basis[j + 2] * r[(n - 1) // 2 * i + j] for j in range(n - 1))
        )
        poly.append(p)
    spl = Spline1Di(tuple(p.coefficients for p in poly))
    return spl


def _nodal_interpolation_system2(
    n: int,
    basis: Iterable[Polynomial1D],
    vals: npt.ArrayLike,
    nds: npt.ArrayLike,
    bc_left: Iterable[SplineBC],
    bc_right: Iterable[SplineBC],
) -> tuple[sp.csr_array, npt.NDArray[np.float64]]:
    """Create interpolation matrix for natural nodal spline.

    Parameters
    ----------
    n : int
        Order of polynomial basis used. Number of terms will be one more.
    basis : Iterable of Polynomial1D
        Basis functions used for the spline.
    vals : (M,) array_like
        Values of function at the nodes.
    nds : (M,) array_like
        Nodes where the function is defined.
    bc_left : Iterable of SplineBoundaryCondition
        Boundary conditions of the left side of the spline (must be of order ``n``).
    bc_right : Iterable of SplineBoundaryCondition
        Boundary conditions of the right side of the spline (must be of order ``n``).

    Returns
    -------
    (M * (n - 1)/2, (M + 1) * (n - 1)/2) csr_array
        System matrix which gives coefficients required to construct the spine.
    (M * (n - 1)/2,) array
        Right side of the equation, which can be solved to find the coefficients.
    """
    assert n & 1 == 1
    basis_values: list[npt.NDArray[np.float64]] = []
    polynomials = tuple(basis)
    for i in range(n):
        values = np.array(tuple(poly([0, 1]) for poly in polynomials), np.float64)
        basis_values.append(values)
        polynomials = tuple(poly.derivative for poly in polynomials)
    v = np.stack(basis_values, axis=0)
    del basis_values
    values = np.array(vals, np.float64)
    x = np.array(nds, np.float64)
    assert len(x.shape) == 1
    nelem = int(x.shape[0]) - 1
    # Contents of v:
    # axis 0: derivative
    # axis 1: basis
    # axis 2: node (left, right)
    dx = x[1:] - x[:-1]
    n_node = (n - 1) // 2
    k = np.zeros(((nelem + 1) * n_node))
    m = sp.lil_matrix(((nelem + 1) * n_node, (nelem + 1) * n_node))
    p = 0
    n_bc_left = 0
    for bc in bc_left:
        inode = np.arange(n - 1)
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Left boundary condition at index {n_bc_left} did not have enough"
                f" coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )

        coeffs = np.zeros(n - 1)

        for j in range(1, n):
            coeffs += (
                bc.coefficients[j]
                * v[j, 2:, 0]
                * (dx[0] ** np.tile(np.arange(n // 2) + 1, 2))
                / dx[0] ** j
            )
            k[p] -= bc.coefficients[j] * (v[j, 0, 0] * values[0] + v[j, 1, 0] * values[1])
        m[p, inode] = coeffs
        k[p] += bc.value
        n_bc_left += 1
        p += 1

    for i in range(0, nelem - 1):
        ileft = np.arange(n - 1)
        iright = n_node + np.arange(n - 1)
        for j in range(n_node + 1, n):
            coeffs = np.zeros(3 * n_node)
            coeffs[ileft] -= (
                v[j, 2:, 1] * (dx[i] ** np.tile(np.arange(n // 2) + 1, 2)) / dx[i] ** j
            )
            coeffs[iright] += (
                v[j, 2:, 0]
                * (dx[i + 1] ** np.tile(np.arange(n // 2) + 1, 2))
                / dx[i + 1] ** j
            )
            m[p, n_node * i + np.arange(coeffs.shape[0])] = coeffs
            k[p] = (v[j, 0, 1] * values[i] + v[j, 1, 1] * values[i + 1]) / dx[i] ** j - (
                v[j, 0, 0] * values[i + 1] + v[j, 1, 0] * values[i + 2]
            ) / dx[i + 1] ** j
            p += 1

    n_bc_right = 0
    for bc in bc_right:
        ind = np.arange(n - 1) + (nelem - 1) * n_node
        if bc.coefficients.shape != (n,):
            raise ValueError(
                f"Right boundary condition at index {n_bc_right} did not have enough"
                f"coefficients (got {bc.coefficients.shape} when expecting"
                f" {(n,)})."
            )
        coeffs = np.zeros(n - 1)

        for j in range(1, n):
            coeffs += (
                bc.coefficients[j]
                * v[j, 2:, 1]
                * (dx[-1] ** np.tile(np.arange(n // 2) + 1, 2))
                / dx[-1] ** j
            )
            k[p] -= bc.coefficients[j] * (
                v[j, 0, 1] * values[-2] + v[j, 1, 1] * values[-1]
            )
        m[p, ind] = coeffs
        k[p] += bc.value
        n_bc_right += 1
        p += 1

    if p != (nelem + 1) * n_node:
        raise ValueError(
            f"There was not enough boundary conditions specified ({n} were needed, but"
            f" {n_bc_left} were given on the left and {n_bc_right} were given on the"
            " right)."
        )

    return m.tocsr(), k


def nodal_interpolating_spline(
    n: int,
    vals: npt.ArrayLike,
    nds: npt.ArrayLike,
    bcs_left: Iterable[SplineBC] | None = None,
    bcs_right: Iterable[SplineBC] | None = None,
) -> Spline1D:
    """Create interpolating spline, which has specified nodal values.

    Parameters
    ----------
    n : int
        Order of the spline. Must be odd, otherwise a ``ValueError`` will be raised.
    vals : (N,) array_like
        Values of function at the nodes.
    nds : (N,) array_like
        Nodes where the function is defined.
    bcs_left : Iterable of SplineBC, optional
        Boundary conditions of the left side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.
    bcs_right : Iterable of SplineBC, optional
        Boundary conditions of the right side of the spline (must be of order ``n``).
        If it is not provided, the natural boundary conditions are used.

    Returns
    -------
    Spline1D
        Interpolating spline which maps from computational space :math:`[0, N-1]` to
        values.
    """
    bcs_left, bcs_right = _ensure_boundary_conditions(n, bcs_left, bcs_right)
    if (n & 1) != 1:
        raise ValueError(f"Spline order must be odd (instead it was {n}).")
    nodes = np.array(nds, np.float64)
    if len(nodes.shape) != 1:
        raise ValueError(
            f"Nodes should be a 1D array, instead they have the shape {nodes.shape}"
        )
    values = np.array(vals, np.float64)
    if values.shape != nodes.shape:
        raise ValueError(
            f"Values don't have the same shape as the nodes (nodes are {nodes.shape}, but"
            f"values are {values.shape})."
        )
    basis = _nodal_interpolating_basis(n)
    m, k = _nodal_interpolation_system2(n, basis, vals, nodes, bcs_left, bcs_right)
    r = sla.spsolve(m, k)
    poly: list[Polynomial1D] = []
    new_basis = tuple(
        b.scale_by(2).offset_by(+1) * 2 ** (i // 2) for i, b in enumerate(basis)
    )

    for i in range(nodes.shape[0] - 1):
        # Make into local basis
        p = (
            new_basis[0] * float(values[i])
            + new_basis[1] * float(values[i + 1])
            + sum(new_basis[j + 2] * (r[(n - 1) // 2 * i + j]) for j in range(n - 1))
        )
        poly.append(p)
    spl = Spline1D(nodes, tuple(p.coefficients for p in poly))
    return spl
