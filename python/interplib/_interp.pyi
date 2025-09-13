from __future__ import annotations

from collections.abc import Sequence
from typing import Self, final

import numpy as np
import numpy.typing as npt

from interplib.enum_type import _BasisTypeHint, _IntegrationMethodHint

def lagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials, defined by

    .. math::

       \mathcal{L}^n_i (x) = \prod\limits_{j=1, j \neq i}^{n} \frac{x - x_j}{x_i - x_j},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_1, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial values at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate Lagrange
    polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import lagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = lagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is lagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = lagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"$\\mathcal{{L}}^{{{order}}}_{{{i + 1}}}$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def dlagrange1d(
    roots: npt.ArrayLike, x: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
) -> npt.NDArray[np.double]:
    r"""Evaluate derivatives of Lagrange polynomials.

    This function efficiently evaluates Lagrange basis polynomials derivatives, defined by

    .. math::

       \frac{d \mathcal{L}^n_i (x)}{d x} =
       \sum\limits_{j=0,j \neq i}^n \prod\limits_{k=0, k \neq i, k \neq j}^{n}
       \frac{1}{x_i - x_j} \cdot \frac{x - x_k}{x_i - x_k},

    where the ``roots`` specifies the zeros of the Polynomials
    :math:`\{x_0, \dots, x_n\}`.

    Parameters
    ----------
    roots : array_like
       Roots of Lagrange polynomials.
    x : array_like
       Points where the derivatives of polynomials should be evaluated.
    out : array, optional
       Array where the results should be written to. If not given, a new one will be
       created and returned. It should have the same shape as ``x``, but with an extra
       dimension added, the length of which is ``len(roots)``.

    Returns
    -------
    array
       Array of Lagrange polynomial derivatives at positions specified by ``x``.

    Examples
    --------
    This example here shows the most basic use of the function to evaluate derivatives of
    Lagrange polynomials. First, let us define the roots.

    .. jupyter-execute::

        >>> import numpy as np
        >>>
        >>> order = 7
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))

    Next, we can evaluate the polynomials at positions. Here the interval between the
    roots is chosen.

    .. jupyter-execute::

        >>> from interplib import dlagrange1d
        >>>
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 128)
        >>> yvals = dlagrange1d(roots, xpos)

    Note that if we were to give an output array to write to, it would also be the
    return value of the function (as in no copy is made).

    .. jupyter-execute::

        >>> yvals is dlagrange1d(roots, xpos, yvals)
        True

    Now we can plot these polynomials.

    .. jupyter-execute::

        >>> from matplotlib import pyplot as plt
        >>>
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.show()

    Accuracy is retained even at very high polynomial order. The following
    snippet shows that even at absurdly high order of 51, the results still
    have high accuracy and don't suffer from rounding errors. It also performs
    well (in this case, the 52 polynomials are each evaluated at 1025 points).

    .. jupyter-execute::

        >>> from time import perf_counter
        >>> order = 51
        >>> roots = - np.cos(np.linspace(0, np.pi, order + 1))
        >>> xpos = np.linspace(np.min(roots), np.max(roots), 1025)
        >>> t0 = perf_counter()
        >>> yvals = dlagrange1d(roots, xpos)
        >>> t1 = perf_counter()
        >>> print(f"Calculations took {t1 - t0: e} seconds.")
        >>> plt.figure()
        >>> for i in range(order + 1):
        ...     plt.plot(
        ...         xpos,
        ...         yvals[..., i],
        ...         label=f"${{\\mathcal{{L}}^{{{order}}}_{{{i}}}}}^\\prime$"
        ...     )
        >>> plt.gca().set(
        ...     xlabel="$x$",
        ...     ylabel="$y$",
        ...     title=f"Lagrange polynomials of order {order}"
        ... )
        >>> # plt.legend() # No, this is too long
        >>> plt.grid()
        >>> plt.show()
    """
    ...

def d2lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute Bernstein polynomials of given order at given locations.

    Parameters
    ----------
    n : int
       Order of polynomials used.
    x : (M,) array_like
       Flat array of locations where the values should be interpolated.

    Returns
    -------
    (M, n) arr
       Matrix containing values of Bernstein polynomial :math:`B^M_j(x_i)` as the
       element ``array[i, j]``.
    """
    ...

def bernstein_coefficients(x: npt.ArrayLike, /) -> npt.NDArray[np.double]:
    """Compute Bernstein polynomial coefficients from a power series polynomial.

    Parameters
    ----------
    x : array_like
       Coefficients of the polynomial from 0-th to the highest order.

    Returns
    -------
    array
       Array of coefficients of Bernstein polynomial series.
    """
    ...

def compute_gll(
    order: int, max_iter: int = 10, tol: float = 1e-15
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Gauss-Legendre-Lobatto integration nodes and weights.

    If you are often re-using these, consider caching them.

    Parameters
    ----------
    order : int
       Order of the scheme. The number of node-weight pairs is one more.
    max_iter : int, default: 10
       Maximum number of iterations used to further refine the values.
    tol : float, default: 1e-15
       Tolerance for stopping the refinement of the nodes.

    Returns
    -------
    array
       Array of ``order + 1`` integration nodes on the interval :math:`[-1, +1]`.
    array
       Array of integration weights which correspond to the nodes.
    """
    ...
@final
class IntegrationRule:
    """Type that describes an integration rule.

    Parameters
    ----------
    order : int
        Order of the integration rule.

    method : interplib.IntegrationMethod, default: "gauss"
        Method used for integration.
    """

    def __new__(cls, order: int, /, method: _IntegrationMethodHint = "gauss") -> Self: ...
    @property
    def order(self) -> int:
        """Order of the integration rule."""
        ...

    @property
    def accuracy(self) -> int:
        """Highest order of polynomial that is integrated exactly."""
        ...

    @property
    def nodes(self) -> npt.NDArray[np.double]:
        """Integration points on the reference domain."""
        ...

    @property
    def weights(self) -> npt.NDArray[np.double]:
        """Weights associated with the integration nodes."""
        ...

    @property
    def pointer(self) -> int:
        """Pointer of the integration rule."""
        ...

@final
class BasisSet:
    """Type that describes a set of basis functions.

    Parameters
    ----------
    basis_type : interplib._typing.BasisType
        Type of the basis used for the set.

    order : int
        Order of the basis in the set.

    integration_rule : IntegrationRule
        Integration rule used with the basis set.
    """

    def __new__(
        cls, basis_type: _BasisTypeHint, order: int, integration_rule: IntegrationRule, /
    ) -> Self: ...
    @property
    def values(self) -> npt.NDArray[np.double]:
        """Values of all basis at integration points."""
        ...

    @property
    def order(self) -> int:
        """Order of the basis set."""
        ...

    @property
    def pointer(self) -> int:
        """Pointer of the basis set."""
        ...

@final
class GeoID:
    """Type used to identify a geometrical object with an index and orientation.

    Parameters
    ----------
    index : int
        Index of the geometrical object.
    reversed : any, default: False
        The object's orientation should be reversed.
    """

    def __new__(cls, index: int, reverse: object = False) -> Self: ...
    @property
    def index(self) -> int:
        """Index of the object referenced by id."""
        ...
    @property
    def reversed(self) -> bool:
        """Is the orientation of the object reversed."""
        ...

    def __bool__(self) -> bool: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __neg__(self) -> GeoID: ...

@final
class Line:
    """Geometrical object, which connects two points.

    Parameters
    ----------
    begin : GeoID or int
        ID of the point where the line beings.
    end : GeoID or int
        ID of the point where the line ends.
    """

    def __new__(cls, begin: GeoID | int, end: GeoID | int) -> Self: ...
    @property
    def begin(self) -> GeoID:
        """ID of the point where the line beings."""
        ...
    @property
    def end(self) -> GeoID:
        """ID of the point where the line ends."""
        ...

    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Surface:
    """Two dimensional geometrical object, which is bound by lines.

    Parameters
    ----------
    *ids : GeoID or int
        Ids of the lines which are the boundary of the surface.
    """

    def __new__(cls, *ids: GeoID | int) -> Self: ...
    def __array__(self, dtype=None, copy=None) -> npt.NDArray: ...
    def __getitem__(self, idx: int) -> GeoID: ...
    def __len__(self) -> int: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Manifold:
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

@final
class Manifold1D(Manifold):
    """One dimensional manifold."""

    @property
    def n_lines(self) -> int:
        """Number of lines in the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the manifold."""
        ...

    def get_line(self, index: GeoID | int, /) -> Line:
        """Get the line of the specified ID."""
        ...

    def find_line(self, line: Line) -> GeoID:
        """Find the ID of the specified line."""
        ...

    @classmethod
    def line_mesh(cls, segments: int, /) -> Manifold1D:
        """Create a new Manifold1D which represents a line.

        Parameters
        ----------
        segments : int
            Number of segments the line is split into. There will be one more point.

        Returns
        -------
        Manifold1D
            Manifold that represents the topology of the line.
        """
        ...

    def compute_dual(self) -> Manifold1D:
        """Compute the dual to the manifold.

        Returns
        -------
        Manifold1D
            The dual to the manifold.
        """
        ...

class Manifold2D(Manifold):
    """A manifold of a finite number of dimensions."""

    @property
    def dimension(self) -> int:
        """Dimension of the manifold."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        ...

    @property
    def n_lines(self) -> int:
        """Number of lines in the mesh."""
        ...

    @property
    def n_surfaces(self) -> int:
        """Number of surfaces in the mesh."""
        ...

    def get_line(self, index: int | GeoID, /) -> Line:
        """Get the line from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the line to get in 1-based indexing or GeoID. If negative, the
           orientation will be reversed.

        Returns
        -------
        Line
           Line object corresponding to the ID.
        """
        ...

    def get_surface(self, index: int | GeoID, /) -> Surface:
        """Get the surface from the mesh.

        Parameters
        ----------
        index : int or GeoID
           Id of the surface to get in 1-based indexing or GeoID. If negative,
           the orientation will be reversed.

        Returns
        -------
        Surface
           Surface object corresponding to the ID.
        """

    @classmethod
    def from_irregular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with non-constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : Sequence of array_like
            Sequence of arrays specifying connectivity of mesh surfaces in 1-based
            indexing, where a negative value means that the line's orientation is
            reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    @classmethod
    def from_regular(
        cls,
        n_points: int,
        line_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
        surface_connectivity: Sequence[npt.ArrayLike] | npt.ArrayLike,
    ) -> Self:
        """Create Manifold2D from surfaces with constant number of lines.

        Parameters
        ----------
        n_points : int
            Number of points in the mesh.
        line_connectivity : (N, 2) array_like
            Connectivity of points which form lines in 0-based indexing.
        surface_connectivity : array_like
            Two dimensional array-like object specifying connectivity of mesh
            surfaces in 1-based indexing, where a negative value means that
            the line's orientation is reversed.

        Returns
        -------
        Self
            Two dimensional manifold.
        """
        ...

    def compute_dual(self) -> Manifold2D:
        """Compute the dual to the manifold.

        A dual of each k-dimensional object in an n-dimensional space is a
        (n-k)-dimensional object. This means that duals of surfaces are points,
        duals of lines are also lines, and that the duals of points are surfaces.

        A dual line connects the dual points which correspond to surfaces which
        the line was a part of. Since the change over a line is computed by
        subtracting the value at the beginning from that at the end, the dual point
        which corresponds to the primal surface where the primal line has a
        positive orientation is the end point of the dual line and conversely the end
        dual point is the one corresponding to the surface which contained the primal
        line in the negative orientation. Since lines may only be contained in a
        single primal surface, they may have an invalid ID as either their beginning or
        their end. This can be used to determine if the line is actually a boundary of
        the manifold.

        A dual surface corresponds to a point and contains dual lines which correspond
        to primal lines, which contained the primal point of which the dual surface is
        the result of. The orientation of dual lines in this dual surface is positive if
        the primal line of which they are duals originated in the primal point in question
        and negative if it was their end point.

        Returns
        -------
        Manifold2D
            Dual manifold.
        """
        ...

    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
