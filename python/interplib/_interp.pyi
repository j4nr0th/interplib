from __future__ import annotations

import numpy as np
import numpy.typing as npt

def test() -> str: ...
def lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def dlagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def d2lagrange1d(
    x: npt.NDArray[np.float64], xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def hermite(
    x: npt.NDArray[np.float64],
    bc1: tuple[float, float, float],
    bc2: tuple[float, float, float],
) -> npt.NDArray[np.float64]: ...
def bernstein1d(n: int, x: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

class Basis1D:
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Basis1D: ...
    @property
    def antiderivative(self) -> Basis1D: ...

class Polynomial1D(Basis1D):
    def __init__(self, coefficients: npt.ArrayLike, /) -> None: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    def __add__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __radd__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __neg__(self) -> Polynomial1D: ...
    def __mul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __rmul__(self, other: Polynomial1D | float) -> Polynomial1D: ...
    def __pos__(self) -> Polynomial1D: ...
    def __pow__(self, i: int) -> Polynomial1D: ...
    @property
    def derivative(self) -> Polynomial1D: ...
    @property
    def antiderivative(self) -> Polynomial1D: ...
    @property
    def order(self) -> int: ...
    # def __len__(self) -> int: ...
    # def __getitem__(self, key: int) -> np.float64: ...
    def __setitem__(self, key: int, value: float | np.floating) -> None: ...
    # def __iter__(self) -> Iterator[np.float64]: ...
    @classmethod
    def lagrange_nodal_basis(cls, nodes: npt.ArrayLike) -> tuple[Polynomial1D, ...]: ...
    @classmethod
    def lagrange_nodal_fit(
        cls, nodes: npt.ArrayLike, values: npt.ArrayLike
    ) -> Polynomial1D: ...
    def offset_by(self, x: float | np.floating) -> Polynomial1D: ...

class Spline1D(Basis1D):  # TODO: implement other methods of Poylnomial1D
    def __init__(self, nodes: npt.ArrayLike, coefficients: npt.ArrayLike, /) -> None: ...
    @property
    def nodes(self) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Spline1D: ...
    @property
    def antiderivative(self) -> Spline1D: ...

class Spline1Di(Spline1D):  # TODO: implement other methods of Poylnomial1D
    def __init__(self, coefficients: npt.ArrayLike, /) -> None: ...
    @property
    def nodes(self) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]: ...
    @property
    def derivative(self) -> Spline1D: ...
    @property
    def antiderivative(self) -> Spline1D: ...
