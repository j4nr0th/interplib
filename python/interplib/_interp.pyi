from __future__ import annotations

import numpy as np
import numpy.typing as npt

def test() -> str:...

def lagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...

def dlagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...

def d2lagrange1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...

def hermite(
    x: npt.NDArray[np.float64],
    bc1: tuple[float, float, float],
    bc2: tuple[float, float, float],
) -> npt.NDArray[np.float64]:...


class Basis1D:

    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]:...

    def derivative(self) -> Basis1D:...

    def antiderivative(self) -> Basis1D:...


class Polynomial1D(Basis1D):

    def __init__(self, coefficients: npt.ArrayLike, /) -> None:...

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:...

    def __add__(self, other: Polynomial1D|float) -> Polynomial1D:...

    def __neg__(self) -> Polynomial1D:...

    def __mul__(self, other: Polynomial1D|float) -> Polynomial1D:...

    def derivative(self) -> Polynomial1D:...

    def antiderivative(self) -> Polynomial1D:...

class Spline1D: # TODO: implement other methos of Poylnomial1D

    def __init__(self, nodes: npt.ArrayLike, coefficients: npt.ArrayLike, /) -> None:...

    @property
    def nodes(self) -> npt.NDArray[np.float64]:...

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:...

    def __call__(self, x: npt.ArrayLike, /) -> npt.NDArray[np.float64]:...

    def derivative(self) -> Spline1D:...

    def antiderivative(self) -> Spline1D:...
