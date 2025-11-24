"""Functions to allow integration of callables."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from interplib._interp import (
    DEFAULT_INTEGRATION_REGISTRY,
    IntegrationRegistry,
    IntegrationSpace,
)


class Integrable(Protocol):
    """Protocol for integrable objects."""

    def __call__(
        self,
        *args: npt.NDArray[np.double],
    ) -> npt.ArrayLike:
        """Evaluate the integrable object at given points.

        Parameters
        ----------
        args : npt.NDArray[np.double]
            Coordinates at which to evaluate the integrable object. Each argument
            corresponds to one dimension.

        Returns
        -------
        npt.ArrayLike
            The evaluated values.
        """
        ...


def integrate_callable(
    func: Integrable,
    integration_space: IntegrationSpace,
    registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
) -> float:
    """Integrate a callable over a specified integration space with given specs.

    Parameters
    ----------
    func : Callable
        The function to integrate.
    integration_space : IntegrationSpace
        The space over which to integrate the function.
    registry : IntegrationRegistry | None, optional
        The registry to use for obtaining the integrator. If None, the default registry is
        used.

    Returns
    -------
    float
        The result of the integration.
    """
    nodes = integration_space.nodes(registry)
    weights = integration_space.weights(registry)
    return float(
        np.sum(
            np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
        )
    )
