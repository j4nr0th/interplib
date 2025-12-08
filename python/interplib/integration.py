"""Functions to allow integration of callables."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from interplib._interp import (
    DEFAULT_INTEGRATION_REGISTRY,
    IntegrationRegistry,
    IntegrationSpace,
    SpaceMap,
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
    integration: IntegrationSpace | SpaceMap,
    /,
    registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
) -> float:
    """Integrate a callable over a specified integration space with given specs.

    Parameters
    ----------
    func : Callable
        The function to integrate. The function should be defined on the space it will
        be integrated on.

    integration_space : IntegrationSpace or SpaceMap
        The space over which to integrate the function or the mapping between the
        integration domain, which is an :math:`N`-dimensional :math:`[-1, +1]` hypercube,
        and the physical domain.

    registry : IntegrationRegistry, optional
        The registry to use for obtaining the integrator. If None, the default registry is
        used.

    Returns
    -------
    float
        The result of the integration.
    """
    match integration:
        case IntegrationSpace() as int_space:
            nodes = int_space.nodes(registry)
            weights = int_space.weights(registry)
        case SpaceMap() as smap:
            int_space = smap.integration_space
            weights = int_space.weights(registry) * smap.determinant
            nodes = np.array(
                [
                    smap.coordinate_map(idim).values
                    for idim in range(smap.output_dimensions)
                ]
            )
        case _:
            raise TypeError(
                f"Only {IntegrationSpace} or {SpaceMap} can be passed, but instead "
                f"{type(integration)} was passed."
            )

    return float(
        np.sum(
            np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
        )
    )
