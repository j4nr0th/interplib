"""Functions to allow integration of callables."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from interplib._interp import (
    DEFAULT_BASIS_REGISTRY,
    DEFAULT_INTEGRATION_REGISTRY,
    BasisRegistry,
    DegreesOfFreedom,
    FunctionSpace,
    IntegrationRegistry,
    IntegrationSpace,
    SpaceMap,
    compute_mass_matrix,
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


def _prepare_integration(
    integration: IntegrationSpace | SpaceMap, registry: IntegrationRegistry
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Prepare nodes and weights."""
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

    return nodes, weights


def integrate_callable(
    func: Integrable,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
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

    registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    Returns
    -------
    float
        The result of the integration.
    """
    nodes, weights = _prepare_integration(integration=integration, registry=registry)
    return float(
        np.sum(
            np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
        )
    )


def projection_l2(
    func: Integrable,
    function_space: FunctionSpace,
    integration: IntegrationSpace | SpaceMap,
    /,
    *,
    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY,
    basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,
) -> DegreesOfFreedom:
    """Compute the L2 projection of the function on the function space.

    Parameters
    ----------
    func : Integratable
        Function to project. It has to be possible to integrate it.

    function_space : FunctionSpace
        Function space on which to project the function.

    integration : IntegrationSpace or SpaceMap
        Specification of the integration domain.

    integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY
        The registry to use for obtaining the integrator.

    basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY
        The registry to use for obtaining the basis values.

    Returns
    -------
    DegreesOfFreedom
        Primal degrees of freedom of the projection.
    """
    nodes, weights = _prepare_integration(
        integration=integration, registry=integration_registry
    )

    func_vals = (
        np.asarray(func(*[nodes[i, ...] for i in range(nodes.shape[0])])) * weights
    )
    del nodes, weights, func

    int_space: IntegrationSpace
    match integration:
        case IntegrationSpace():
            int_space = integration
        case SpaceMap():
            int_space = integration.integration_space
        case _:
            assert False

    basis_values = function_space.values_at_integration_nodes(
        int_space,
        integration_registry=integration_registry,
        basis_registry=basis_registry,
    )
    del integration_registry, basis_registry

    func_vals = func_vals.flatten()
    basis_values = basis_values.reshape((func_vals.size, -1))
    dual_dofs = np.sum(func_vals[:, None] * basis_values, axis=0)
    del func_vals, basis_values

    mass_matrix = compute_mass_matrix(function_space, function_space, integration)
    del integration

    primal_dofs = np.linalg.solve(mass_matrix, dual_dofs)
    del dual_dofs, mass_matrix
    return DegreesOfFreedom(function_space, primal_dofs)
