"""Package dedicated to interpolation using data defined on different topologies."""

# C base types
from interplib._interp import DEFAULT_BASIS_REGISTRY as DEFAULT_BASIS_REGISTRY
from interplib._interp import DEFAULT_INTEGRATION_REGISTRY as DEFAULT_INTEGRATION_REGISTRY
from interplib._interp import BasisRegistry as BasisRegistry
from interplib._interp import BasisSpecs as BasisSpecs
from interplib._interp import CoordinateMap as CoordinateMap
from interplib._interp import DegreesOfFreedom as DegreesOfFreedom
from interplib._interp import FunctionSpace as FunctionSpace
from interplib._interp import IntegrationRegistry as IntegrationRegistry
from interplib._interp import IntegrationSpace as IntegrationSpace
from interplib._interp import IntegrationSpecs as IntegrationSpecs
from interplib._interp import bernstein1d as bernstein1d
from interplib._interp import bernstein_coefficients as bernstein_coefficients
from interplib._interp import compute_gll as compute_gll
from interplib._interp import compute_mass_matrix as compute_mass_matrix
from interplib._interp import dlagrange1d as dlagrange1d
from interplib._interp import lagrange1d as lagrange1d

# DoFs functions
from interplib.degrees_of_freedom import (
    compute_dual_degrees_of_freedom as compute_dual_degrees_of_freedom,
)
from interplib.degrees_of_freedom import reconstruct as reconstruct

# Domains
from interplib.domains import Line as Line
from interplib.domains import Quad as Quad

# Enum types
from interplib.enum_type import BasisType as BasisType
from interplib.enum_type import IntegrationMethod as IntegrationMethod

# Integration functions
from interplib.integration import integrate_callable as integrate_callable
from interplib.integration import projection_l2_dual as projection_l2_dual
from interplib.integration import projection_l2_primal as projection_l2_primal

# C wrapper functions
from interplib.lagrange import (
    interp1d_2derivative_samples as interp1d_2derivative_samples,
)
from interplib.lagrange import (
    interp1d_derivative_samples as interp1d_derivative_samples,
)
from interplib.lagrange import interp1d_function_samples as interp1d_function_samples
from interplib.lagrange import (
    lagrange_2derivative_samples as lagrange_2derivative_samples,
)
from interplib.lagrange import (
    lagrange_derivative_samples as lagrange_derivative_samples,
)
from interplib.lagrange import lagrange_function_samples as lagrange_function_samples
