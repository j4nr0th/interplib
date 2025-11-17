"""Package dedicated to interpolation using data defined on different topologies."""

# C base types
from interplib._interp import BasisSpecs as BasisSpecs
from interplib._interp import IntegrationSpecs as IntegrationSpecs
from interplib._interp import bernstein1d as bernstein1d
from interplib._interp import bernstein_coefficients as bernstein_coefficients
from interplib._interp import compute_gll as compute_gll
from interplib._interp import dlagrange1d as dlagrange1d
from interplib._interp import lagrange1d as lagrange1d

# Enum types
from interplib.enum_type import BasisType as BasisType
from interplib.enum_type import IntegrationMethod as IntegrationMethod

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
