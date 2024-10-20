"""Package dedicated to interpolation using data defined on different topological objects."""

from interplib._interp import test as test
from interplib.hermite import HermiteSpline as HermiteSpline
from interplib.hermite import SplineBC as SplineBC
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
from interplib.rbf import SIRBF as SIRBF
