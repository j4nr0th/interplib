"""Submodule to deal with different differentional forms."""

# Boundary conditions
from interplib.kforms.boundary import (
    BoundaryCondition1D as BoundaryCondition1D,
)
from interplib.kforms.boundary import (
    BoundaryCondition1DStrong as BoundaryCondition1DStrong,
)
from interplib.kforms.boundary import (
    BoundaryCondition1DWeak as BoundaryCondition1DWeak,
)

# Basic K-forms
from interplib.kforms.kform import KForm as KForm
from interplib.kforms.kform import KFormDerivative as KFormDerivative
from interplib.kforms.kform import KFormDual as KFormDual
from interplib.kforms.kform import KFormEquaton as KFormEquaton
from interplib.kforms.kform import KFormInnerProduct as KFormInnerProduct
from interplib.kforms.kform import KFormPrimal as KFormPrimal
from interplib.kforms.kform import KFormProjection as KFormProjection
from interplib.kforms.kform import KFormSum as KFormSum
from interplib.kforms.kform import KFormSystem as KFormSystem
from interplib.kforms.kform import element_system as element_system
