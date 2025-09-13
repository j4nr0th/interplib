"""
Basis Sets Visualization
========================

This example demonstrates how different basis sets look for polynomial orders.
We compare Lagrange, Legendre, and Bernstein-type bases using different node
distributions (uniform, Gauss, Gauss-Lobatto, Chebyshev-Gauss).

The basis values are sampled at the integration points (nodes), and the
functions are visualized on the reference interval [-1, 1].
"""  # noqa: D400 D205

# Assume classes are defined in interplib
from interplib import BasisSet, BasisType, IntegrationMethod, IntegrationRule
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# %%
# Helper function to plot a basis set
# -----------------------------------


def plot_basis_set(
    basis_type: BasisType, order: int, method: IntegrationMethod, ax: Axes
) -> None:
    """Plot basis in the set."""
    rule = IntegrationRule(10 * order, method=method)
    basis_set = BasisSet(basis_type, order, rule)

    x = rule.nodes
    v = basis_set.values

    for i in range(v.shape[0]):
        ax.plot(x, v[i, :], label=f"ϕ_{i}")
    ax.set_title(f"{basis_type.value}, order={order}")
    ax.set_xlim(-1, +1)
    ax.set_ylim(-1.2, +1.2)
    ax.legend()
    ax.grid(True)


##############################################################################
# Compare different basis types
# -----------------------------


fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

for ax, btype in zip(axes.flat, BasisType):
    plot_basis_set(BasisType(btype), order=4, method=IntegrationMethod.GAUSS, ax=ax)

fig.suptitle("Comparison of Basis Sets for Order=4", fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
