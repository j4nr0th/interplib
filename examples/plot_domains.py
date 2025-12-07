"""
Domain Examples
===============

This example demonstrates how different domains look and behave like.
"""  # noqa: D205 D400

import numpy as np
from interplib import Line, Quad
from matplotlib import pyplot as plt

# %%
#
# Lines
# -----
#


ln1 = Line((-1, -1), (+1, +3))
ln2 = Line((-1, -1), (0, 0), (+1, +3))
ln3 = Line((-1, -1), (0, 0), (+2, +3), (+1, +3))

fig, ax = plt.subplots()

xplt = np.linspace(-1, +1, 101)
ax.scatter(ln3.knots[:, 0], ln3.knots[:, 1], label="3", color="blue")
ax.scatter(ln2.knots[:, 0], ln2.knots[:, 1], label="2", color="green")
ax.scatter(ln1.knots[:, 0], ln1.knots[:, 1], label="1", color="red")
ax.plot(*ln3.sample(xplt), color="blue")
ax.plot(*ln2.sample(xplt), color="green")
ax.plot(*ln1.sample(xplt), color="red")
ax.legend()
ax.set(aspect="equal")
fig.tight_layout()
plt.show()


# %%
#
# Surfaces
# --------

bottom = Line((-1, -1), (+1, -1))
right = Line((+1, -1), (+2, 0), (+1, +1))
top = Line((+1, +1), (-1, +1))
left = Line((-1, +1), (-1, -1))

quad = Quad(bottom, right, top, left)

fig, ax = plt.subplots()
xp, yp = np.meshgrid(np.linspace(-1, +1, 11), np.linspace(-1, +1, 11))

ax.plot(*bottom.sample(xplt), color="blue", label="bottom")
ax.plot(*right.sample(xplt), color="blue", label="right")
ax.plot(*top.sample(xplt), color="blue", label="top")
ax.plot(*left.sample(xplt), color="blue", label="left")

ax.scatter(*quad.sample(xp, yp), color="red")
ax.set(aspect="equal")

ax.legend()
fig.tight_layout()
plt.show()
