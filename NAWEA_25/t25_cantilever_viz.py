import numpy as np
import matplotlib.pyplot as plt

from stablib.models.model3flap_optimized import cantilever_modes


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

L = 1.5
n_modes = 1

# Exaggerated flapwise deflection
deflection_scale = 0 * L

# Blade width
root_width = 0.16
tip_width = 0.06

# Number of points
n_points = 300


# ------------------------------------------------------------
# Mode shape
# ------------------------------------------------------------

modes = cantilever_modes(n_modes, L)

mode = modes[0]


# ------------------------------------------------------------
# Span
# ------------------------------------------------------------

r = np.linspace(0, L, n_points)

# Normalize mode
w_mode = mode(r)
w_mode /= np.max(np.abs(w_mode))

# Exaggerated deflection
w = deflection_scale * w_mode


# ------------------------------------------------------------
# Blade azimuths
# ------------------------------------------------------------

phi_blades = np.array([
    0.0,
    2.0 * np.pi / 3.0,
    4.0 * np.pi / 3.0
])


# ------------------------------------------------------------
# Blade width
# ------------------------------------------------------------

x = r / L

width = (
    root_width
    + (tip_width - root_width) * x
)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(
    111,
    projection='3d'
)

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

ax.grid(False)
# ------------------------------------------------------------
# Hub
# ------------------------------------------------------------

ax.scatter(
    0,
    0,
    0,
    s=100,
    color='black'
)


# ------------------------------------------------------------
# Blades
# ------------------------------------------------------------

for k, phi in enumerate(phi_blades):

    # --------------------------------------------------------
    # Local radial direction
    # --------------------------------------------------------

    er = np.array([
        np.cos(phi),
        np.sin(phi),
        0.0
    ])

    # Local tangential direction
    etheta = np.array([
        -np.sin(phi),
        np.cos(phi),
        0.0
    ])

    # --------------------------------------------------------
    # Undeformed centerline
    # --------------------------------------------------------

    centerline = np.outer(
        r,
        er
    )

    # --------------------------------------------------------
    # Deflected centerline
    # --------------------------------------------------------

    centerline_def = (
        centerline
        + np.outer(w, np.array([0.0, 0.0, 1.0]))
    )

    # --------------------------------------------------------
    # Blade surface
    # --------------------------------------------------------

    # Width direction
    left = (
        centerline_def
        - 0.5 * width[:, None] * etheta
    )

    right = (
        centerline_def
        + 0.5 * width[:, None] * etheta
    )

    # Surface coordinates
    X = np.column_stack([
        left[:, 0],
        right[:, 0]
    ])

    Y = np.column_stack([
        left[:, 1],
        right[:, 1]
    ])

    Z = np.column_stack([
        left[:, 2],
        right[:, 2]
    ])

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=0.85,
        linewidth=0
    )

    # --------------------------------------------------------
    # Undeformed centerline
    # --------------------------------------------------------

    ax.plot(
        centerline[:, 0],
        centerline[:, 1],
        centerline[:, 2],
        '--',
        linewidth=1,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Deflected centerline
    # --------------------------------------------------------

    ax.plot(
        centerline_def[:, 0],
        centerline_def[:, 1],
        centerline_def[:, 2],
        linewidth=2
    )


# ------------------------------------------------------------
# Hub disk
# ------------------------------------------------------------

theta = np.linspace(
    0,
    2 * np.pi,
    100
)

hub_radius = 0.15

ax.plot(
    hub_radius * np.cos(theta),
    hub_radius * np.sin(theta),
    np.zeros_like(theta),
    linewidth=3
)


# ------------------------------------------------------------
# View
# ------------------------------------------------------------

ax.view_init(
    elev=25,
    azim=45
)

ax.set_box_aspect(
    (1, 1, 0.6)
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_title(
    'Three-Bladed Rotor — First Flapwise Mode'
)

ax.grid(False)

plt.tight_layout()
plt.show()

