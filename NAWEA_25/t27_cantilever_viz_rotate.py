import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from stablib.models.model3flap_optimized import cantilever_modes


# ============================================================
# PARAMETERS
# ============================================================

L = 1.5
n_modes = 1

# Animation
duration = 10.0
fps = 20
n_frames = int(duration * fps)

# Rotor rotation rate [rev/s]
rotor_speed = 0.20

# Visual deflection amplification
deflection_scale = 0.45 * L

# Blade width
root_width = 0.16
tip_width = 0.06

# Number of points along blade
n_points = 100

# Blade color
blade_color = "steelblue"


# ============================================================
# CANTILEVER MODE
# ============================================================

modes = cantilever_modes(n_modes, L)
mode = modes[0]

r = np.linspace(0, L, n_points)

mode_shape = np.array([
    mode(ri) for ri in r
])

# Normalize mode shape
mode_shape /= np.max(np.abs(mode_shape))


# ============================================================
# BLADE WIDTH
# ============================================================

x = r / L

width = (
    root_width
    + (tip_width - root_width) * x
)


# ============================================================
# TIME
# ============================================================

t = np.linspace(
    0,
    duration,
    n_frames,
    endpoint=False
)

dt = t[1] - t[0]
fs = 1.0 / dt


# ============================================================
# SMOOTH PSEUDO-RANDOM FLAP MOTION
# ============================================================

rng = np.random.default_rng(12)

freqs = np.array([
    0.7,
    1.3,
    2.1,
    3.2
])

amps = rng.uniform(
    0.08,
    0.22,
    len(freqs)
)

phases = rng.uniform(
    0,
    2 * np.pi,
    len(freqs)
)

flap_signal = np.zeros_like(t)

for a, f, p in zip(
    amps,
    freqs,
    phases
):

    flap_signal += (
        a
        * np.sin(
            2 * np.pi * f * t + p
        )
    )


# Normalize
flap_signal /= np.max(
    np.abs(flap_signal)
)


# ------------------------------------------------------------
# Make motion one-sided
# ------------------------------------------------------------

flap_signal = (
    0.5
    * (flap_signal + 1.0)
)


# ============================================================
# FREQUENCY SPECTRUM
# ============================================================

signal_for_fft = (
    flap_signal
    - np.mean(flap_signal)
)

fft = np.fft.rfft(
    signal_for_fft
)

frequency = np.fft.rfftfreq(
    n_frames,
    dt
)

spectrum = np.abs(fft)

spectrum /= np.max(spectrum)


# ============================================================
# BLADE AZIMUTHS
# ============================================================

blade_offsets = np.array([
    0.0,
    2 * np.pi / 3,
    4 * np.pi / 3
])


# ============================================================
# ROTOR FIGURE
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# WHITE BACKGROUND
# ============================================================

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.xaxis.pane.set_facecolor("white")
ax.yaxis.pane.set_facecolor("white")
ax.zaxis.pane.set_facecolor("white")

ax.grid(False)


# ============================================================
# ROTOR AXIS / CAMERA
# ============================================================

lim = 1.15 * L

ax.set_xlim(
    -lim,
    lim
)

ax.set_ylim(
    -lim,
    lim
)

ax.set_zlim(
    -0.15 * L,
    0.65 * L
)

ax.set_box_aspect(
    (1, 1, 0.55)
)

# Make z appear horizontally in the view
ax.view_init(
    elev=20,
    azim=90
)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")

ax.set_title(
    "Flexible Three-Bladed Rotor"
)


# ============================================================
# CREATE INITIAL BLADES
# ============================================================

blade_surfaces = []
blade_centerlines = []


for phi0 in blade_offsets:

    phi = phi0

    # Radial direction
    er = np.array([
        np.cos(phi),
        np.sin(phi),
        0.0
    ])

    # Tangential direction
    etheta = np.array([
        -np.sin(phi),
        np.cos(phi),
        0.0
    ])

    # Undeformed centerline
    centerline = np.outer(
        r,
        er
    )

    # Initial deflection
    centerline_def = centerline.copy()

    # Blade edges
    left = (
        centerline_def
        - 0.5
        * width[:, None]
        * etheta
    )

    right = (
        centerline_def
        + 0.5
        * width[:, None]
        * etheta
    )

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

    # Blade surface
    surface = ax.plot_surface(
        X,
        Y,
        Z,
        color=blade_color,
        alpha=0.9,
        linewidth=0
    )

    blade_surfaces.append(
        surface
    )

    # Centerline
    centerline_plot, = ax.plot(
        centerline_def[:, 0],
        centerline_def[:, 1],
        centerline_def[:, 2],
        linewidth=1.5
    )

    blade_centerlines.append(
        centerline_plot
    )


# ============================================================
# HUB
# ============================================================

ax.scatter(
    0,
    0,
    0,
    s=100
)

theta = np.linspace(
    0,
    2 * np.pi,
    80
)

hub_radius = 0.15

ax.plot(
    hub_radius * np.cos(theta),
    hub_radius * np.sin(theta),
    np.zeros_like(theta),
    linewidth=2
)


# ============================================================
# ANIMATION UPDATE — ROTOR
# ============================================================

def update_rotor(frame):

    current_time = t[frame]

    # --------------------------------------------------------
    # Rotor rotation
    # --------------------------------------------------------

    rotor_angle = (
        2
        * np.pi
        * rotor_speed
        * current_time
    )

    # --------------------------------------------------------
    # Current flap
    # --------------------------------------------------------

    flap = flap_signal[frame]

    w = (
        deflection_scale
        * flap
        * mode_shape
    )

    # --------------------------------------------------------
    # Update each blade
    # --------------------------------------------------------

    for k, phi0 in enumerate(
        blade_offsets
    ):

        phi = (
            phi0
            + rotor_angle
        )

        # Radial direction
        er = np.array([
            np.cos(phi),
            np.sin(phi),
            0.0
        ])

        # Tangential direction
        etheta = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ])

        # Centerline
        centerline = np.outer(
            r,
            er
        )

        # Apply backward flap
        centerline_def = (
            centerline
            + np.outer(
                w,
                np.array([
                    0.0,
                    0.0,
                    1.0
                ])
            )
        )

        # ----------------------------------------------------
        # Blade edges
        # ----------------------------------------------------

        left = (
            centerline_def
            - 0.5
            * width[:, None]
            * etheta
        )

        right = (
            centerline_def
            + 0.5
            * width[:, None]
            * etheta
        )

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

        # ----------------------------------------------------
        # Replace blade surface
        # ----------------------------------------------------

        blade_surfaces[k].remove()

        blade_surfaces[k] = (
            ax.plot_surface(
                X,
                Y,
                Z,
                color=blade_color,
                alpha=0.9,
                linewidth=0
            )
        )

        # ----------------------------------------------------
        # Centerline
        # ----------------------------------------------------

        blade_centerlines[k].set_data(
            centerline_def[:, 0],
            centerline_def[:, 1]
        )

        blade_centerlines[k].set_3d_properties(
            centerline_def[:, 2]
        )

    return (
        blade_surfaces
        + blade_centerlines
    )


# ============================================================
# CREATE ROTOR ANIMATION
# ============================================================

ani_rotor = FuncAnimation(
    fig,
    update_rotor,
    frames=n_frames,
    interval=1000 / fps,
    blit=False
)


# ============================================================
# SAVE ROTOR GIF
# ============================================================

rotor_output = (
    "flexible_rotor.gif"
)

print(
    f"Saving rotor animation to {rotor_output}..."
)

ani_rotor.save(
    rotor_output,
    writer=PillowWriter(
        fps=fps
    ),
    dpi=100
)

print(
    f"Saved: {rotor_output}"
)

plt.close(fig)


# ============================================================
# FLAP MOTION FIGURE
# ============================================================

fig_flap, ax_flap = plt.subplots(
    figsize=(10, 4.5)
)

fig_flap.patch.set_facecolor("white")
ax_flap.set_facecolor("white")


# ------------------------------------------------------------
# Complete flap signal
# ------------------------------------------------------------

ax_flap.plot(
    t,
    flap_signal,
    linewidth=2,
    label="Flap displacement"
)


# ------------------------------------------------------------
# Moving time indicator
# ------------------------------------------------------------

current_flap_line = ax_flap.axvline(
    t[0],
    linewidth=2
)


# ------------------------------------------------------------
# Current flap value
# ------------------------------------------------------------

current_flap_point, = ax_flap.plot(
    [t[0]],
    [flap_signal[0]],
    "o",
    markersize=8
)


# ------------------------------------------------------------
# Axis formatting
# ------------------------------------------------------------

ax_flap.set_xlim(
    0,
    duration
)

ax_flap.set_ylim(
    0,
    1.05
)

ax_flap.set_xlabel(
    "Time [s]"
)

ax_flap.set_ylabel(
    "Normalized flap displacement"
)

ax_flap.set_title(
    "Flexible Rotor Flap Motion"
)

ax_flap.grid(
    True,
    alpha=0.25
)


# ============================================================
# FLAP ANIMATION UPDATE
# ============================================================

def update_flap(frame):

    current_time = t[frame]

    current_flap = flap_signal[frame]

    # Move vertical time indicator
    current_flap_line.set_xdata(
        [current_time, current_time]
    )

    # Move current-value marker
    current_flap_point.set_data(
        [current_time],
        [current_flap]
    )

    return (
        current_flap_line,
        current_flap_point
    )


# ============================================================
# CREATE FLAP ANIMATION
# ============================================================

ani_flap = FuncAnimation(
    fig_flap,
    update_flap,
    frames=n_frames,
    interval=1000 / fps,
    blit=True
)


# ============================================================
# SAVE FLAP GIF
# ============================================================

flap_output = (
    "flap_motion.gif"
)

print(
    f"Saving flap animation to {flap_output}..."
)

ani_flap.save(
    flap_output,
    writer=PillowWriter(
        fps=fps
    ),
    dpi=120
)

print(
    f"Saved: {flap_output}"
)

plt.close(fig_flap)


# ============================================================
# FREQUENCY SPECTRUM FIGURE
# ============================================================

fig_fft, ax_fft = plt.subplots(
    figsize=(10, 4.5)
)

fig_fft.patch.set_facecolor("white")
ax_fft.set_facecolor("white")


# ------------------------------------------------------------
# Spectrum
# ------------------------------------------------------------

ax_fft.plot(
    frequency,
    spectrum,
    linewidth=2
)

ax_fft.set_xlim(
    0,
    4.5
)

ax_fft.set_ylim(
    0,
    1.1
)

ax_fft.set_xlabel(
    "Frequency [Hz]"
)

ax_fft.set_ylabel(
    "Normalized amplitude"
)

ax_fft.set_title(
    "Flap Motion Frequency Spectrum"
)

ax_fft.grid(
    True,
    alpha=0.25
)


# ============================================================
# SAVE FREQUENCY SPECTRUM
# ============================================================

spectrum_output = (
    "flap_spectrum.png"
)

fig_fft.savefig(
    spectrum_output,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Saved: {spectrum_output}"
)

plt.close(fig_fft)


# ============================================================
# DONE
# ============================================================

print()
print("Finished.")
print("Generated files:")
print("  flexible_rotor.gif")
print("  flap_motion.gif")
print("  flap_spectrum.png")
