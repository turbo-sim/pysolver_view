"""Draw the local two-variable LM/dogleg geometry used by the math note.

Run run_step_geometry.py from the experiment root. It writes PNG and SVG to
results_step_geometry/ and prints independently derived example values. This is a
linearized subproblem illustration, not a nonlinear convergence benchmark.
"""

import json
from pathlib import Path
import argparse
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.optimize import brentq

from pysolver_view.roots.solvers.linear_model import LinearModel


def example():
    """Derive both constrained steps independently of the solver code."""
    diagonal = np.array([1.0, 3.0])
    a = np.diag(diagonal)
    residual = np.array([3.0, 3.0])
    radius = 1.5
    gradient = a.T @ residual
    gauss_newton = np.linalg.lstsq(a, -residual, rcond=None)[0]
    cauchy = -gradient * (gradient @ gradient) / np.linalg.norm(a @ gradient) ** 2

    def damped_step(damping):
        """Return the independent diagonal-model LM step for a nonnegative damping
        parameter.
        """
        return -diagonal * residual / (diagonal**2 + damping)

    damping = brentq(
        lambda value: np.linalg.norm(damped_step(value)) - radius,
        0.0,
        np.linalg.norm(gradient) / radius,
        xtol=1e-14,
    )
    lm = damped_step(damping)
    ray = gauss_newton - cauchy
    beta = cauchy @ ray
    ray_squared = ray @ ray
    fraction = (
        -beta + np.sqrt(beta**2 + ray_squared * (radius**2 - cauchy @ cauchy))
    ) / ray_squared
    dogleg = cauchy + fraction * ray

    # Independent formulas agree with the actual experimental implementation.
    model = LinearModel(a, residual)
    np.testing.assert_allclose(model.step("lm", radius), lm, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(model.step("hybr", radius), dogleg, rtol=2e-14)
    points = {
        "origin": np.zeros(2),
        "gauss_newton": gauss_newton,
        "cauchy": cauchy,
        "dogleg": dogleg,
        "lm": lm,
    }
    values = {
        name: {
            "z": point.tolist(),
            "step_norm": float(np.linalg.norm(point)),
            "model_residual_norm": float(np.linalg.norm(residual + a @ point)),
            "q": float(0.5 * np.linalg.norm(residual + a @ point) ** 2),
        }
        for name, point in points.items()
    }
    values["lm_damping"] = damping
    values["dogleg_fraction"] = fraction
    return radius, points, damped_step, values


def main():
    """Parse the output directory, draw the independently checked local-model geometry,
    and print its values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_step_geometry",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    radius, points, damped_step, values = example()
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(10.8, 7), layout="constrained")
    first = np.linspace(-3.5, 1.8, 700)
    second = np.linspace(-1.85, 1.85, 500)
    z1, z2 = np.meshgrid(first, second)
    merit = 0.5 * ((3 + z1) ** 2 + (3 + 3 * z2) ** 2)
    contours = axis.contour(
        z1,
        z2,
        merit,
        levels=[0.25, 1.0, 2.0, 4.0, 6.0, 9.0, 14.0, 20.0],
        colors="#adb5bd",
        linewidths=0.85,
        zorder=0,
    )
    axis.clabel(contours, fontsize=8, inline=True, fmt="%g")
    axis.add_patch(
        Circle((0, 0), radius, facecolor="#e8eff6", alpha=0.4, edgecolor="none")
    )
    axis.add_patch(
        Circle((0, 0), radius, fill=False, edgecolor="#263445", linewidth=1.7)
    )
    dogleg_path = np.vstack(
        [points["origin"], points["cauchy"], points["gauss_newton"]]
    )
    axis.plot(
        dogleg_path[:, 0],
        dogleg_path[:, 1],
        color="#cb7b15",
        linewidth=2.2,
        linestyle="--",
        label="Dogleg path",
        zorder=2,
    )
    damping_path = np.vstack(
        [
            points["origin"],
            [damped_step(value) for value in np.geomspace(1e5, 1e-6, 600)],
        ]
    )
    axis.plot(
        damping_path[:, 0],
        damping_path[:, 1],
        color="#16659a",
        linewidth=2.2,
        label="LM damping path",
        zorder=2,
    )
    styles = {
        "origin": ("o", "#263445", 45),
        "gauss_newton": ("s", "#263445", 65),
        "cauchy": ("o", "#cb7b15", 65),
        "dogleg": ("D", "#cb7b15", 85),
        "lm": ("*", "#16659a", 180),
    }
    labels = {
        "origin": ("Current point\n$z=0$", (0.17, 0.16)),
        "gauss_newton": ("Gauss–Newton\n$z_{GN}=(-3,-1)$", (-3.30, -0.54)),
        "cauchy": ("Cauchy point", (0.02, -1.33)),
        "dogleg": ("Dogleg boundary step", (-2.46, -1.57)),
        "lm": ("LM boundary step", (-2.50, -0.27)),
    }
    for name, point in points.items():
        marker, color, size = styles[name]
        axis.scatter(*point, s=size, marker=marker, color=color, zorder=4)
        label, location = labels[name]
        axis.annotate(
            label,
            xy=point,
            xytext=location,
            color=color,
            fontsize=11,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
            zorder=5,
        )
    axis.text(0.30, 1.62, r"$\|z\|_2=\Delta=1.5$", color="#263445")
    axis.set(
        xlim=(first[0], first[-1]),
        ylim=(second[0], second[-1]),
        xlabel=r"Scaled displacement $z_1$",
        ylabel=r"Scaled displacement $z_2$",
        aspect="equal",
        title="Two ways to restrict the same Gauss–Newton model",
    )
    axis.legend(loc="upper left", framealpha=0.95)
    axis.grid(alpha=0.1)
    fig.supxlabel(
        r"Contours: $q(z)=\frac{1}{2}[(3+z_1)^2+(3+3z_2)^2]$. "
        "Both constrained steps lie on the circle; LM minimizes this model over the disk.",
        fontsize=11,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"lm_hybrid_step_geometry.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
