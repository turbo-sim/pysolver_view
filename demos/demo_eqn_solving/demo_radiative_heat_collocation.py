"""Compare root solvers on a nonlinear, radiating heat-conduction boundary problem.

Run this file directly, or use --no-show for saved figures only. The comparison
uses the public library="pysolver" interface and an analytic Jacobian. Its
Chebyshev collocation layout follows Nozzlex's heat-equation example; no Nozzlex,
JAX, property library, or development-directory imports are needed.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.integrate import solve_bvp
from scipy.interpolate import BarycentricInterpolator

import pysolver_view as psv

NUM_POINTS = 500  # Collocation nodes; polynomial degree is NUM_POINTS - 1.

METHODS = {
    "newton_raphson": "Newton",
    "levenberg_marquardt": "LM",
    "powell_hybrid": "Powell hybrid",
}
PROFILES = {False: "Full / large steps", True: "Initial limit 0.1"}


def chebyshev_lobatto(degree, left=0.0, right=1.0):
    """Return ascending Chebyshev nodes and a physical first-derivative matrix.

    A polynomial of degree N is represented by N+1 nodal values. Off-diagonal
    entries differentiate the Lagrange interpolant; zero row sums differentiate
    a constant to zero. Reversal makes the first/last nodes the left/right ends.
    """
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or degree < 2
    ):
        raise ValueError("degree must be an integer >= 2.")
    if not np.isfinite([left, right]).all() or right <= left:
        raise ValueError("Require finite endpoints with left < right.")
    indices = np.arange(degree + 1)
    reference_nodes = np.cos(np.pi * indices / degree)
    factors = np.where((indices == 0) | (indices == degree), 2.0, 1.0)
    factors *= (-1.0) ** indices
    differences = reference_nodes[:, None] - reference_nodes[None, :]
    derivative = (factors[:, None] / factors[None, :]) / (
        differences + np.eye(degree + 1)
    )
    derivative -= np.diag(derivative.sum(axis=1))
    nodes = left + (reference_nodes + 1.0) * (right - left) / 2.0
    return nodes[::-1].copy(), (2.0 * derivative / (right - left))[::-1, ::-1].copy()


class RadiativeHeatProblem(psv.NonlinearSystemProblem):
    """Steady conduction with uniform heating and distributed radiation exchange.

    Dimensionless temperature satisfies T'' + Q - R*(T**4 - 1) = 0 on [0, 1],
    with T(0)=T(1)=1. Conductivity is constant; the ambient temperature is one.
    Positive Q heats the rod, and the radiation term cools it above ambient.

    Unknowns are [T_nodes, v_nodes], where v=-T'/Q is normalized conductive flux.
    There are 2*(degree+1) equations: Fourier's law at every node, energy balance
    at interior nodes, and the two boundary temperatures. NUM_POINTS sets the
    default resolution. All residuals and variables are dimensionless.
    """

    def __init__(self, degree=NUM_POINTS - 1, heating=50.0, radiation=10.0):
        """Build the spatial operator once and validate physical coefficients."""
        if not np.isfinite(heating) or heating <= 0:
            raise ValueError("heating must be finite and positive.")
        if not np.isfinite(radiation) or radiation < 0:
            raise ValueError("radiation must be finite and nonnegative.")
        self.degree = degree
        self.heating = float(heating)
        self.radiation = float(radiation)
        self.nodes, self.D = chebyshev_lobatto(degree)
        self.num_nodes = len(self.nodes)

    def unpack(self, state):
        """Split nodal temperature and normalized flux without discarding complex probes."""
        state = np.asarray(state)
        if state.shape != (2 * self.num_nodes,):
            raise ValueError("Expected temperature and flux at every collocation node.")
        return state[: self.num_nodes], state[self.num_nodes :]

    def initial_guess(self, temperature=0.3):
        """Return uniform cold temperature and zero flux, including incorrect boundaries."""
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Initial absolute temperature must be finite and positive."
            )
        return np.r_[np.full(self.num_nodes, temperature), np.zeros(self.num_nodes)]

    def residual(self, state):
        """Evaluate Fourier's law, interior energy balances, and boundary equations."""
        temperature, flux = self.unpack(state)
        if np.any(temperature.real <= 0):
            raise psv.DomainError("Absolute temperature must remain positive.")
        conduction = self.D @ temperature / self.heating + flux
        energy = (
            self.D @ flux + self.radiation / self.heating * (temperature**4 - 1.0) - 1.0
        )
        return np.r_[conduction, energy[1:-1], temperature[0] - 1, temperature[-1] - 1]

    def gradient(self, state):
        """Return the exact Jacobian, including the radiation derivative 4*R*T**3/Q."""
        temperature, _ = self.unpack(state)
        identity = np.eye(self.num_nodes)
        radiation_derivative = np.diag(
            4 * self.radiation / self.heating * temperature**3
        )
        return np.block(
            [
                [self.D / self.heating, identity],
                [radiation_derivative[1:-1], self.D[1:-1]],
                [identity[[0, -1]], np.zeros((2, self.num_nodes))],
            ]
        )

    def interpolate(self, state, positions):
        """Evaluate both nodal polynomial fields at arbitrary positions in the rod."""
        temperature, flux = self.unpack(state)
        weights = (-1.0) ** np.arange(self.num_nodes)
        weights[[0, -1]] *= 0.5
        interpolant = BarycentricInterpolator(
            self.nodes, np.vstack((temperature, flux)), axis=1, wi=weights
        )
        return interpolant(positions)


def solve_reference(problem, tolerance=1e-10):
    """Compute an independent adaptive BVP reference in temperature/slope coordinates.

    This uses SciPy's piecewise fourth-order collocation, not the spectral
    differentiation matrix or any of the three root drivers being compared.
    Its warm initial guess is used only for validation, never for the six runs.
    """

    def rhs(position, fields):
        """Express T'=slope and slope'=R*(T**4-1)-Q on the adaptive mesh."""
        temperature, slope = fields
        return np.vstack(
            (slope, problem.radiation * (temperature**4 - 1) - problem.heating)
        )

    def boundaries(left, right):
        """Fix the two end temperatures at ambient."""
        return np.array([left[0] - 1, right[0] - 1])

    mesh = np.linspace(0, 1, 41)
    guess = np.vstack((np.ones_like(mesh), np.zeros_like(mesh)))
    reference = solve_bvp(rhs, boundaries, mesh, guess, tol=tolerance, max_nodes=20000)
    if not reference.success:
        raise RuntimeError(f"Independent BVP reference failed: {reference.message}")
    if np.min(reference.sol(np.linspace(0, 1, 1001))[0]) <= 0:
        raise RuntimeError(
            "Independent reference is outside the positive-temperature domain."
        )
    return reference


def solver_options(method, controlled):
    """Select full Newton/large initial radii or a 0.1 limit with other control defaults.

    LM and hybrid intrinsically retain their trust regions. Their baseline starts
    at the default maximum radius of 100. This can still restrict the first
    direction, depending on resolution and numerical scaling.
    A limit of 0.1 bounds the Euclidean displacement of the dimensionless state;
    it is neither a fraction of the Newton direction nor a per-node step size.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    if method == "newton_raphson":
        if not controlled:
            return {"backtracking": None}
        return {"backtracking": {"step_limits": {"initial_limit": 0.1}}}
    return {"initial_radius": 0.1 if controlled else 100.0}


def solve_comparison(
    problem,
    initial_temperature=0.3,
    tolerance=1e-8,
    max_trials=500,
    jacobian_scaling=True,
):
    """Run six independent solves from the same uniform guess, preserving all trials."""
    initial = problem.initial_guess(initial_temperature)
    runs = []
    for method in METHODS:
        for controlled in (False, True):
            solver = psv.NonlinearSystemSolver(
                problem,
                library="pysolver",
                method=method,
                tolerance=tolerance,
                max_iterations=max_trials,
                options=dict(
                    solver_options(method, controlled),
                    jacobian_scaling=jacobian_scaling,
                ),
                update_on="iteration",
                print_convergence=True,
                plot_convergence=False,
            )
            solver.solve(initial.copy())
            runs.append(dict(method=method, controlled=controlled, solver=solver))
    return runs


def summarize_runs(problem, runs, reference):
    """Audit residuals and off-grid field errors after solving, outside solver counts."""
    positions = np.linspace(0, 1, 1001)
    reference_temperature, reference_slope = reference.sol(positions)
    reference_flux = -reference_slope / problem.heating
    rows = []
    for run in runs:
        solver = run["solver"]
        result = solver.result
        temperature, flux = problem.interpolate(result.x, positions)
        valid_evaluations = [row for row in result.evaluations if row["valid"]]
        rows.append(
            dict(
                method=METHODS[run["method"]],
                profile=PROFILES[run["controlled"]],
                success=bool(result.success),
                iterations=int(result.nit),
                residual_calls=int(result.nfev),
                jacobian_calls=int(result.njev),
                rejected_trials=int(result.nrejected),
                invalid_trials=int(result.ninvalid),
                residual_norm=float(np.linalg.norm(problem.residual(result.x))),
                first_step_norm=(
                    float(result.trials[0]["scaled_step_norm"])
                    if result.trials
                    else 0.0
                ),
                max_trial_residual=max(
                    row["residual_norm"] for row in valid_evaluations
                ),
                temperature_error=float(
                    np.max(np.abs(temperature - reference_temperature))
                ),
                flux_error=float(np.max(np.abs(flux - reference_flux))),
                peak_temperature=float(np.max(temperature)),
                message=result.message,
                jacobian_scaling=result.jacobian_scaling,
            )
        )
    return rows


def print_summary(problem, rows):
    """Print an ASCII comparison that separates iterations, work, and physical accuracy."""
    print(
        f"Radiating rod: {problem.num_nodes} nodes, {2 * problem.num_nodes} unknowns."
    )
    print(f"Uniform heating Q={problem.heating:g}; radiation R={problem.radiation:g}.")
    print("Newton baseline: full steps. LM/hybrid baseline: initial radius 100.")
    print(
        "Controlled: initial scaled 2-norm limit 0.1; other step controls use defaults."
    )
    print("LM/hybrid always retain trust-region acceptance and recovery.")
    if rows:
        print(
            f"Jacobian-based numerical scaling: {rows[0].get('jacobian_scaling', False)}."
        )
    full_names = {
        "Newton": "Newton-Raphson",
        "LM": "Levenberg-Marquardt",
        "Powell hybrid": "Powell hybrid",
    }
    method_names = [full_names.get(row["method"], row["method"]) for row in rows]
    method_width = max(len("Method"), *(len(name) for name in method_names))
    header = (
        f"{'Method':<{method_width}}  {'Profile':<19} {'OK':>3} {'Iter':>5} {'Calls':>6} "
        f"{'Jac':>4} {'Reject':>6} {'First step':>11} {'Max ||F||':>11} {'Final ||F||':>12} {'T error':>10}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for row, method_name in zip(rows, method_names):
        print(
            f"{method_name:<{method_width}}  {row['profile']:<19} {'yes' if row['success'] else 'no':>3} "
            f"{row['iterations']:>5} {row['residual_calls']:>6} {row['jacobian_calls']:>4} "
            f"{row['rejected_trials']:>6} {row['first_step_norm']:>11.3g} "
            f"{row['max_trial_residual']:>11.3e} {row['residual_norm']:>12.3e} "
            f"{row['temperature_error']:>10.2e}"
        )
        if not row["success"]:
            print(f"  Stop reason: {row['message']}")
    print(
        "T error: maximum temperature difference from the independent BVP on 1001 points."
    )
    print(
        "Max ||F|| includes rejected trials and the initial state; Iter counts accepted moves."
    )
    print(
        "Analytic Jacobians are used. Reference solves and post-solve audits are excluded from counts."
    )


def plot_marked_curve(axis, x, y, style):
    """Draw a curve and separate white markers above every line in its subplot."""
    axis.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        zorder=2,
    )
    axis.plot(
        x,
        y,
        linestyle="none",
        marker=style["marker"],
        markersize=style["markersize"],
        markerfacecolor="white",
        markeredgecolor=style["color"],
        zorder=3,
    )


def plot_comparison(problem, runs, reference, initial_temperature):
    """Plot both physical fields and accepted/all-trial histories without extra model calls."""
    psv.set_plot_options(grid=False)
    colors = dict(zip(METHODS, plt.get_cmap("plasma")(np.linspace(0.2, 0.8, 3))))
    solution_fig, solution_axes = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained"
    )
    history_fig, history_axes = plt.subplots(
        2, 2, figsize=(12, 8), layout="constrained"
    )
    positions = np.linspace(0, 1, 1001)
    reference_temperature, reference_slope = reference.sol(positions)
    for axis, values in zip(
        solution_axes, (reference_temperature, -reference_slope / problem.heating)
    ):
        axis.plot(
            positions, values, color="black", linewidth=2.5, label="Independent BVP"
        )
    initial = problem.unpack(problem.initial_guess(initial_temperature))
    for axis, values in zip(solution_axes, initial):
        axis.plot(
            problem.nodes,
            values,
            ":",
            color="0.5",
            marker=".",
            label="Uniform initial guess",
        )

    reference_nodal_temperature = reference.sol(problem.nodes)[0]
    run_handles = []
    for run in runs:
        solver = run["solver"]
        result = solver.result
        label = f"{METHODS[run['method']]}: {PROFILES[run['controlled']]}"
        if not result.success:
            label += " (failed)"
        style = dict(
            color=colors[run["method"]],
            linestyle="-" if run["controlled"] else "--",
            marker="o" if run["controlled"] else "s",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=colors[run["method"]],
            linewidth=1.2,
            label=label,
        )
        run_handles.append(Line2D([], [], **style))
        for axis, values in zip(solution_axes, problem.unpack(result.x)):
            plot_marked_curve(axis, problem.nodes, values, style)
        history = solver.iteration_history
        evaluations = result.evaluations
        plot_marked_curve(
            history_axes[0, 0],
            [row["iteration"] for row in history],
            [row["residual_norm"] for row in history],
            style,
        )
        plot_marked_curve(
            history_axes[0, 1],
            [row["nfev"] for row in evaluations],
            [row["residual_norm"] if row["valid"] else np.nan for row in evaluations],
            style,
        )
        plot_marked_curve(
            history_axes[1, 0],
            [row["trial"] for row in result.trials],
            [row["scaled_step_norm"] for row in result.trials],
            style,
        )
        plot_marked_curve(
            history_axes[1, 1],
            [row["iteration"] for row in history],
            [
                np.max(
                    np.abs(row["x"][: problem.num_nodes] - reference_nodal_temperature)
                )
                for row in history
            ],
            style,
        )

    for axis, ylabel in zip(
        solution_axes,
        ("Temperature / ambient temperature", "Normalized conductive heat flux"),
    ):
        axis.set(xlabel="Position / rod length", ylabel=ylabel)
        axis.grid(alpha=0.25)
    titles = (
        "Accepted states",
        "All residual evaluations (including rejected trials)",
        "Attempted displacements",
        "Temperature error at collocation nodes",
    )
    xlabels = (
        "Accepted iterations",
        "Residual calls",
        "Trial attempts",
        "Accepted iterations",
    )
    ylabels = (
        r"$\|F\|_2$",
        r"$\|F\|_2$",
        r"$\|\Delta z\|_2$ (dimensionless state)",
        "Maximum temperature error",
    )
    for axis, title, xlabel, ylabel in zip(history_axes.flat, titles, xlabels, ylabels):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, yscale="symlog")
        axis.set_yscale("symlog", linthresh=1e-14)
        axis.yaxis.get_major_locator().set_params(numticks=7)
        axis.grid(alpha=0.25)
    for axis in history_axes[0]:
        axis.axhline(
            runs[0]["solver"].result.residual_tolerance, color="0.5", linestyle=":"
        )
    history_axes[1, 0].axhline(0.1, color="0.5", linestyle=":")
    solution_fig.suptitle("Steady conduction with distributed radiation")
    history_fig.suptitle("Root solver comparison from the same uniform cold guess")
    for figure, axis, columns in (
        (solution_fig, solution_axes[0], 4),
        (history_fig, history_axes[0, 0], 3),
    ):
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles + run_handles,
            labels + [handle.get_label() for handle in run_handles],
            loc="outside lower center",
            ncol=columns,
            fontsize=8,
        )
    return solution_fig, history_fig


def save_results(output, problem, runs, rows, reference, figures, initial_temperature):
    """Save reusable numerical results, actual histories, and the two figures."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def json_value(value):
        """Convert NumPy arrays/scalars in detached result histories to JSON values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    payload = dict(
        versions=dict(
            numpy=np.__version__, scipy=scipy.__version__, pysolver_view=psv.__version__
        ),
        problem=dict(
            degree=problem.degree,
            heating=problem.heating,
            radiation=problem.radiation,
            initial_temperature=initial_temperature,
        ),
        nodes=problem.nodes,
        reference=dict(
            nodes=reference.x.size,
            max_relative_rms_residual=float(np.max(reference.rms_residuals)),
        ),
        runs=[
            dict(
                summary=row,
                options=run["solver"].result.effective_options,
                jacobian_scale=run["solver"].result.jacobian_scale,
                jacobian_scale_updates=run["solver"].result.jacobian_scale_updates,
                solution=run["solver"].x_final,
                history=run["solver"].iteration_history,
                evaluations=run["solver"].result.evaluations,
                trials=run["solver"].result.trials,
            )
            for run, row in zip(runs, rows)
        ],
    )
    (output / "results.json").write_text(
        json.dumps(payload, default=json_value, indent=2), encoding="utf-8"
    )
    for name, figure in zip(("solution", "convergence"), figures):
        figure.savefig(output / f"{name}.png", dpi=180)
        figure.savefig(output / f"{name}.svg")


def main(argv=None):
    """Run the physical example, print the comparison, and save/show its figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degree",
        type=int,
        default=NUM_POINTS - 1,
        help="Polynomial degree; total unknowns = 2*(degree+1)",
    )
    parser.add_argument(
        "--heating",
        type=float,
        default=50.0,
        help="Dimensionless uniform heat generation Q",
    )
    parser.add_argument(
        "--radiation",
        type=float,
        default=10.0,
        help="Dimensionless radiative exchange coefficient R",
    )
    parser.add_argument("--initial-temperature", type=float, default=0.3)
    parser.add_argument(
        "--no-jacobian-scaling",
        action="store_true",
        help="Disable numerical column scaling (enabled by default)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save figures without opening windows"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results_radiative_heat"
    )
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")
    problem = RadiativeHeatProblem(args.degree, args.heating, args.radiation)
    reference = solve_reference(problem)
    runs = solve_comparison(
        problem,
        initial_temperature=args.initial_temperature,
        jacobian_scaling=not args.no_jacobian_scaling,
    )
    rows = summarize_runs(problem, runs, reference)
    print_summary(problem, rows)
    figures = plot_comparison(problem, runs, reference, args.initial_temperature)
    save_results(
        args.output, problem, runs, rows, reference, figures, args.initial_temperature
    )
    print(f"Saved results to {args.output.resolve()}")
    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
