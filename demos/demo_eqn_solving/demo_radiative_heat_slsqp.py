"""Compare Newton and SLSQP feasibility solves with matching initial fractions.

All physical equations, derivatives, collocation and reference calculations are
imported from demo_radiative_heat_collocation. Only the solver formulation changes:
SLSQP minimizes zero subject to each original residual being an equality. Its
first-step fractions, like Newton's, multiply proposed directions, not distances.
Run directly with --no-show for saved figures, or --degree 48 for a smaller case.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import scipy

import pysolver_view as psv

if __package__:
    from .demo_radiative_heat_collocation import (
        NUM_POINTS,
        RadiativeHeatProblem,
        plot_marked_curve,
        solve_reference,
    )
else:
    from demo_radiative_heat_collocation import (
        NUM_POINTS,
        RadiativeHeatProblem,
        plot_marked_curve,
        solve_reference,
    )


FIRST_STEP_FRACTIONS = (1.0, 0.3, 0.2, 0.1)
STEP_GROWTH_FACTOR = 1.5
RESIDUAL_TOLERANCE = 1e-8
MAX_ITERATIONS = 200
# Display-only vertical shift; zero disables it without changing solver data.
SLSQP_VERTICAL_OFFSET_POINTS = 2.0


def solve_comparison(
    problem,
    initial_temperature=0.3,
    tolerance=RESIDUAL_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    print_convergence=True,
):
    """Run Newton and equality-constrained SLSQP for each initial fraction.

    Every run starts from a copy of the same cold state. Newton's backtracking
    is disabled for all four runs. Both methods start with the chosen direction
    fraction and grow their caps by 1.5 toward 1. SLSQP retains its native merit
    line search and domain recovery, so even its full-step run can shorten steps.
    The adapter adds no bounds or scaling to the original dimensionless model.
    Numerical budgets retain backend meanings: Newton counts trial attempts,
    while SLSQP limits native major iterations.
    """
    initial = problem.initial_guess(initial_temperature)
    runs = []
    for fraction in FIRST_STEP_FRACTIONS:
        newton = psv.NonlinearSystemSolver(
            problem,
            library="pysolver",
            method="newton_raphson",
            tolerance=tolerance,
            max_iterations=max_iterations,
            options={
                "backtracking": None,
                "first_step_fraction": fraction,
                "step_growth_factor": STEP_GROWTH_FACTOR,
            },
            update_on="iteration",
            print_convergence=print_convergence,
            plot_convergence=False,
        )
        newton.solve(initial.copy())
        setting = "full steps" if fraction == 1 else f"initial fraction {fraction:g}"
        runs.append(
            dict(
                label=f"Newton-Raphson, {setting}",
                method="newton",
                fraction=fraction,
                solver=newton,
            )
        )
    for fraction in FIRST_STEP_FRACTIONS:
        adapted = psv.as_optimization_problem(
            problem, n_variables=initial.size, n_equations=initial.size
        )
        optimizer = psv.OptimizationSolver(
            adapted,
            library="pysolver",
            method="slsqp",
            tolerance=tolerance,
            max_iterations=max_iterations,
            problem_scale=None,
            update_on="iteration",
            print_convergence=print_convergence,
            plot_convergence=False,
            extra_options={
                "first_step_fraction": fraction,
                "max_step_fraction": 1.0,
                "step_growth_factor": STEP_GROWTH_FACTOR,
                "domain_backtracking": True,
                "feasibility_tol": tolerance,
                "optimality_tol": tolerance,
            },
        )
        optimizer.solve(initial.copy())
        setting = "full steps" if fraction == 1 else f"initial fraction {fraction:g}"
        runs.append(
            dict(
                label=f"SLSQP, {setting}",
                method="slsqp",
                fraction=fraction,
                solver=optimizer,
            )
        )
    return runs


def accepted_history(run):
    """Normalize stored accepted states without evaluating the physical model.

    Accepted-move indices are consecutive; the original backend iteration number
    is retained separately because SLSQP major iterations can differ. SLSQP
    fitness rows contain a zero objective followed by the original equalities.
    """
    history = []
    for index, record in enumerate(run["solver"].backend_result.history):
        residual_norm = (
            record["residual_norm"]
            if run["method"] == "newton"
            else np.linalg.norm(record["fitness"][1:])
        )
        history.append(
            dict(
                accepted_move=index,
                backend_iteration=int(record["iteration"]),
                residual_norm=float(residual_norm),
                step_norm=float(record["step_norm"]),
                nfev=int(record["nfev"]),
                njev=int(record["njev"]),
                x=record["x"].copy(),
            )
        )
    return history


def summarize_runs(problem, runs, reference, tolerance=RESIDUAL_TOLERANCE):
    """Audit original equations and off-grid fields outside solve call counters.

    Root acceptance uses the residual two-norm, separately from native stopping
    and the optimizer's maximum-component feasibility check. A zero objective
    or stationary Lagrangian by itself supplies no evidence of a root.
    """
    positions = np.linspace(0.0, 1.0, 1001)
    expected_temperature, expected_slope = reference.sol(positions)
    rows = []
    for run in runs:
        result = run["solver"].result
        history = accepted_history(run)
        try:
            residual = problem.residual(result.x)
            residual_norm = float(np.linalg.norm(residual))
            residual_max = float(np.max(np.abs(residual)))
        except psv.DomainError:
            residual_norm = residual_max = float("inf")
        temperature, flux = problem.interpolate(result.x, positions)
        rows.append(
            dict(
                label=run["label"],
                method=run["method"],
                first_step_fraction=run["fraction"],
                solver_success=bool(result.success),
                backend_success=bool(result.solver_success),
                root_found=bool(
                    np.isfinite(residual_norm) and residual_norm <= tolerance
                ),
                residual_norm=residual_norm,
                max_residual=residual_max,
                residual_tolerance=tolerance,
                accepted_moves=max(0, len(history) - 1),
                backend_iterations=int(result.nit),
                residual_calls=int(result.nfev),
                jacobian_calls=int(result.njev),
                domain_rejections=int(
                    run["solver"].backend_result.get(
                        "ninvalid",
                        run["solver"].backend_result.get("domain_rejections", 0),
                    )
                ),
                first_accepted_step=(
                    history[1]["step_norm"] if len(history) > 1 else None
                ),
                temperature_error=float(
                    np.max(np.abs(temperature - expected_temperature))
                ),
                flux_error=float(
                    np.max(np.abs(flux + expected_slope / problem.heating))
                ),
                minimum_temperature=float(np.min(temperature)),
                elapsed_time=float(result.elapsed_time),
                message=str(result.message),
            )
        )
    return rows


def print_summary(problem, rows):
    """Display independent root acceptance alongside optimizer status and cost."""
    print(
        f"\n{problem.num_nodes} nodes; {2 * problem.num_nodes} variables and equations"
    )
    print(
        f"{'Solver':<33} {'Root?':>6} {'Solver?':>8} {'||F||_2':>11} {'Moves':>7} {'F calls':>8} {'J calls':>8} {'Time [s]':>10}"
    )
    for row in rows:
        print(
            f"{row['label']:<33} {str(row['root_found']):>6} "
            f"{str(row['solver_success']):>8} {row['residual_norm']:>11.3e} "
            f"{row['accepted_moves']:>7} {row['residual_calls']:>8} "
            f"{row['jacobian_calls']:>8} {row['elapsed_time']:>10.3f}"
        )
        if not row["solver_success"] or not row["root_found"]:
            print(f"  {row['message']}")
    print(
        "Moves count accepted displacements; native iteration numbers are saved separately."
    )
    print(
        "Reference calculations and fresh final audits are outside solve counters and timings."
    )
    print(
        "SLSQP's objective is identically zero; Root? checks the original residual two-norm."
    )


def plot_comparison(problem, runs, rows, reference, initial_temperature, tolerance):
    """Plot stored results with a small display-only upward shift for SLSQP.

    The offset is in typographic points, so it works equally on linear and log
    axes without changing the plotted data arrays or exported numerical results.
    """
    psv.set_plot_options(fontsize=10, grid=False)
    convergence, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    solution, fields = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    colors = dict(
        zip(
            FIRST_STEP_FRACTIONS,
            plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(FIRST_STEP_FRACTIONS))),
        )
    )
    positions = np.linspace(0, 1, 1001)
    expected_temperature, expected_slope = reference.sol(positions)
    for axis, values in zip(
        fields, (expected_temperature, -expected_slope / problem.heating)
    ):
        axis.plot(
            positions,
            values,
            color="black",
            linewidth=2,
            label="Independent BVP reference",
        )
    initial = problem.initial_guess(initial_temperature)
    for axis, values in zip(fields, problem.unpack(initial)):
        axis.plot(
            problem.nodes, values, color="0.6", linestyle=":", label="Initial guess"
        )
    for run, row in zip(runs, rows):
        records = accepted_history(run)
        offset = SLSQP_VERTICAL_OFFSET_POINTS if run["method"] == "slsqp" else 0.0
        label = run["label"] + (f" (+{offset:g} pt display offset)" if offset else "")
        style = dict(
            color=colors[run["fraction"]],
            linestyle="-" if run["method"] == "newton" else "--",
            linewidth=1.3,
            marker="o" if run["method"] == "newton" else "s",
            markersize=4,
        )
        moves = [record["accepted_move"] for record in records]
        norms = [
            max(record["residual_norm"], np.finfo(float).tiny) for record in records
        ]
        calls = [record["nfev"] for record in records]
        steps = [record["step_norm"] for record in records]
        for axis, x, y in (
            (axes[0], moves, norms),
            (axes[1], calls, norms),
            (axes[2], moves[1:], steps[1:]),
        ):
            plot_marked_curve(axis, x, y, style)
            for artist in axis.lines[-2:]:
                artist.set_transform(
                    axis.transData
                    + ScaledTranslation(0, offset / 72, axis.figure.dpi_scale_trans)
                )
            axis.lines[-2].set_label(label)
        for axis, values in zip(fields, problem.unpack(run["solver"].result.x)):
            plot_marked_curve(axis, problem.nodes, values, style)
            for artist in axis.lines[-2:]:
                artist.set_transform(
                    axis.transData
                    + ScaledTranslation(0, offset / 72, axis.figure.dpi_scale_trans)
                )
            suffix = "" if row["root_found"] else " (root target unmet)"
            axis.lines[-2].set_label(label + suffix)
    for axis in axes[:2]:
        axis.set_yscale("log")
        axis.axhline(tolerance, color="0.5", linestyle=":", linewidth=1)
        axis.set_ylabel(r"Residual norm $\|F\|_2$")
    axes[0].set_xlabel("Accepted move")
    axes[1].set_xlabel("Residual calls (including rejected trials)")
    axes[2].set_xlabel("Accepted move")
    axes[2].set_ylabel("Step size")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8)
    for axis, label in zip(
        fields, ("Dimensionless temperature", "Normalized conductive flux")
    ):
        axis.set_xlabel("Position")
        axis.set_ylabel(label)
    fields[0].legend(fontsize=8)
    return convergence, solution


def json_value(value):
    """Convert detached arrays and unavailable/nonfinite metrics to strict JSON."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(output, problem, runs, rows, reference, figures, initial_temperature):
    """Export summaries, actual accepted states, trial diagnostics and figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(
        versions=dict(
            pysolver_view=psv.__version__, numpy=np.__version__, scipy=scipy.__version__
        ),
        problem=dict(
            degree=problem.degree,
            heating=problem.heating,
            radiation=problem.radiation,
            initial_temperature=initial_temperature,
        ),
        reference=dict(
            nodes=int(reference.x.size),
            max_relative_rms_residual=float(np.max(reference.rms_residuals)),
        ),
        runs=[
            dict(
                summary=row,
                options=run["solver"].result.effective_options,
                solution=run["solver"].result.x,
                accepted_history=accepted_history(run),
                trials=run["solver"].backend_result.get("trials", []),
                events=run["solver"].backend_result.get("events", []),
            )
            for run, row in zip(runs, rows)
        ],
    )
    (output / "results.json").write_text(
        json.dumps(json_value(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    for name, figure in zip(("convergence", "solution"), figures):
        for extension in ("png", "svg"):
            figure.savefig(output / f"{name}.{extension}", dpi=180)


def main(argv=None):
    """Run the configurable eight-solve comparison and export its audited results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=NUM_POINTS - 1)
    parser.add_argument("--heating", type=float, default=50.0)
    parser.add_argument("--radiation", type=float, default=10.0)
    parser.add_argument("--initial-temperature", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=RESIDUAL_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_radiative_heat_slsqp",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.no_show:
        plt.switch_backend("Agg")
    problem = RadiativeHeatProblem(args.degree, args.heating, args.radiation)
    reference = solve_reference(problem)
    runs = solve_comparison(
        problem, args.initial_temperature, args.tolerance, args.max_iterations
    )
    rows = summarize_runs(problem, runs, reference, args.tolerance)
    print_summary(problem, rows)
    figures = plot_comparison(
        problem, runs, rows, reference, args.initial_temperature, args.tolerance
    )
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
