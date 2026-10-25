"""Compare four controlled root solvers using the public PySolverView interface.

Run with --no-show to save the figures without opening a window. Problem
definitions come from the installed pysolver_view.roots.problems package.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pysolver_view as psv

from pysolver_view.roots.problems import (
    BroydenTridiagonalProblem,
    ExtendedRosenbrockProblem,
)


METHOD_STYLES = {
    "levenberg_marquardt": ("Levenberg-Marquardt", "-.", 1.8),
    "powell_hybrid": ("Powell hybrid", ":", 2.2),
    "newton_raphson": ("Newton-Raphson", "-", 3.0),
    "gauss_newton": ("Gauss-Newton", "--", 1.7),
}


def solve_examples():
    """Solve both systems with shared scales and conservative displacement limits."""
    cases = []
    for problem in (ExtendedRosenbrockProblem(n=20), BroydenTridiagonalProblem(n=20)):
        solvers = {}
        for method in METHOD_STYLES:
            options = dict(x_scale=1.0, residual_tol=1e-8, max_iterations=1000)
            if method in ("levenberg_marquardt", "powell_hybrid"):
                options.update(initial_radius=0.1, max_radius=1.0, growth_factor=1.5)
            else:
                options["backtracking"] = dict(
                    step_limits=dict(
                        initial_limit=0.1, max_limit=1.0, growth_factor=1.5
                    )
                )
            solver = psv.NonlinearSystemSolver(
                problem,
                library="pysolver",
                method=method,
                options=options,
                update_on="iteration",
                print_convergence=False,
            )
            solution = solver.solve(problem.x0)
            norm = float(np.linalg.norm(problem.residual(solution)))
            if not solver.success or norm > 1e-8:
                raise RuntimeError(f"{problem.name}, {method}: {solver.message}")
            solvers[method] = solver
            print(
                f"{problem.name}: {method}: {solver.result.nit} iterations, "
                f"{solver.result.nfev} residual calls, ||F|| = {norm:.2e}"
            )
        cases.append((problem, solvers))
    return cases


def plot_examples(cases, output, show):
    """Plot accepted histories; a dashed overlay exposes coincident Newton curves."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(METHOD_STYLES, matplotlib.colormaps["plasma"](
        np.linspace(0.2, 0.8, len(METHOD_STYLES))
    )))

    with plt.rc_context():
        psv.set_plot_options(fontsize=10, grid=True, color_order=list(colors.values()))
        figures = []
        for problem, solvers in cases:
            figure, axes = plt.subplots(
                1, 2, figsize=(10, 4), dpi=100, layout="constrained"
            )
            figures.append(figure)
            for method, solver in solvers.items():
                name, linestyle, width = METHOD_STYLES[method]
                history = solver.iteration_history
                # Exact zeros need a visible plotting floor on a logarithmic axis.
                # Leave the stored histories and verification values unchanged.
                residuals = [max(item["residual_norm"], 1e-16) for item in history]
                for axis, field in zip(axes, ("iteration", "nfev")):
                    axis.semilogy(
                        [item[field] for item in history],
                        residuals,
                        label=name,
                        color=colors[method],
                        linestyle=linestyle,
                        linewidth=width,
                    )
            for axis, label in zip(
                axes, ("Accepted iterations", "Residual evaluations")
            ):
                axis.set(title=problem.name, xlabel=label, ylabel=r"$\|F(x)\|_2$")
                axis.grid(alpha=0.25)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="outside lower center", ncol=4)
            figure.suptitle(
                f"{problem.name}: four controlled root solvers\n"
                "Residuals below 1e-16 shown at 1e-16",
                fontsize=11,
            )
            for extension in ("png", "svg"):
                figure.savefig(
                    output / f"convergence_{problem.name}.{extension}", dpi=150
                )
        if show:
            plt.show()
        for figure in figures:
            plt.close(figure)


def main():
    """Run the example, save results, and optionally display both problem figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results_custom_roots"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = solve_examples()
    summary = [
        dict(
            problem=problem.name,
            method=method,
            solution=solver.x_final.tolist(),
            residual_norm=solver.result.residual_norm,
            nfev=solver.result.nfev,
            njev=solver.result.njev,
            nit=solver.result.nit,
        )
        for problem, solvers in cases
        for method, solver in solvers.items()
    ]
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_examples(cases, args.output, show=not args.no_show)
    print(
        "On nonsingular square systems, Newton and Gauss-Newton directions agree. "
        "The dashed Gauss-Newton curve may overlap the solid Newton-Raphson curve."
    )
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
