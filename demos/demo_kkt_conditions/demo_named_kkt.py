"""Solve a named constrained problem and explain its first-order KKT reports.

Run from the repository root:
    python demos/demo_kkt_conditions/demo_named_kkt.py

The problem has an analytic solution and known multipliers. Both SciPy SLSQP
and the controlled pysolver SLSQP should recover them. The final section checks
two deliberately bad designs to distinguish feasibility from optimality.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pysolver_view as psv


class NamedAllocationProblem(psv.OptimizationProblem):
    """Allocate two streams and a reserve with one equality and two inequalities.

    Minimize 0.5*((stream_a-2)**2 + (stream_b-1)**2 + (reserve+1)**2),
    subject to stream_a + stream_b = 2, stream_a <= 1.2, stream_b <= 2.5,
    and 0 <= each variable <= 3. All values are dimensionless in this example.
    """

    def __init__(self):
        """Declare the one authoritative ordering used by arrays and reports."""
        self.variable_names = ["stream_a", "stream_b", "reserve"]

    def pack(self, named_values):
        """Convert a complete dictionary to a vector using explicit name order."""
        if set(named_values) != set(self.variable_names):
            raise ValueError("Supply exactly stream_a, stream_b, and reserve.")
        return np.asarray([named_values[name] for name in self.variable_names])

    def unpack(self, values):
        """Associate a solver vector with its variable names for application use."""
        if np.shape(values) != (len(self.variable_names),):
            raise ValueError("Expected a vector with one entry per named variable.")
        return dict(zip(self.variable_names, values))

    def fitness(self, x):
        """Return objective/equality/inequalities and same-point named metadata."""
        values = self.unpack(x)
        a = values["stream_a"]
        b = values["stream_b"]
        reserve = values["reserve"]
        objective = 0.5 * ((a - 2) ** 2 + (b - 1) ** 2 + (reserve + 1) ** 2)
        equality = a + b - 2
        inequalities = [a - 1.2, b - 2.5]

        # Report physical values and targets while fitness returns residuals.
        # Entries follow the same order as their equality/inequality residuals.
        self.constraint_data = []
        for name, kind, value, target in [
            ("total_delivery", "=", a + b, 2.0),
            ("stream_a_capacity", "<", a, 1.2),
            ("stream_b_capacity", "<", b, 2.5),
        ]:
            mismatch = float(np.real(value - target))
            self.constraint_data.append(
                dict(
                    name=name,
                    type=kind,
                    value=float(np.real(value)),
                    target=target,
                    normalized_mismatch=mismatch,
                    satisfied=(
                        abs(mismatch) <= 1e-6 if kind == "=" else mismatch <= 1e-6
                    ),
                )
            )
        return np.asarray([objective, equality, *inequalities])

    def gradient(self, x):
        """Return analytic Jacobian rows in the same order as fitness entries."""
        a, b, reserve = x
        return np.asarray(
            [[a - 2, b - 1, reserve + 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]]
        )

    def get_bounds(self):
        """Give each named variable a lower bound of zero and upper bound of three."""
        return np.zeros(3), np.full(3, 3.0)

    def get_nec(self):
        """Declare the total-delivery equality."""
        return 1

    def get_nic(self):
        """Declare the two capacity inequalities in their fitness order."""
        return 2


def check_known_solution(solver):
    """Verify the solution and multipliers against independently derived values."""
    np.testing.assert_allclose(solver.x_final, [1.2, 0.8, 0.0], atol=2e-6, rtol=0)
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.f_final, 0.84, atol=2e-6, rtol=0)
    assessment = solver.evaluate_kkt_conditions(solver.x_final_norm, tol=1e-6)
    assert assessment["available"] and assessment["passed"]
    multipliers = assessment["multipliers"]
    np.testing.assert_allclose(multipliers["equality"], [0.2], atol=2e-6, rtol=0)
    np.testing.assert_allclose(multipliers["inequality"], [0.6, 0.0], atol=2e-6, rtol=0)
    # Bound residuals use solver coordinates, so their multipliers transform.
    bound_multiplier = (
        1.0
        if solver.problem.problem_scale is None
        else 3.0 / solver.problem.problem_scale
    )
    np.testing.assert_allclose(
        multipliers["lower"], [0.0, 0.0, bound_multiplier], atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(multipliers["upper"], np.zeros(3), atol=2e-6, rtol=0)


def plot_comparison(solvers):
    """Compare recorded progress using model calls, including derivative work.

    SciPy records native major-iteration points; the custom driver records
    accepted points. They are different event streams, so the common horizontal
    axis measures model work rather than implying identical iteration semantics.
    """
    psv.set_plot_options(grid=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for label, solver in solvers.items():
        history = solver.convergence_history
        calls = history["func_count_total"]
        axes[0].plot(calls, history["objective_value"], "o-", label=label)
        axes[1].plot(calls, history["constraint_violation"], "o-", label=label)
    axes[0].axhline(0.84, color="gray", linestyle="--", label="Known minimum")
    axes[0].set_ylabel("Objective value")
    axes[1].set_ylabel("Maximum constraint violation (including bounds)")
    # Preserve exact zeros while resolving small feasibility errors.
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].axhline(1e-6, color="gray", linestyle="--", label="Feasibility target")
    for ax in axes:
        ax.set_xlabel("Total model calls (including derivative/diagnostic calls)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Named allocation problem: SciPy versus controlled SLSQP")
    return fig


def main(argv=None):
    """Print both solve histories and named reports, audit bad designs, and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Disable automatic KKT checks; explicitly audit the final solution",
    )
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="Use solver variables in [0, 1] and explain transformed bound multipliers",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        metavar="DIRECTORY",
        help="Save each final report to this explicit directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the comparison plot without opening a window",
    )
    args = parser.parse_args(argv)
    print("Expected optimum: stream_a=1.2, stream_b=0.8, reserve=0; objective=0.84.")
    print(
        "Active restrictions: total_delivery, stream_a_capacity, reserve lower bound."
    )
    expected_lower = 3.0 if args.scaled else 1.0
    print(
        f"Expected multipliers: equality=0.2, active capacity=0.6, reserve lower={expected_lower:g}."
    )
    if args.scaled:
        print("Scaling: physical variables [0, 3] map to solver coordinates [0, 1].")
        print(
            "The bound multiplier changes from 1 to 3; the physical optimum is unchanged."
        )
    if args.final_only:
        print(
            "Automatic KKT checks are disabled; final solutions are explicitly audited below."
        )

    print("Two separate solves follow, each with its own history and final report.")
    solvers = {}
    for run_number, library in enumerate(["scipy", "pysolver"], start=1):
        label = (
            "SciPy SLSQP"
            if library == "scipy"
            else "Custom SLSQP (first fraction 0.25)"
        )
        print(f"\n{'=' * 80}\nRun {run_number}/2: {label}\n{'=' * 80}")
        print(
            "History: native major-iteration points (acceptance not guaranteed)."
            if library == "scipy"
            else "History: initial and accepted points."
        )
        problem = NamedAllocationProblem()
        # Deliberately use a different dictionary order: pack() resolves by name.
        initial_design = {"reserve": 0.6, "stream_b": 1.5, "stream_a": 0.5}
        controls = (
            {
                "first_step_fraction": 0.25,
                "feasibility_tol": 1e-6,
                "optimality_tol": 1e-6,
            }
            if library == "pysolver"
            else {}
        )
        solver = psv.OptimizationSolver(
            problem,
            library=library,
            method="slsqp",
            tolerance=1e-10,
            max_iterations=200,
            print_convergence=True,
            plot_convergence=False,
            update_on="iteration",
            check_kkt=not args.final_only,
            problem_scale=1.0 if args.scaled else None,
            extra_options=controls,
        )
        solver.solve(problem.pack(initial_design))
        solvers[label] = solver
        if args.final_only:
            assert solver.result.kkt_evaluations == 0
            assert solver.result.kkt_passed is None
        check_known_solution(solver)
        named_solution = problem.unpack(solver.x_final)
        readable_solution = ", ".join(
            f"{name}={value:.6g}" for name, value in named_solution.items()
        )
        print(f"\n{library} SLSQP: {readable_solution}")
        print(f"Solver success: {solver.success}; independent KKT check: passed.")
        solver.print_optimization_report(
            tol=1e-6,
            include_design_variables=True,
            include_constraints=True,
            include_kkt_conditions=True,
            include_multipliers=True,
            savefile=args.save_report is not None,
            filename=f"{library}_kkt_report.txt",
            output_dir=args.save_report,
        )
        if args.save_report is not None:
            print(
                f"Saved report: {(args.save_report / f'{library}_kkt_report.txt').resolve()}"
            )

    print("\n" + "=" * 80)
    print("Separate diagnostic examples: two bad designs, not additional solves.")
    print("=" * 80)
    final_solution = solver.x_final.copy()
    history_length = len(solver.convergence_history["x"])
    for label, design, expected_feasible in [
        (
            "Feasible but nonoptimal",
            {"stream_a": 0.8, "stream_b": 1.2, "reserve": 0.4},
            True,
        ),
        (
            "Lower-bound violation",
            {"stream_a": 1.2, "stream_b": 0.8, "reserve": -0.1},
            False,
        ),
    ]:
        point = problem.scale_physical_to_normalized(problem.pack(design))
        assessment = solver.evaluate_kkt_conditions(point, tol=1e-6)
        assert assessment["available"] and not assessment["passed"]
        assert (assessment["primal_feasibility"] <= 1e-6) == expected_feasible
        print(f"\n{label}: {design}")
        print(solver.make_kkt_optimality_report(point, tol=1e-6))

    np.testing.assert_array_equal(solver.x_final, final_solution)
    assert len(solver.convergence_history["x"]) == history_length
    print(
        "\nAll checks passed. Diagnostic model/derivative calls may increase counters;"
    )
    print("they do not add iterations or replace the optimized design.")
    print(
        "KKT checks establish first-order conditions, not a general proof of a minimum."
    )
    fig = plot_comparison(solvers)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
