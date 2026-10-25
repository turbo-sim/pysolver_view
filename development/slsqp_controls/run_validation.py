"""Audit tolerance gates and inconsistent constraints without turbine models."""

import argparse
import json
from pathlib import Path

import pysolver_view as psv
from pysolver_view.optimization.problems import StudyProblem
from pysolver_view.diagnostics.slsqp_audit import audit_solution


def main():
    """Run five distinct stopping/acceptance profiles on four mathematical cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_validation",
    )
    args = parser.parse_args()
    profiles = {
        "loose": (1e-5, 1e-3, 1e-2),
        "standard": (1e-12, 1e-6, 1e-4),
        "strict": (1e-13, 1e-9, 1e-6),
        "feasibility_gate": (1e-5, 1e-12, 1e-2),
        "optimality_gate": (1e-5, 1e-3, 1e-10),
    }
    records = []
    for case in ["rosenbrock", "constrained_rosenbrock", "square_root", "inconsistent"]:
        for name, (ftol, feasibility, optimality) in profiles.items():
            problem = StudyProblem(case)
            solver = psv.OptimizationSolver(
                problem,
                library="pysolver",
                method="slsqp",
                tolerance=ftol,
                max_iterations=100,
                update_on="iteration",
                print_convergence=False,
                extra_options=dict(
                    first_step_fraction=1.0,
                    step_growth_factor=None,
                    domain_backtracking=True,
                    feasibility_tol=feasibility,
                    optimality_tol=optimality,
                ),
            )
            solver.solve(problem.initial)
            assessment = solver.result.kkt
            audit = audit_solution(problem, solver, assessment)
            row = dict(
                case=case,
                profile=name,
                ftol=ftol,
                feasibility_tol=feasibility,
                optimality_tol=optimality,
                success=bool(solver.success),
                solver_success=bool(solver.result.solver_success),
                kkt_passed=bool(solver.result.kkt_passed),
                kkt_ratio=float(assessment["kkt_ratio"]),
                nfev=solver.result.nfev,
                audit=audit,
                message=solver.message,
            )
            records.append(row)
            print(
                f"{case:24} {name:18} native={row['solver_success']} accepted={row['success']} audit={audit['passed']}"
            )
            assert audit["passed"], row
            if case == "inconsistent":
                assert not solver.success
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Independent tolerance and rejection audit",
        "",
        "Every audit reconstructs KKT using fresh model values and independently implemented formulas.",
        "An audit pass does not mean optimization success. Inconsistent problems must be rejected.",
        "",
        "| Case | Profile | Native success | Accepted | Audit passed | KKT ratio |",
        "|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['case']} | {row['profile']} | {row['solver_success']} | {row['success']} | {row['audit']['passed']} | {row['kkt_ratio']:.3g} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
