"""Separate MINPACK scaling, custom LM scaling, and initial-radius effects.

Run from any directory. This investigation leaves the demonstration and solver
implementations unchanged. Results are saved beside the script. Single-thread
BLAS defaults keep this small diagnostic reproducible on large desktop CPUs;
these elapsed times are observations, not a performance benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy
from scipy.optimize import root

import pysolver_view as psv
from demos.demo_eqn_solving.demo_radiative_heat_collocation import RadiativeHeatProblem


class RecordedProblem:
    """Count actual evaluations and the lowest temperature attempted by a backend."""

    def __init__(self, problem):
        """Keep the original physical equations and an initially empty evaluation log."""
        self.problem = problem
        self.nfev = 0
        self.njev = 0
        self.minimum_attempted_temperature = float("inf")
        self.trace = []

    def residual(self, state):
        """Record each trial before the physical positive-temperature domain check."""
        self.nfev += 1
        temperature, _ = self.problem.unpack(state)
        minimum = float(temperature.min())
        self.minimum_attempted_temperature = min(
            self.minimum_attempted_temperature, minimum
        )
        record = dict(
            call=self.nfev, minimum_temperature=minimum, valid=False, residual_norm=None
        )
        self.trace.append(record)
        values = self.problem.residual(state)
        record.update(valid=True, residual_norm=float(np.linalg.norm(values)))
        return values

    def gradient(self, state):
        """Count complete analytic Jacobians without adding model evaluations."""
        self.njev += 1
        return self.problem.gradient(state)


def run_native(problem, identity_scaling):
    """Run native MINPACK LM, preserving actual domain errors rather than masking them."""
    recorded = RecordedProblem(problem)
    options = dict(ftol=1e-12, xtol=1e-12, gtol=1e-12, maxiter=500)
    if identity_scaling:
        options["diag"] = np.ones(2 * problem.num_nodes)
    start = time.perf_counter()
    try:
        result = root(
            recorded.residual,
            problem.initial_guess(),
            jac=recorded.gradient,
            method="lm",
            options=options,
        )
        residual_norm = float(np.linalg.norm(result.fun))
        native_success = bool(result.success)
        message = str(result.message)
        final_minimum = float(result.x[: problem.num_nodes].min())
    except psv.DomainError as error:
        # Native root() exposes no accepted state after a callback exception.
        # Do not mislabel the last evaluated trial as its final solution.
        residual_norm = final_minimum = None
        native_success = False
        message = f"DomainError: {error}"
    return dict(
        configuration=(
            "scipy_identity_scaling" if identity_scaling else "scipy_default_scaling"
        ),
        native_success=native_success,
        root_success=bool(native_success and residual_norm <= 1e-8),
        residual_norm=residual_norm,
        minimum_final_temperature=final_minimum,
        minimum_attempted_temperature=recorded.minimum_attempted_temperature,
        nfev=recorded.nfev,
        njev=recorded.njev,
        elapsed_seconds=time.perf_counter() - start,
        message=message,
        evaluations=recorded.trace,
    )


def run_custom(problem, configuration, x_scale, initial_radius, max_radius):
    """Change only the fixed metric or radius, retaining raw physical residuals."""
    recorded = RecordedProblem(problem)
    start = time.perf_counter()
    result = psv.solve_levenberg_marquardt(
        recorded.residual,
        problem.initial_guess(),
        jac=recorded.gradient,
        options=dict(
            x_scale=x_scale,
            jacobian_scaling=False,  # Isolate the original manual-scaling ablation.
            initial_radius=initial_radius,
            max_radius=max_radius,
            residual_tol=1e-8,
            max_iterations=200,
            history_level="summary",
        ),
    )
    return dict(
        configuration=configuration,
        root_success=bool(result.success),
        residual_norm=float(np.linalg.norm(problem.residual(result.x))),
        minimum_final_temperature=float(result.x[: problem.num_nodes].min()),
        minimum_attempted_temperature=recorded.minimum_attempted_temperature,
        nfev=recorded.nfev,
        njev=recorded.njev,
        nit=int(result.nit),
        rejected=int(result.nrejected),
        invalid=int(result.ninvalid),
        scale_minimum=float(np.min(x_scale)),
        scale_maximum=float(np.max(x_scale)),
        initial_radius=initial_radius,
        max_radius=max_radius,
        elapsed_seconds=time.perf_counter() - start,
        message=result.message,
        linear_algebra=result.linear_algebra,
        history=result.history,
        evaluations=recorded.trace,
    )


def main():
    """Run the scaling ablations and record residual-based success, not just exit flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results_backend_comparison.json"),
    )
    args = parser.parse_args()
    problem = RadiativeHeatProblem(degree=args.points - 1)
    initial = problem.initial_guess()
    columns = np.linalg.norm(problem.gradient(initial), axis=0)
    fixed_column_scale = np.median(columns) / columns
    mesh_factor = np.sqrt(args.points / 49)
    results = []
    for identity in (False, True):
        result = run_native(problem, identity)
        results.append(result)
        print(
            result["configuration"],
            result["root_success"],
            result["residual_norm"],
            flush=True,
        )
    configurations = (
        ("custom_identity_initial_0.1", 1.0, 0.1, 100.0),
        ("custom_fixed_column_scaling", fixed_column_scale, 0.1, 100.0),
        ("custom_identity_mesh_radii", 1.0, 0.1 * mesh_factor, 100 * mesh_factor),
        ("custom_identity_large_radius", 1.0, 1000.0, 1000.0),
    )
    for configuration in configurations:
        result = run_custom(problem, *configuration)
        results.append(result)
        print(
            result["configuration"],
            result["root_success"],
            result["residual_norm"],
            flush=True,
        )
    payload = dict(
        points=args.points,
        unknowns=initial.size,
        numpy=np.__version__,
        scipy=scipy.__version__,
        root_residual_target=1e-8,
        note="Native default and identity scaling use MINPACK's factor=100; radii/metrics are not identical to custom LM.",
        results=results,
    )
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
