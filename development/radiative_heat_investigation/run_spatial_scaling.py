"""Read-only experiments separating radiation-demo merit scaling from root accuracy."""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from scipy.linalg import svdvals
import pysolver_view as psv
from demos.demo_eqn_solving.demo_radiative_heat_collocation import RadiativeHeatProblem


def clenshaw_curtis_weights(nodes):
    """Return positive quadrature weights on [0,1] for Chebyshev-Lobatto nodes."""
    degree = nodes - 1
    angles = np.pi * np.arange(nodes) / degree
    interior = np.arange(1, degree)
    values = np.ones(degree - 1)
    weights = np.empty(nodes)
    if degree % 2 == 0:
        weights[[0, -1]] = 1 / (degree**2 - 1)
        for frequency in range(1, degree // 2):
            values -= (
                2 * np.cos(2 * frequency * angles[interior]) / (4 * frequency**2 - 1)
            )
        values -= np.cos(degree * angles[interior]) / (degree**2 - 1)
    else:
        weights[[0, -1]] = 1 / degree**2
        for frequency in range(1, (degree + 1) // 2):
            values -= (
                2 * np.cos(2 * frequency * angles[interior]) / (4 * frequency**2 - 1)
            )
    weights[interior] = 2 * values / degree
    return weights / 2


def run_case(
    nodes,
    boundary_weight=1.0,
    radius=0.1,
    normalized=False,
    fixed_boundary=False,
    quadrature=False,
):
    """Compare positive row reweighting/fixed boundaries, auditing original residuals."""
    problem = RadiativeHeatProblem(degree=nodes - 1)
    weight = np.ones(2 * nodes)
    weight[-2:] = boundary_weight
    if normalized:
        weight[:-2] /= np.sqrt(nodes)
    if quadrature:
        spatial_weights = clenshaw_curtis_weights(nodes)
        weight[:nodes] = np.sqrt(spatial_weights)
        weight[nodes:-2] = np.sqrt(spatial_weights[1:-1])
    initial = problem.initial_guess()
    if fixed_boundary:
        kept = np.r_[np.arange(1, nodes - 1), np.arange(nodes, 2 * nodes)]

        def unpack(state):
            """Insert fixed boundary temperatures in the original nodal vector."""
            full = np.ones(2 * nodes)
            full[kept] = state
            return full

        def residual(state):
            """Evaluate all PDE equations after eliminating boundary unknowns."""
            return problem.residual(unpack(state))[:-2]

        def gradient(state):
            """Restrict the exact Jacobian consistently with boundary elimination."""
            return problem.gradient(unpack(state))[:-2, kept]

        initial = initial[kept]
    else:

        def unpack(state):
            """Keep the full vector for the row-weighted formulation."""
            return state

        def residual(state):
            """Apply positive row weights without changing the root set."""
            return weight * problem.residual(state)

        def gradient(state):
            """Apply the same weights to Jacobian rows."""
            return weight[:, None] * problem.gradient(state)

    start = time.perf_counter()
    result = psv.solve_root(
        residual,
        initial,
        jac=gradient,
        method="lm",
        options=dict(
            initial_radius=radius,
            jacobian_scaling=False,  # Isolate residual weighting from automatic scaling.
            max_iterations=300,
            residual_tol=1e-10,
            history_level="full",
        ),
    )
    state = unpack(result.x)
    temperature, _ = problem.unpack(state)
    raw_residual = float(np.linalg.norm(problem.residual(state)))
    record = dict(
        nodes=nodes,
        boundary_weight=boundary_weight,
        initial_radius=radius,
        normalized=normalized,
        fixed_boundary=fixed_boundary,
        quadrature=quadrature,
        success=bool(result.success),
        nit=result.nit,
        nfev=result.nfev,
        ninvalid=result.ninvalid,
        raw_residual=raw_residual,
        original_root_passed=bool(raw_residual <= 1e-8),
        original_residual_tolerance=1e-8,
        scaled_residual=result.residual_norm,
        temperature_ends=temperature[[0, -1]].tolist(),
        temperature_min=float(temperature.min()),
        temperature_max=float(temperature.max()),
        raw_boundary_errors=(temperature[[0, -1]] - 1).tolist(),
        weighted_tolerance=1e-10,
        min_row_weight=float(weight.min()),
        max_row_weight=float(weight.max()),
        time=time.perf_counter() - start,
        message=result.message,
    )
    print(json.dumps(record), flush=True)
    return record


def matrix_diagnostics(nodes):
    """Measure conditioning and boundary-merit share at the common uniform guess."""
    problem = RadiativeHeatProblem(degree=nodes - 1)
    state = problem.initial_guess()
    matrix = problem.gradient(state)
    residual = problem.residual(state)
    singular_values = svdvals(matrix, check_finite=False)
    weighted_matrix = matrix.copy()
    weighted_matrix[-2:] *= np.sqrt(nodes)
    weighted_singular_values = svdvals(weighted_matrix, check_finite=False)
    return dict(
        nodes=nodes,
        D_inf_norm=float(np.linalg.norm(problem.D, np.inf)),
        initial_jacobian_cond2=float(singular_values[0] / singular_values[-1]),
        sigma_max=float(singular_values[0]),
        sigma_min=float(singular_values[-1]),
        weighted_cond2=float(
            weighted_singular_values[0] / weighted_singular_values[-1]
        ),
        initial_boundary_fraction_squared_norm=float(
            np.linalg.norm(residual[-2:]) ** 2 / np.linalg.norm(residual) ** 2
        ),
        constant_derivative_error=float(
            np.linalg.norm(problem.D @ np.ones(nodes), np.inf)
        ),
    )


def main():
    """Run representative mesh and boundary-weight variants and save scalar evidence."""
    print("Each run stops at weighted residual 2-norm <= 1e-10.")
    print(
        "raw_residual always audits the original unweighted PDE and boundary equations."
    )
    print(
        "Row scaling preserves roots, but weighted termination alone is not an original-residual guarantee."
    )
    records = []
    for nodes in [49, 100, 200, 250, 251, 500]:
        records.append(run_case(nodes))
    for weight in [2, 4, 10, np.sqrt(500)]:
        records.append(run_case(500, boundary_weight=float(weight)))
    records.append(run_case(500, normalized=True))
    records.append(run_case(500, fixed_boundary=True))
    records.append(run_case(500, boundary_weight=np.sqrt(500), radius=100))
    records.append(run_case(500, quadrature=True))
    output = Path(__file__).parent
    (output / "results_spatial_scaling.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    conditioning = [matrix_diagnostics(nodes) for nodes in (49, 100, 250, 500)]
    (output / "results_spatial_conditioning.json").write_text(
        json.dumps(conditioning, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
