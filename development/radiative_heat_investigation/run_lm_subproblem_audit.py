"""Independently audit custom LM subproblems at saved 500-node heat states.

Run from the repository root. Input states are preserved in lm_audit_states.json,
so rerunning the demo cannot change this audit. No production code is patched.
"""

import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import norm, svd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.demo_eqn_solving.demo_radiative_heat_collocation import RadiativeHeatProblem
from pysolver_view.solvers.root.linear_model import LinearModel


def audit_step(jacobian, residual, radius, independent_factors):
    """Check the convex trust-region optimality certificate and an independent step."""
    model = LinearModel(jacobian, residual)
    step = model.levenberg_marquardt_step(radius)
    damping = 0.0 if model._mu is None else model._mu * model._svd[0][0] ** 2
    gradient = jacobian.T @ residual
    shifted_product = jacobian.T @ (jacobian @ step) + damping * step
    equation_residual = shifted_product + gradient
    relative_residual = norm(equation_residual) / max(
        norm(gradient), norm(shifted_product), np.finfo(float).tiny
    )

    # Use GESVD rather than production's default GESDD, and solve the multiplier
    # equation in logarithmic coordinates with a separate Brent implementation.
    u, singular_values, vh = independent_factors
    projected = u.T @ residual
    full_step = -(vh.T @ (projected / singular_values))
    if norm(full_step) <= radius:
        reference_damping = 0.0
        reference_step = full_step
    else:

        def excess(log_damping):
            """Measure the independent damped step's excess over the requested radius."""
            coefficients = (
                singular_values * projected / (singular_values**2 + np.exp(log_damping))
            )
            return norm(coefficients) / radius - 1.0

        upper = norm(gradient) / radius
        root = brentq(excess, np.log(upper) - 100, np.log(upper) + 1, xtol=1e-13)
        reference_damping = np.exp(root)
        reference_step = -(
            vh.T
            @ (singular_values * projected / (singular_values**2 + reference_damping))
        )
    return {
        "radius": radius,
        "step_norm": norm(step),
        "relative_radius_error": (norm(step) - radius) / radius,
        "damping": damping,
        "reference_damping": reference_damping,
        "relative_step_difference": norm(step - reference_step) / norm(reference_step),
        "relative_normal_equation_residual": relative_residual,
        "normal_equation_residual_norm": float(norm(equation_residual)),
        "gradient_norm": float(norm(gradient)),
        "shifted_hessian_minimum_eigenvalue": float(singular_values[-1] ** 2 + damping),
        "trust_constraint_violation": float(max(0.0, norm(step) - radius)),
        "trust_complementarity_absolute": float(damping * abs(norm(step) - radius)),
        "predicted_reduction": float(
            -residual @ (jacobian @ step) - 0.5 * norm(jacobian @ step) ** 2
        ),
        "secular_iterations": model.stats["lm_secular_iterations"],
        "condition_number_2": float(singular_values[0] / singular_values[-1]),
        "smallest_singular_value": float(singular_values[-1]),
        "largest_singular_value": float(singular_values[0]),
        "qr_healthy": bool(model._qr_healthy),
        "minimum_temperature_direction": float(np.min(step[: step.size // 2])),
        "endpoint_temperature_direction": step[[0, step.size // 2 - 1]].tolist(),
    }


def conditioning(problem, temperature, singular_values=None):
    """Measure initial Jacobian conditioning and unscaled column norm disparities."""
    matrix = problem.gradient(problem.initial_guess(temperature))
    if singular_values is None:
        singular_values = svd(matrix, compute_uv=False, check_finite=False)
    column_norms = norm(matrix, axis=0)
    return dict(
        nodes=problem.num_nodes,
        unknowns=matrix.shape[1],
        condition_number_2=float(singular_values[0] / singular_values[-1]),
        largest_singular_value=float(singular_values[0]),
        smallest_singular_value=float(singular_values[-1]),
        minimum_column_norm=float(column_norms.min()),
        maximum_column_norm=float(column_norms.max()),
        column_norm_ratio=float(column_norms.max() / column_norms.min()),
        minimum_temperature_column_norm=float(column_norms[: problem.num_nodes].min()),
        maximum_temperature_column_norm=float(column_norms[: problem.num_nodes].max()),
        minimum_flux_column_norm=float(column_norms[problem.num_nodes :].min()),
        maximum_flux_column_norm=float(column_norms[problem.num_nodes :].max()),
    )


def main():
    """Audit initial and near-domain-boundary states from the failed controlled run."""
    snapshot = Path(__file__).with_name("lm_audit_states.json")
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    configuration = data["problem"]
    problem = RadiativeHeatProblem(
        configuration["degree"], configuration["heating"], configuration["radiation"]
    )
    results = []
    initial_conditioning = []
    for saved in data["states"]:
        state = np.asarray(saved["state"])
        residual = problem.residual(state)
        jacobian = problem.gradient(state)
        independent_factors = svd(jacobian, full_matrices=False, lapack_driver="gesvd")
        temperature = state[: problem.num_nodes]
        if saved["iteration"] == 0:
            coarse = RadiativeHeatProblem(48, problem.heating, problem.radiation)
            initial_conditioning = [
                conditioning(coarse, configuration["initial_temperature"]),
                conditioning(
                    problem,
                    configuration["initial_temperature"],
                    independent_factors[1],
                ),
            ]
        for radius in saved["requested_radii"]:
            row = audit_step(jacobian, residual, radius, independent_factors)
            row.update(
                state_label=saved["label"],
                accepted_iteration=saved["iteration"],
                base_residual_norm=norm(residual),
                base_temperature_minimum=float(temperature.min()),
                base_temperature_maximum=float(temperature.max()),
                base_endpoint_temperatures=temperature[[0, -1]].tolist(),
                base_residual_block_norms={
                    "fourier": float(norm(residual[: problem.num_nodes])),
                    "energy": float(norm(residual[problem.num_nodes : -2])),
                    "boundary": float(norm(residual[-2:])),
                },
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    payload = dict(
        description="Quadratic LM trust-region audit, no nonlinear solver rerun.",
        problem=configuration,
        versions=dict(numpy=np.__version__, scipy=scipy.__version__),
        input_snapshot=snapshot.name,
        certificate_equation="J.T @ (J @ step + residual) + damping * step = 0; damping >= 0; ||step|| <= radius",
        certificate_normalization="normal-equation residual 2-norm / max(||J.T @ residual||, ||J.T @ J @ step + damping * step||)",
        production_method="GESDD SVD with safeguarded Newton damping equation",
        independent_method="GESVD SVD with Brent damping equation in log coordinates",
        initial_conditioning=initial_conditioning,
        subproblem_checks=results,
    )
    target = Path(__file__).with_name("results_lm_subproblem.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(initial_conditioning, indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
