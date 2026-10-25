"""Check the heat demo's discretization, physics, and six public root solves."""

import numpy as np
import pytest
from scipy.integrate import quad

import pysolver_view as psv
from demos.demo_eqn_solving.demo_radiative_heat_collocation import (
    RadiativeHeatProblem,
    chebyshev_lobatto,
    solve_comparison,
    solve_reference,
    solver_options,
)

METHODS = ("newton_raphson", "levenberg_marquardt", "powell_hybrid")


@pytest.fixture(scope="module")
def heat_comparison():
    """Solve a fixed 49-node case, independent of the editable demo resolution."""
    problem = RadiativeHeatProblem(degree=48)
    reference = solve_reference(problem, tolerance=1e-10)
    runs = solve_comparison(problem)
    assert reference.success, reference.message
    assert len(runs) == 6
    results = {(run["method"], run["controlled"]): run["solver"] for run in runs}
    assert set(results) == {
        (method, controlled) for method in METHODS for controlled in (False, True)
    }
    return problem, reference, results


def test_chebyshev_derivatives_on_a_shifted_nonunit_interval():
    """Differentiation must have the correct sign, ordering, and interval scale."""
    nodes, differentiation = chebyshev_lobatto(12, left=-2.0, right=3.0)
    assert nodes.shape == (13,)
    assert differentiation.shape == (13, 13)
    assert np.all(np.diff(nodes) > 0)
    np.testing.assert_allclose(nodes[[0, -1]], [-2.0, 3.0], atol=1e-14)
    np.testing.assert_allclose(differentiation @ np.ones(13), 0.0, atol=1e-12)
    values = nodes**5 - 2 * nodes**3 + 3 * nodes - 4
    derivative = 5 * nodes**4 - 6 * nodes**2 + 3
    np.testing.assert_allclose(
        differentiation @ values, derivative, atol=2e-10, rtol=1e-12
    )


def test_analytic_jacobian_matches_complex_step_at_a_nonuniform_state():
    """Check every temperature/flux column and both boundary equations independently."""
    problem = RadiativeHeatProblem(degree=12)
    positions = problem.nodes
    temperature = 1.1 + 0.3 * np.sin(np.pi * positions) + 0.1 * positions
    flux = 0.2 * np.cos(2 * np.pi * positions) - 0.15
    state = np.r_[temperature, flux]
    numerical = np.empty((state.size, state.size))
    for column in range(state.size):
        perturbed = state.astype(complex)
        perturbed[column] += 1e-30j
        numerical[:, column] = np.imag(problem.residual(perturbed)) / 1e-30
    np.testing.assert_allclose(
        problem.gradient(state), numerical, atol=2e-12, rtol=2e-12
    )


def test_without_radiation_the_exact_conduction_profile_is_recovered():
    """An exactly representable parabola checks source, flux, and boundary signs."""
    heating = 7.0
    problem = RadiativeHeatProblem(degree=12, heating=heating, radiation=0.0)
    positions = problem.nodes
    temperature = 1 + 0.5 * heating * positions * (1 - positions)
    flux = positions - 0.5
    state = np.r_[temperature, flux]
    np.testing.assert_allclose(problem.residual(state), 0.0, atol=2e-12)
    dense = np.linspace(0, 1, 101)
    exact = np.vstack((1 + 0.5 * heating * dense * (1 - dense), dense - 0.5))
    np.testing.assert_allclose(problem.interpolate(state, dense), exact, atol=2e-12)


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_nonpositive_absolute_temperature_is_a_domain_error(temperature):
    """Unphysical temperatures must trigger the root solver's recovery mechanism."""
    problem = RadiativeHeatProblem(degree=8)
    state = problem.initial_guess()
    state[problem.num_nodes // 2] = temperature
    with pytest.raises(psv.DomainError):
        problem.residual(state)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("controlled", [False, True])
def test_all_methods_match_independent_adaptive_bvp(
    heat_comparison, method, controlled
):
    """A small algebraic residual must also reproduce the independently solved field."""
    problem, reference, runs = heat_comparison
    solver = runs[method, controlled]
    result = solver.result
    assert result.success, result.message
    assert np.linalg.norm(problem.residual(result.x)) <= 1e-8
    positions = np.linspace(0, 1, 1001)
    temperature, flux = problem.interpolate(result.x, positions)
    reference_temperature, reference_derivative = reference.sol(positions)
    np.testing.assert_allclose(temperature, reference_temperature, atol=2e-7, rtol=0)
    np.testing.assert_allclose(flux, -reference_derivative / 50.0, atol=2e-7, rtol=0)
    np.testing.assert_allclose(temperature[[0, -1]], 1.0, atol=1e-8)
    # At an interior maximum the radiative loss cannot exceed the source.
    # An interior minimum below the wall temperature contradicts the PDE.
    assert temperature.min() >= 1.0 - 1e-8
    assert temperature.max() <= 6.0**0.25 + 1e-8


@pytest.mark.parametrize("method", METHODS)
def test_first_trial_limit_and_full_step_baselines(heat_comparison, method):
    """Compare the actual attempted displacements, including rejected first trials."""
    problem, _, runs = heat_comparison
    initial = problem.initial_guess()
    newton_direction = np.linalg.solve(
        problem.gradient(initial), -problem.residual(initial)
    )
    unrestricted = runs[method, False].result.trials[0]
    controlled = runs[method, True].result.trials[0]
    np.testing.assert_allclose(
        unrestricted["x"] - initial, newton_direction, atol=2e-9, rtol=2e-9
    )
    assert np.linalg.norm(newton_direction) > 1.0
    displacement = np.linalg.norm(controlled["x"] - controlled["x_base"])
    assert 0 < displacement <= 0.1 * (1 + 1e-12)
    assert controlled["residual_norm"] < np.linalg.norm(problem.residual(initial))


@pytest.mark.parametrize("method", METHODS)
def test_controls_reduce_large_excursions_and_histories_count_accepted_moves(
    heat_comparison, method
):
    """Protect the demo's qualitative comparison without fixing exact iteration counts."""
    problem, _, runs = heat_comparison
    initial_norm = np.linalg.norm(problem.residual(problem.initial_guess()))
    peaks = {}
    for controlled in (False, True):
        solver = runs[method, controlled]
        result = solver.result
        trials = result.trials
        assert result.ntrials == len(trials)
        assert result.nit == sum(trial["accepted"] for trial in trials)
        assert result.nrejected == sum(not trial["accepted"] for trial in trials)
        assert len(solver.iteration_history) == result.nit + 1
        assert solver.iteration_count == result.nit
        assert result.derivative_nfev == 0
        assert result.nfev == result.ntrials + 1
        peaks[controlled] = max(
            initial_norm,
            *(trial["residual_norm"] for trial in trials if trial["valid"])
        )
        if controlled:
            residuals = np.array(
                [row["residual_norm"] for row in solver.iteration_history]
            )
            assert np.all(np.diff(residuals) <= 1e-12)
    # Rejected trials can increase the residual even with protective controls.
    assert peaks[False] > 10 * initial_norm
    assert peaks[True] < peaks[False] / 10


def test_spatial_refinement_reduces_off_grid_temperature_error(heat_comparison):
    """Separate discretization accuracy from the algebraic root stopping criterion."""
    _, reference, _ = heat_comparison
    positions = np.linspace(0, 1, 1001)
    expected = reference.sol(positions)[0]
    errors = []
    for degree in (8, 16, 32):
        problem = RadiativeHeatProblem(degree=degree)
        solver = psv.NonlinearSystemSolver(
            problem,
            library="pysolver",
            method="newton_raphson",
            tolerance=1e-10,
            max_iterations=500,
            options=solver_options("newton_raphson", controlled=True),
            print_convergence=False,
            update_on="iteration",
        )
        solver.solve(problem.initial_guess())
        assert solver.success, solver.message
        temperature = problem.interpolate(solver.x_final, positions)[0]
        errors.append(np.max(np.abs(temperature - expected)))
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10
    assert errors[2] < 1e-7


def test_converged_field_satisfies_integrated_energy_balance(heat_comparison):
    """Radiation plus net outward conductive flux must equal the volumetric heating."""
    problem, _, runs = heat_comparison
    state = runs["newton_raphson", True].result.x

    def radiation_loss(position):
        """Evaluate the radiative sink from the interpolated computed temperature."""
        temperature = problem.interpolate(state, np.asarray([position]))[0, 0]
        return 10.0 * (temperature**4 - 1)

    radiation, _ = quad(radiation_loss, 0, 1, epsabs=1e-8, epsrel=1e-10)
    _, flux = problem.unpack(state)
    net_conduction = 50.0 * (flux[-1] - flux[0])
    np.testing.assert_allclose(net_conduction + radiation, 50.0, atol=2e-6, rtol=0)
