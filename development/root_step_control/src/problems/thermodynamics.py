"""A coupled heated-gas duct with real CoolProp property evaluations.

Each cell has unknown outlet temperature and pressure. The model balances a
prescribed heat input per unit mass and a lumped pressure loss proportional to
mass flux squared / density. It is a low-speed engineering timing workload,
not a full compressible-flow, choking, or two-phase model.

CoolProp is an optional dependency of this experiment, not of pysolver_view.
The public low-level interface and single-phase property derivatives are used:
https://coolprop.org/coolprop/LowLevelAPI.html#partial-derivatives
https://coolprop.org/coolprop/HighLevelAPI.html#partial-derivatives
"""

import numpy as np

from pysolver_view.solvers.root import DomainError
from .base import BenchmarkProblem

try:
    import CoolProp as CP
except ImportError:
    CP = None


class HeatedGasDuctProblem(BenchmarkProblem):
    """Solve 2*n_cells coupled energy and pressure-loss equations.

    Unknowns are interleaved ``[T_1, p_1, ..., T_N, p_N]`` in K and Pa.
    With fixed inlet enthalpy h_0 and pressure p_0, cell i satisfies

    ``h(T_i,p_i) - h(T_(i-1),p_(i-1)) - q_i = 0`` and
    ``p_(i-1) - p_i - 0.5*K*G**2 / rho(T_i,p_i) = 0``.

    q_i is heat per mass [J/kg], K is a dimensionless lumped loss coefficient,
    and G is prescribed mass flux [kg/(m^2*s)]. Energy equations neglect changes
    in kinetic energy, appropriate to this low-speed benchmark. Wall heat and
    resistance are prescribed, not fitted to produce a chosen solution.

    Every residual call performs exactly n_cells fresh CoolProp PT updates.
    Every analytic-Jacobian call also performs n_cells fresh PT updates and
    obtains four analytic derivatives per cell. There is no residual cache,
    artificial sleep, repeated dummy work, or reuse of a previous call's state.
    One AbstractState is constructed outside solve timing and each subsequent
    property call fully supplies pressure and temperature. Instances are not
    thread safe; concurrent solves need separate instances. Sequential solves
    can reuse the instance because each call fully establishes its state.
    """

    source = "https://coolprop.org/coolprop/LowLevelAPI.html#partial-derivatives"

    def __init__(
        self, n_cells=10, *, total_heat_per_mass=None, total_loss_coefficient=None
    ):
        """Configure the cell grid, physical loads, residual scales and reusable CoolProp state.

        n_cells >= 2 determines two unknowns per cell. Optional total_heat_per_mass
        [J/kg] and total_loss_coefficient are divided among cells. Without them, each
        cell receives 10,000 J/kg and loss coefficient 8. State construction and inlet
        property evaluation occur before any solver timing.
        """
        if CP is None:
            raise ImportError(
                "The thermodynamic timing case requires CoolProp; "
                "install it to run this workload."
            )
        if (
            isinstance(n_cells, bool)
            or not isinstance(n_cells, (int, np.integer))
            or n_cells < 2
        ):
            raise ValueError("n_cells must be an integer >= 2.")
        self.n_cells = int(n_cells)
        self.n = 2 * self.n_cells
        self.solution = None
        self.name = f"heated_nitrogen_duct_n{self.n}"
        self.description = (
            f"{self.n_cells}-cell low-speed heated nitrogen duct; "
            f"{self.n} coupled T/p unknowns, CoolProp HEOS properties "
            "and analytic thermodynamic derivatives."
        )
        self.fluid = "Nitrogen"
        self.backend = "HEOS"
        self.inlet_temperature = 320.0
        self.inlet_pressure = 2e6
        self.mass_flux = 300.0
        self.loss_coefficient = (
            8.0
            if total_loss_coefficient is None
            else float(total_loss_coefficient) / self.n_cells
        )
        heat_per_cell = (
            1e4
            if total_heat_per_mass is None
            else float(total_heat_per_mass) / self.n_cells
        )
        if not np.isfinite(self.loss_coefficient) or self.loss_coefficient <= 0:
            raise ValueError("Total resistance must be finite and positive.")
        if not np.isfinite(heat_per_cell) or heat_per_cell <= 0:
            raise ValueError("Total heat per mass must be finite and positive.")
        self.heat_per_mass = np.full(self.n_cells, heat_per_cell)
        self.description += (
            f" Total heat {float(np.sum(self.heat_per_mass)):g} J/kg; "
            f"total loss coefficient {self.n_cells * self.loss_coefficient:g}."
        )
        self.loss_constant = 0.5 * self.loss_coefficient * self.mass_flux**2
        self.energy_scale = 1e4
        self.pressure_residual_scale = 2e4
        self.x_scale = np.tile([20.0, 1e5], self.n_cells)
        # Uniform inlet state: no approximate solution is precomputed.
        self.x0 = np.tile([self.inlet_temperature, self.inlet_pressure], self.n_cells)
        super().__init__(self.name, self.x0, self.description, x_scale=self.x_scale)
        self._state = CP.AbstractState(self.backend, self.fluid)
        self._state.update(CP.PT_INPUTS, self.inlet_pressure, self.inlet_temperature)
        self.inlet_enthalpy = self._state.hmass()
        self.property_updates = 0

    def _properties(self, x, *, derivatives=False):
        """Evaluate all cell enthalpies and densities from the supplied T/p vector.

        When derivatives=True, also return dh/dT, dh/dp, drho/dT and drho/dp for each
        cell. Every call refreshes every state. Recognized invalid/non-gas property
        states raise DomainError; the instance is intended for sequential use.
        """
        point = np.asarray(x, dtype=float)
        if point.shape != (self.n,):
            raise ValueError(f"Expected a vector of shape {(self.n,)}.")
        if not np.all(np.isfinite(point)) or np.any(point <= 0):
            raise DomainError("Temperature and pressure must be finite and positive.")
        enthalpy = np.empty(self.n_cells)
        density = np.empty(self.n_cells)
        partials = np.empty((self.n_cells, 4)) if derivatives else None
        for i, (temperature, pressure) in enumerate(point.reshape(-1, 2)):
            self.property_updates += 1
            try:
                self._state.update(CP.PT_INPUTS, pressure, temperature)
                if self._state.phase() not in (
                    CP.iphase_gas,
                    CP.iphase_supercritical_gas,
                    CP.iphase_supercritical,
                ):
                    raise DomainError(
                        "The duct benchmark requires a homogeneous gas state."
                    )
                enthalpy[i] = self._state.hmass()
                density[i] = self._state.rhomass()
                if derivatives:
                    partials[i] = (
                        self._state.first_partial_deriv(CP.iHmass, CP.iT, CP.iP),
                        self._state.first_partial_deriv(CP.iHmass, CP.iP, CP.iT),
                        self._state.first_partial_deriv(CP.iDmass, CP.iT, CP.iP),
                        self._state.first_partial_deriv(CP.iDmass, CP.iP, CP.iT),
                    )
            except ValueError as error:
                # This narrow block contains CoolProp state/property calls,
                # not arbitrary application code whose errors could be hidden.
                raise DomainError(
                    f"Undefined nitrogen state in cell {i + 1}: {error}"
                ) from error
        return enthalpy, density, partials

    def physical_residuals(self, x):
        """Return unscaled conservation errors, in J/kg and Pa respectively."""
        enthalpy, density, _ = self._properties(x)
        pressures = np.asarray(x)[1::2]
        previous_enthalpy = np.r_[self.inlet_enthalpy, enthalpy[:-1]]
        previous_pressure = np.r_[self.inlet_pressure, pressures[:-1]]
        return (
            enthalpy - previous_enthalpy - self.heat_per_mass,
            previous_pressure - pressures - self.loss_constant / density,
        )

    def residual(self, x):
        """Return the interleaved dimensionless energy and pressure-loss residual vector
        of length 2*n_cells.
        """
        energy_error, pressure_error = self.physical_residuals(x)
        values = np.empty(self.n)
        values[::2] = energy_error / self.energy_scale
        values[1::2] = pressure_error / self.pressure_residual_scale
        return values

    def gradient(self, x):
        """Return the dense analytic Jacobian of the normalized equations using fresh
        property derivatives.
        """
        _, density, partials = self._properties(x, derivatives=True)
        jacobian = np.zeros((self.n, self.n))
        for i, (dh_dt, dh_dp, drho_dt, drho_dp) in enumerate(partials):
            row = 2 * i
            jacobian[row, row : row + 2] = [dh_dt, dh_dp]
            if i:
                jacobian[row, row - 2 : row] = -partials[i - 1, :2]
                jacobian[row + 1, row - 1] = 1.0
            loss_density_factor = self.loss_constant / density[i] ** 2
            jacobian[row + 1, row] = loss_density_factor * drho_dt
            jacobian[row + 1, row + 1] = -1 + loss_density_factor * drho_dp
        jacobian[::2] /= self.energy_scale
        jacobian[1::2] /= self.pressure_residual_scale
        return jacobian

    def solution_diagnostics(self, x):
        """Physical checks for post-timing verification, not part of solve time."""
        point = np.asarray(x, dtype=float)
        energy_error, pressure_error = self.physical_residuals(point)
        enthalpy, density, _ = self._properties(point)
        return {
            "outlet_temperature_K": float(point[-2]),
            "outlet_pressure_Pa": float(point[-1]),
            "max_energy_error_J_per_kg": float(np.max(np.abs(energy_error))),
            "max_pressure_error_Pa": float(np.max(np.abs(pressure_error))),
            "total_heat_error_J_per_kg": float(
                enthalpy[-1] - self.inlet_enthalpy - np.sum(self.heat_per_mass)
            ),
            "maximum_velocity_m_per_s": float(np.max(self.mass_flux / density)),
            "temperature_increases": bool(
                np.all(np.diff(np.r_[self.inlet_temperature, point[::2]]) > 0)
            ),
            "pressure_decreases": bool(
                np.all(np.diff(np.r_[self.inlet_pressure, point[1::2]]) < 0)
            ),
        }
