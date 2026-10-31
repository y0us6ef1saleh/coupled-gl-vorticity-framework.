from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import argparse
import json

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


Array = np.ndarray
Forcing = float | Callable[[float], float]


@dataclass
class GLVParameters:
    """Parameters of the reduced amplitude model."""
    gamma: float
    beta_0: float
    beta_VS: float

    def validate(self) -> None:
        values = np.asarray([self.gamma, self.beta_0, self.beta_VS], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("All model parameters must be finite.")
        if self.beta_0 <= 0:
            raise ValueError("beta_0 must be positive.")
        if self.beta_VS < 0:
            raise ValueError("beta_VS must be non-negative in this dissipative closure.")


@dataclass
class CalibrationResult:
    parameters: GLVParameters
    success: bool
    message: str
    cost: float
    rmse: float
    r_squared: float
    predicted_amplitude: Array
    residuals: Array


class CoupledGLVModel:
    """ Time-dependent amplitude model with vorticity-dependent cubic damping. The supplied vorticity signal is interpolated using a shape-preserving PCHIP interpolator. This avoids introducing artificial oscillations that can occur with high-order polynomial interpolation. """

    def __init__( self, time: Sequence[float], vorticity: Sequence[float], parameters: GLVParameters, forcing: Forcing = 0.0, ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.vorticity = np.asarray(vorticity, dtype=float)
        self.parameters = parameters
        self.forcing = forcing

        if self.time.ndim != 1 or self.vorticity.ndim != 1:
            raise ValueError("time and vorticity must be one-dimensional.")
        if len(self.time) != len(self.vorticity) or len(self.time) < 2:
            raise ValueError("time and vorticity must have equal lengths >= 2.")
        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.vorticity)):
            raise ValueError("Input data must be finite.")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing.")
        self.parameters.validate()

        self._omega = PchipInterpolator(self.time, self.vorticity, extrapolate=False)

    def omega_at(self, t: float) -> float:
        """Interpolated prescribed vorticity at time t."""
        return float(self._omega(t))

    def beta_effective(self, t: float, parameters: Optional[GLVParameters] = None) -> float:
        p = parameters or self.parameters
        omega = self.omega_at(t)
        return float(p.beta_0 + p.beta_VS * omega**2)

    def _forcing_at(self, t: float) -> float:
        value = self.forcing(t) if callable(self.forcing) else self.forcing
        return float(value)

    def rhs(self, t: float, y: Array, parameters: Optional[GLVParameters] = None) -> Array:
        """Right-hand side of the reduced governing equation."""
        p = parameters or self.parameters
        A = float(y[0])
        omega = self.omega_at(t)
        beta_eff = p.beta_0 + p.beta_VS * omega**2
        dA_dt = p.gamma * A - beta_eff * A**3 + self._forcing_at(t)
        return np.asarray([dA_dt], dtype=float)

    def simulate( self, initial_amplitude: float, parameters: Optional[GLVParameters] = None, evaluation_times: Optional[Sequence[float]] = None, rtol: float = 1e-8, atol: float = 1e-10, ) -> Array:
        """Integrate the governing equation and return A at requested times."""
        p = parameters or self.parameters
        p.validate()

        if not np.isfinite(initial_amplitude):
            raise ValueError("initial_amplitude must be finite.")

        t_eval = self.time if evaluation_times is None else np.asarray(evaluation_times, dtype=float)
        if t_eval.ndim != 1 or len(t_eval) == 0:
            raise ValueError("evaluation_times must be a non-empty 1D sequence.")
        if np.any(np.diff(t_eval) <= 0):
            raise ValueError("evaluation_times must be strictly increasing.")
        if t_eval[0] < self.time[0] or t_eval[-1] > self.time[-1]:
            raise ValueError("evaluation_times must lie within the vorticity-data interval.")

        solution = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, p),
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=[float(initial_amplitude)],
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"ODE integration failed: {solution.message}")
        return solution.y[0]

    def calibrate(
        self,
        observed_amplitude: Sequence[float],
        initial_guess: GLVParameters = GLVParameters(0.5, 1.0, 0.1),
        lower_bounds: Sequence[float] = (0.0, 1e-12, 0.0),
        upper_bounds: Sequence[float] = (np.inf, np.inf, np.inf),
        max_nfev: int = 500,
        fit_vorticity_coupling: bool = True,
        ) -> CalibrationResult:

        observed = np.asarray(observed_amplitude, dtype=float)

        if observed.shape != self.time.shape:
            raise ValueError("observed_amplitude must have the same length as time.")

        if not np.all(np.isfinite(observed)):
            raise ValueError("observed_amplitude must contain finite values.")

        y0 = float(observed[0])

        if fit_vorticity_coupling:
            # Full model: optimize all three parameters
            x0 = np.asarray([
                initial_guess.gamma,
                initial_guess.beta_0,
                initial_guess.beta_VS
            ])

        lb = np.asarray(lower_bounds, dtype=float)
        ub = np.asarray(upper_bounds, dtype=float)

        def unpack(x):
            return GLVParameters(*map(float, x))

    else:
        # Baseline model: beta_VS is fixed exactly at zero
        x0 = np.asarray([
            initial_guess.gamma,
            initial_guess.beta_0
        ])

        lb = np.asarray(lower_bounds[:2], dtype=float)
        ub = np.asarray(upper_bounds[:2], dtype=float)

        def unpack(x):
            return GLVParameters(
                gamma=float(x[0]),
                beta_0=float(x[1]),
                beta_VS=0.0
            )

    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("Initial guess must lie inside parameter bounds.")

    def residual_function(x):
        p = unpack(x)

        try:
            prediction = self.simulate(
                y0,
                parameters=p,
                evaluation_times=self.time
            )

            residual = prediction - observed

            if not np.all(np.isfinite(residual)):
                raise FloatingPointError

            return residual

        except (RuntimeError, ValueError, FloatingPointError):
            return np.full_like(observed, 1e12, dtype=float)

    fit = least_squares(
        residual_function,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_nfev,
        x_scale="jac",
    )

    fitted_parameters = unpack(fit.x)

    predicted = self.simulate(
        y0,
        parameters=fitted_parameters,
        evaluation_times=self.time
    )

    residuals = predicted - observed

    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((observed - np.mean(observed))**2))

    r_squared = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0 else float("nan")
    )

    return CalibrationResult(
        parameters=fitted_parameters,
        success=bool(fit.success),
        message=str(fit.message),
        cost=float(fit.cost),
        rmse=rmse,
        r_squared=r_squared,
        predicted_amplitude=predicted,
        residuals=residuals,
    )

        fit = least_squares(
            residual_function,
            x0=x0,
            bounds=(lb, ub),
            method="trf",
            max_nfev=max_nfev,
            x_scale="jac",
        )

        fitted_parameters = GLVParameters(*map(float, fit.x))
        fitted_parameters.validate()
        predicted = self.simulate(y0, parameters=fitted_parameters, evaluation_times=self.time)
        residuals = predicted - observed

        rmse = float(np.sqrt(np.mean(residuals**2)))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((observed - np.mean(observed))**2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return CalibrationResult(
            parameters=fitted_parameters,
            success=bool(fit.success),
            message=str(fit.message),
            cost=float(fit.cost),
            rmse=rmse,
            r_squared=r_squared,
            predicted_amplitude=predicted,
            residuals=residuals,
        )


def load_experiment_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV with time, amplitude and vorticity columns."""
    df = pd.read_csv(path)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    required = ("time", "amplitude", "vorticity")
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(missing)
            + ". Required columns: time, amplitude, vorticity."
        )

    data = pd.DataFrame({
        name: pd.to_numeric(df[normalized[name]], errors="coerce")
        for name in required
    }).dropna()

    if len(data) < 3:
        raise ValueError("At least three valid data rows are required.")
    data = data.sort_values("time").reset_index(drop=True)
    if data["time"].duplicated().any():
        raise ValueError("Duplicate time values are not allowed.")
    return data


def run_calibration( csv_path: str | Path, output_path: str | Path = "glv_calibrated_results.csv", initial_guess: GLVParameters = GLVParameters(0.5, 1.0, 0.1), ) -> CalibrationResult:
    """Load measurements, calibrate the governing ODE, and save predictions."""
    data = load_experiment_csv(csv_path)
    model = CoupledGLVModel(
        time=data["time"].to_numpy(),
        vorticity=data["vorticity"].to_numpy(),
        parameters=initial_guess,
    )
    result = model.calibrate(data["amplitude"].to_numpy(), initial_guess=initial_guess)

    output = data.copy()
    output["predicted_amplitude"] = result.predicted_amplitude
    output["residual"] = result.residuals
    output["beta_effective"] = [
        result.parameters.beta_0 + result.parameters.beta_VS * w**2
        for w in data["vorticity"].to_numpy()
    ]
    output.to_csv(output_path, index=False)

    report = {
        "parameters": asdict(result.parameters),
        "success": result.success,
        "message": result.message,
        "rmse": result.rmse,
        "r_squared": result.r_squared,
        "cost": result.cost,
        "scope_note": (
            "0D amplitude equation with prescribed vorticity; "
            "not a solution of the full vorticity-velocity PDE system."
        ),
    }
    report_path = Path(output_path).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Coupled GL–Vorticity calibration ===")
    print(f"gamma = {result.parameters.gamma:.8g}")
    print(f"beta_0 = {result.parameters.beta_0:.8g}")
    print(f"beta_VS = {result.parameters.beta_VS:.8g}")
    print(f"RMSE = {result.rmse:.8g}")
    print(f"R^2 = {result.r_squared:.8g}")
    print(f"Results: {output_path}")
    print(f"Report: {report_path}")
    print("\nScope: prescribed-vorticity 0D reduction; not full CFD.")
    return result


def _self_test() -> None:
    """Small numerical sanity checks; these are not physical validation."""
    t = np.linspace(0.0, 2.0, 101)
    omega = np.zeros_like(t)
    p = GLVParameters(gamma=1.0, beta_0=2.0, beta_VS=0.0)
    model = CoupledGLVModel(t, omega, p)

    # With beta_VS=0 and omega=0, the equation reduces to dA/dt=gamma*A-beta_0*A^3.
    prediction = model.simulate(initial_amplitude=0.2)
    assert prediction.shape == t.shape
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0)

    # With beta_VS=0, changing the supplied vorticity must not change the solution.
    omega2 = np.linspace(0.0, 5.0, len(t))
    model2 = CoupledGLVModel(t, omega2, p)
    prediction2 = model2.simulate(initial_amplitude=0.2)
    assert np.allclose(prediction, prediction2, rtol=1e-6, atol=1e-8)

    # Positive beta_VS must increase cubic damping when vorticity is nonzero.
    p_coupled = GLVParameters(gamma=1.0, beta_0=2.0, beta_VS=0.5)
    model3 = CoupledGLVModel(t, omega2, p_coupled)
    prediction3 = model3.simulate(initial_amplitude=0.2)
    assert prediction3[-1] < prediction2[-1]

    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the reduced GL-vorticity model.")
    parser.add_argument("csv", nargs="?", help="CSV file with time, amplitude, vorticity.")
    parser.add_argument("--output", default="glv_calibrated_results.csv")
    parser.add_argument("--test", action="store_true", help="Run built-in numerical sanity checks.")
    args = parser.parse_args()

    if args.test:
        _self_test()
    if args.csv:
        run_calibration(args.csv, args.output)
    if not args.test and not args.csv:
        parser.error("Provide a CSV file or use --test.")


if __name__ == "__main__":
    main()
