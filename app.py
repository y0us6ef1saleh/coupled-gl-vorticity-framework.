from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import CoupledGLVModel, GLVParameters


st.set_page_config(
    page_title="Coupled GL–Vorticity Model",
    page_icon="〰️",
    layout="wide",
)

st.title("Coupled Ginzburg–Landau–Vorticity Model")
st.caption("Reduced 0D amplitude model with vorticity-dependent nonlinear damping")

st.info(
    "Model scope: this prototype solves the time-dependent amplitude equation "
    "using a supplied vorticity signal. It does not yet solve the spatial "
    "vorticity–velocity PDE system or perform 2D/3D CFD."
)

with st.expander("Governing equation", expanded=True):

    st.latex(r"""
    \frac{dA}{dt}
    =
    \gamma A
    -
    \left(\beta_0+\beta_{VS}|\omega(t)|^2\right)A^3
    +
    F(t)
    """)

    st.write(
        "The uploaded vorticity history is treated as prescribed input. "
        "The optimizer estimates the parameters by repeatedly integrating "
        "this governing ODE and comparing its solution with the measured amplitude."
    )
    st.markdown(
        "The uploaded vorticity history is treated as prescribed input. "
        "The optimizer estimates the parameters by repeatedly integrating "
        "this governing ODE and comparing its solution with the measured amplitude."
    )

uploaded_file = st.file_uploader(
    "Upload experimental or numerical data",
    type=["csv", "xlsx", "xls"],
    help="Required columns: time, amplitude, vorticity.",
)

with st.sidebar:

    st.header("Model Configuration")

    model_mode = st.radio(
        "Select model",
        [
            "Full Coupled Model",
            "Baseline Model (No Vorticity Coupling)"
        ]
    )

    fit_coupling = model_mode == "Full Coupled Model"

    st.divider()

    st.header("Initial parameter estimates")

    gamma_guess = st.number_input(
        "gamma",
        value=0.6,
        format="%.6f"
    )

    beta0_guess = st.number_input(
        "beta_0",
        min_value=1e-12,
        value=1.1,
        format="%.6f"
    )

    if fit_coupling:

        beta_vs_guess = st.number_input(
            "beta_VS",
            min_value=0.0,
            value=0.11,
            format="%.6f"
        )

    else:

        beta_vs_guess = 0.0

        st.metric(
            "beta_VS (Fixed)",
            "0.000000"
        )

    max_nfev = st.number_input(
        "Maximum optimizer evaluations",
        min_value=50,
        max_value=10000,
        value=2000,
        step=50
    )

    run_button = st.button(
        "Calibrate model",
        type="primary",
        use_container_width=True
    )
if uploaded_file is None:
    st.warning("Upload a CSV or Excel file to begin. The file must contain time, amplitude and vorticity columns.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)

    normalized = {str(c).strip().lower(): c for c in raw.columns}
    required = ("time", "amplitude", "vorticity")
    missing = [c for c in required if c not in normalized]
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        st.stop()

    data = pd.DataFrame({
        name: pd.to_numeric(raw[normalized[name]], errors="coerce")
        for name in required
    }).dropna().sort_values("time").reset_index(drop=True)

    if len(data) < 3:
        st.error("At least three valid rows are required.")
        st.stop()
    if data["time"].duplicated().any():
        st.error("The time column contains duplicate values.")
        st.stop()
    if not np.all(np.diff(data["time"].to_numpy()) > 0):
        st.error("Time values must be strictly increasing.")
        st.stop()

except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

st.subheader("Input data")
st.dataframe(data, use_container_width=True, hide_index=True)

preview = go.Figure()
preview.add_trace(go.Scatter(
    x=data["time"], y=data["amplitude"], mode="lines+markers",
    name="Measured amplitude"
))
preview.add_trace(go.Scatter(
    x=data["time"], y=data["vorticity"], mode="lines+markers",
    name="Prescribed vorticity", yaxis="y2"
))
preview.update_layout(
    xaxis_title="Time",
    yaxis=dict(title="Amplitude"),
    yaxis2=dict(title="Vorticity", overlaying="y", side="right"),
    legend=dict(orientation="h"),
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(preview, use_container_width=True)

if run_button:
    if beta0_guess <= 0 or beta_vs_guess < 0:
        st.error("beta_0 must be positive and beta_VS must be non-negative.")
        st.stop()

    initial = GLVParameters(
        gamma=float(gamma_guess),
        beta_0=float(beta0_guess),
        beta_VS=float(beta_vs_guess),
    )

    try:
        model = CoupledGLVModel(
            time=data["time"].to_numpy(),
            vorticity=data["vorticity"].to_numpy(),
            parameters=initial,
        )
        with st.spinner("Integrating the governing equation during parameter calibration..."):
            result = model.calibrate(
                observed_amplitude=data["amplitude"].to_numpy(),
                initial_guess=initial,
                max_nfev=int(max_nfev),
                fit_vorticity_coupling=fit_coupling,
             )

        st.subheader("Calibration results")
        st.write("Selected model:", model_mode)

        if not fit_coupling:
            st.success(
                "Baseline Model: beta_VS is fixed exactly at zero "
                "and excluded from optimization."
            )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("gamma", f"{result.parameters.gamma:.6g}")
        c2.metric("beta_0", f"{result.parameters.beta_0:.6g}")
        c3.metric("beta_VS", f"{result.parameters.beta_VS:.6g}")
        c4.metric("RMSE", f"{result.rmse:.5g}")
        c5.metric("R²", f"{result.r_squared:.5g}" if np.isfinite(result.r_squared) else "undefined")

        if result.success:
            st.success("Optimizer terminated successfully. This indicates numerical convergence, not physical validation.")
        else:
            st.warning(f"Optimizer did not report success: {result.message}")

        fitted = data.copy()
        fitted["predicted_amplitude"] = result.predicted_amplitude
        fitted["residual"] = result.residuals
        fitted["beta_effective"] = (
            result.parameters.beta_0
            + result.parameters.beta_VS * fitted["vorticity"].to_numpy() ** 2
        )

        comparison = go.Figure()
        comparison.add_trace(go.Scatter(
            x=fitted["time"], y=fitted["amplitude"],
            mode="markers", name="Measured amplitude"
        ))
        comparison.add_trace(go.Scatter(
            x=fitted["time"], y=fitted["predicted_amplitude"],
            mode="lines", name="Model solution"
        ))
        comparison.update_layout(
            xaxis_title="Time",
            yaxis_title="Amplitude",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(comparison, use_container_width=True)

        beta_plot = go.Figure()
        beta_plot.add_trace(go.Scatter(
            x=fitted["time"], y=fitted["beta_effective"],
            mode="lines+markers", name="Effective beta"
        ))
        beta_plot.update_layout(
            xaxis_title="Time",
            yaxis_title="beta_effective",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(beta_plot, use_container_width=True)

        st.subheader("Residuals")
        residual_plot = go.Figure()
        residual_plot.add_trace(go.Scatter(
            x=fitted["time"], y=fitted["residual"],
            mode="lines+markers", name="Model − measurement"
        ))
        residual_plot.add_hline(y=0, line_dash="dash")
        residual_plot.update_layout(
            xaxis_title="Time",
            yaxis_title="Residual",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(residual_plot, use_container_width=True)

        st.download_button(
            "Download calibrated data (CSV)",
            data=fitted.to_csv(index=False).encode("utf-8"),
            file_name="glv_calibrated_results.csv",
            mime="text/csv",
        )

        report = {
            "gamma": result.parameters.gamma,
            "beta_0": result.parameters.beta_0,
            "beta_VS": result.parameters.beta_VS,
            "rmse": result.rmse,
            "r_squared": result.r_squared,
            "optimizer_success": result.success,
            "optimizer_message": result.message,
            "model_scope": "0D amplitude equation with prescribed vorticity; not full CFD",
        }
        st.download_button(
            "Download calibration report (JSON)",
            data=__import__("json").dumps(report, indent=2),
            file_name="glv_calibration_report.json",
            mime="application/json",
        )

        st.caption(
            "Interpretation warning: a high R² alone does not establish that the "
            "coupling law is physically correct or that the model is novel. "
            "Use independent validation data and report parameter uncertainty."
        )

    except Exception as exc:
        st.exception(exc)
else:
    st.caption("After checking the uploaded data and initial guesses, click “Calibrate model”.")
