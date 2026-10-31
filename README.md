# Coupled Ginzburg–Landau–Vorticity Framework
### Interactive Dashboard, Parameter Calibration & 3D Solver Development

**Live Dashboard:** [Open the Streamlit Application](https://coupled-gl-vorticity-framework-aagsun8rgwpyy55vbivktb.streamlit.app/)

---

## 🧬 Overview

The **Coupled Ginzburg–Landau–Vorticity (GLV) Framework** is an exploratory mathematical and computational framework designed to investigate the interaction between amplitude dynamics and vorticity-dependent nonlinear effects.

The current public repository provides an interactive scientific dashboard and parameter-calibration environment. A separate three-dimensional numerical solver is under active academic development.

The framework introduces a vorticity-dependent nonlinear saturation coefficient:

$$
\beta(\boldsymbol{\omega}) = \beta_0 + \beta_{VS}|\boldsymbol{\omega}|^2
$$

where:

- $\beta_0$ represents the baseline nonlinear saturation coefficient.
- $\beta_{VS}$ controls the coupling between vorticity magnitude and nonlinear saturation.
- $\boldsymbol{\omega}$ denotes the local vorticity vector.

The purpose of this coupling is to investigate how changes in the vorticity field may influence the evolution and saturation of an amplitude field.

The framework is being developed with potential applications in fluid-mechanical systems, including aerodynamic and hemodynamic flow modeling. Its predictive capabilities and applicability to specific physical systems require further numerical and experimental validation.

---

## 🖥️ Current Public Component: Interactive Dashboard

The public repository currently hosts a **Streamlit-based, reduced-order 0D prototype**.

The dashboard is intended to demonstrate the mathematical framework, parameter interaction, and data-calibration workflow without requiring access to high-performance computing resources.

### Main capabilities

- **Theoretical Simulation Panel:** Generates amplitude profiles and provides analytical benchmark comparisons, including Tollmien–Schlichting wave-related examples.
- **Data Ingestion:** Supports uploading experimental or numerical datasets through CSV and Excel files.
- **Parameter Calibration:** Uses least-squares optimization and curve fitting to estimate model parameters from supplied data.
- **Calibration Diagnostics:** Calculates residuals and error metrics to assess agreement between model predictions and input datasets.
- **Multi-Domain Design:** Provides a framework intended for investigating aerodynamic and hemodynamic applications.
- **AI-Assisted Interpretation:** Includes an optional AI module for interpreting calibration results, with a fallback mechanism for environments without an active API key.

The dashboard is a reduced-order research prototype. Its calibration results should not be interpreted as independent experimental validation or proof of predictive accuracy.

---

## ⚙️ Mathematical Framework

The reduced-order amplitude model is expressed as:

$$
\frac{dA}{dt}= \gamma A -
\left(
\beta_0+\beta_{VS}|\boldsymbol{\omega}(t)|^2
\right)A^3
+
F(t)
$$

where:

- $A(t)$ is the amplitude variable.
- $\gamma$ is the linear growth coefficient.
- $\beta_0$ is the baseline nonlinear saturation coefficient.
- $\beta_{VS}$ is the vorticity-sensitivity coupling coefficient.
- $\boldsymbol{\omega}(t)$ represents the prescribed or supplied vorticity input.
- $F(t)$ represents an external forcing term.

This reduced-order formulation is intended for parameter exploration and calibration. It is distinct from the full spatially resolved three-dimensional fluid-dynamics formulation under development.

---

## 🔬 Research Development: 3D Numerical Solver

A separate three-dimensional numerical solver is currently under active academic development.

The solver is designed to investigate the spatially resolved evolution of coupled amplitude and vorticity fields.

### Planned and developing numerical components

- Three-dimensional periodic computational domains.
- WENO5-based conservative flux reconstruction for nonlinear advection.
- Fourier spectral methods for differential operators.
- Biot–Savart velocity reconstruction from vorticity.
- Vorticity stretching and diffusion terms.
- Helmholtz projection to control the divergence-free vorticity constraint.
- Second-order Heun time integration.
- CPU-based numerical testing and GPU acceleration using CUDA/CuPy.

The numerical implementation is undergoing verification and validation. Its stability, convergence properties, and physical applicability must be assessed through systematic numerical tests before scientific conclusions are drawn.

### Computational architecture

The full 3D solver is intended for execution on high-performance computing (HPC) systems and GPU-accelerated machines due to its computational and memory requirements.

The complete solver source code is maintained separately from this public dashboard repository during the ongoing research and development process.

The separation also allows the public dashboard to remain lightweight and accessible while the computational backend evolves.

---

## 🔒 Repository Scope & Intellectual Property

This public repository currently contains the interactive dashboard and its associated calibration framework.

The full 3D numerical solver is not included in this repository. It is maintained separately during active academic development and prior to the completion of the intended research and publication process.

The distinction between the publicly available reduced-order prototype and the developing 3D solver is intentional.

---

## 🚀 Future Development

Planned development directions include:

1. Systematic numerical verification of the 3D solver.
2. Grid- and time-step convergence studies.
3. Comparison against established analytical solutions and benchmark problems.
4. Further investigation of the vorticity-dependent nonlinear coupling.
5. Evaluation against suitable experimental or independently generated datasets.
6. Integration of the Streamlit dashboard with remote HPC and GPU-based computational backends.
7. Expansion of the framework's potential applications in fluid mechanics.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Interactive Dashboard | Streamlit, Python |
| Numerical Processing | NumPy, SciPy |
| Optimization | SciPy Optimization and Least-Squares Methods |
| Data Handling | Pandas |
| Visualization | Matplotlib, Plotly |
| 3D Solver Development | NumPy, SciPy, CUDA/CuPy |
| Planned HPC Acceleration | GPU Computing |

---

## ⚠️ Scientific Scope & Disclaimer

The Coupled Ginzburg–Landau–Vorticity Framework is an ongoing research project.

The public dashboard demonstrates a reduced-order mathematical prototype and parameter-calibration workflow. The 3D solver is under development and has not yet been established as a fully validated predictive model.

Calibration quality, numerical stability, convergence, and physical relevance must be evaluated independently before interpreting results as evidence of a physical discovery or validated engineering application.

---

## 📄 Project Status

**Public Dashboard:** Available online.

**Reduced-Order Calibration Framework:** Available for exploration.

**3D Numerical Solver:** Under active development and numerical verification.

**HPC Integration:** Planned.

---

*An ongoing computational research project exploring vorticity-dependent nonlinear amplitude dynamics.*

## 🏃 How to Run the UI Locally
1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

