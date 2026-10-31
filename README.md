# Coupled Ginzburg–Landau Vorticity Framework (Dashboard & Calibration UI)

https://coupled-gl-vorticity-framework-aagsun8rgwpyy55vbivktb.streamlit.app/

This repository contains the interactive scientific dashboard and optimization framework for the **Coupled Ginzburg–Landau Vorticity System**, a reduced-order engineering tool developed to predict boundary-layer transitions in aerodynamics and hemodynamics.

## 🧬 Overview

The core framework implements a coupled GL–vorticity system where the Landau coefficient dynamically adapts based on local vorticity magnitude:

$$ \beta(\omega) = \beta_0 + \beta_{VS}|\omega|^2 $$

This dashboard bridges the theoretical mathematical framework with an automated Data Ingestion & Calibration Engine, allowing aerospace engineers and medical researchers to upload empirical databases (e.g., NACA 0012 wind tunnel logs or blood viscosity charts) and automatically calibrate physics-based constants using least-squares curve fitting.
mathematical framework with an automated **Data Ingestion & Calibration Engine**, allowing aerospace engineers and medical researchers to upload empirical databases (e.g., NACA 0012 wind tunnel logs or blood viscosity charts) and automatically calibrate physics-based constants using least-squares curve fitting.

### ⚠️ Note on Deployment Architecture & Intellectual Property

This repository currently hosts **one component**:
1. **The Interactive Dashboard (Streamlit):** A lightweight, web-based 0D prototype (running on free cloud infrastructure) designed to demonstrate the user interface, real-time parameter coupling, and automated data calibration.

**Regarding the Full HPC Backend:**
2. **The 3D WENO5 Numerical Solver:** The complete 3D WENO5 numerical solver is currently under active academic development. Due to intellectual property considerations prior to academic publication, the full HPC source code is kept private and is **not** hosted on this public repository. 

Due to the extreme computational intensity (RAM and CPU requirements) of 3D fluid dynamics simulations, the full 3D solver is designed to be executed on university-grade clusters or GPU-accelerated machines. Future work will focus on connecting this Streamlit UI to cloud-based HPC instances for remote execution.

## 🚀 Features
- **Theoretical Simulation Panel:** Real-time generation of Tollmien–Schlichting wave amplitude profiles and exact analytical benchmarks.
- **Generic Data Ingestion:** Drag-and-drop CSV/Excel database loader.
- **Automated AI Optimization Engine:** Dynamic parameter mapping and automated calculation of residuals and training errors against experimental validation criteria.
- **Multi-Domain Framework:** Structurally designed to support both Aerodynamic airfoils and Hemodynamic blood flow analysis.
- **AI Integration:** Features an integrated AI module to analyze the physical implications of the calibration results, designed with a secure fallback mechanism for environments without active API keys.

## 🛠️ Tech Stack
- **Frontend & UI:** Streamlit (Python)
- **Mathematical Processing:** NumPy, SciPy (Optimization & Curve Fitting)
- **Data & Visualizations:** Pandas, Plotly
- **Planned Backend Integration:** GPU-accelerated WENO5 / SSP-RK3 solver via CUDA/CuPy (Private Repository).

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

