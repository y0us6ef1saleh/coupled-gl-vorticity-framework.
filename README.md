# Coupled Ginzburg–Landau Vorticity Framework (Dashboard & Calibration UI)

https://coupled-gl-vorticity-framework-aagsun8rgwpyy55vbivktb.streamlit.app/

This repository contains the interactive scientific dashboard and optimization framework for the **Coupled Ginzburg–Landau Vorticity System**, a reduced-order engineering tool developed to predict boundary-layer transitions in aerodynamics and hemodynamics.

## 🧬 Overview
The core framework implements a coupled GL–vorticity system where the Landau coefficient dynamically adapts based on local vorticity magnitude:
$$\beta(\omega) = \beta_0 + \beta_{VS}|\omega|^2$$

This dashboard bridges the high-performance simulation backend with an automated **Data Ingestion & Calibration Engine**, allowing aerospace engineers and medical researchers to upload empirical databases (e.g., NACA 0012 wind tunnel logs or blood viscosity charts) and automatically calibrate physics-based constants using least-squares curve fitting.

### ⚠️ Note on Deployment
This Streamlit application is a simple, preliminary prototype designed to demonstrate the user interface and the 0D mathematical coupling. The full 3D WENO5 numerical solver is highly computationally intensive. Therefore, we will integrate the full solver and deploy it using larger-scale computational machines in the future.

## 🚀 Features
- **Theoretical Simulation Panel:** Real-time generation of Tollmien–Schlichting wave amplitude profiles and exact analytical benchmarks.
- **Generic Data Ingestion:** Drag-and-drop CSV/Excel database loader.
- **Automated AI Optimization Engine:** Dynamic parameter mapping and automated calculation of residuals and training errors against experimental validation criteria (<5% target error).
- **Multi-Domain Support:** Pre-configured environments for Aerodynamic airfoils and Hemodynamic blood flow analysis.

## 🛠️ Tech Stack
- **Frontend & UI:** Streamlit (Python)
- **Mathematical Processing:** NumPy, SciPy (Optimization & Curve Fitting)
- **Data & Visualizations:** Pandas, Matplotlib
- **Planned Backend Integration:** GPU-accelerated WENO5 / SSP-RK3 solver via CUDA/CuPy.

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

