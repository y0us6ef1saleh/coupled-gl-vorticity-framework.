import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model import GLVorticityModel, calibrate_model, NACA_X, NACA_AMP
import anthropic

st.set_page_config(page_title="GL-Vorticity Framework", layout="wide")

st.title("🧬 Coupled Ginzburg-Landau Vorticity Framework")
st.markdown("### Reduced-Order Transition Modeling & Data Calibration")

# Sidebar
st.sidebar.header("⚙️ Model Physical Parameters")
gamma = st.sidebar.slider("Linear Growth Rate (γ)", 0.1, 2.0, 1.0)
beta_0 = st.sidebar.slider("Standard Landau Coefficient (β₀)", 0.1, 3.0, 1.0)
beta_VS = st.sidebar.slider("Vortex Stretching Coupling (β_VS)", 0.0, 5.0, 2.5)

tab1, tab2 = st.tabs(["1. Theoretical Simulation", "2. Automated Calibration (NACA 0012)"])

# ==========================================
# TAB 1: Theoretical Simulation
# ==========================================
with tab1:
    st.header("Theoretical Amplitude Evolution")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Mathematical Framework:**")
        st.latex(r"\partial_t A = \gamma A - \beta(\omega)|A|^2 A + \nu_{eff} \Delta A")
        st.latex(r"\beta(\omega) = \beta_0 + \beta_{VS} |\omega|^2")
        A_ss = np.sqrt(gamma / beta_0) if beta_0 > 0 else 0.0
        st.success(f"Exact Analytical Steady State: $A_{{ss}} = {A_ss:.3f}$")

    with col2:
        model = GLVorticityModel(gamma=gamma, beta_0=beta_0, beta_VS=beta_VS)
        t, A, omega = model.solve()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=A, mode='lines', name='Coupled GL Amplitude |A(t)|', line=dict(color='royalblue', width=3)))
        fig.update_layout(title="Amplitude Evolution Over Time", xaxis_title="Dimensionless Time (T)", yaxis_title="Wave Amplitude |A|")
        st.plotly_chart(fig, use_container_width=True)

 # ==========================================
 # TAB 2: Calibration, AI & Optimization
 # ==========================================
  # ==========================================
# TAB 2: Calibration & AI Analysis
# ==========================================
with tab2:
    st.header("Data Ingestion & Automated Calibration")
    st.markdown("Upload your experimental database (CSV or Excel) to test, calibrate, and fit the Ginzburg-Landau model to real-world data.")
    
    # 1. رفع الملف
    uploaded_file = st.file_uploader("Upload Experimental Dataset (CSV / XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        
        # مسح النتائج القديمة إذا تم رفع ملف جديد
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.calibrated = False
            st.session_state.current_file = uploaded_file.name
    else:
        st.info("💡 Showing simulated benchmark dataset (NACA 0012) for demonstration.")
        df = pd.DataFrame({"Chord Position (x/c)": NACA_X, "Measured Transition Amplitude": NACA_AMP})
        st.dataframe(df.head())
        x_data = NACA_X
        y_data = NACA_AMP
        if 'current_file' in st.session_state and st.session_state.current_file is not None:
            st.session_state.calibrated = False
            st.session_state.current_file = None

    st.markdown("---")
    
    # 2. زر المعايرة
    if st.button("Run Automated Optimization"):
        with st.spinner("Calibrating model against data..."):
            b0_cal, bvs_cal, r2 = calibrate_model(x_data, y_data)
            
            st.session_state.b0_cal = b0_cal
            st.session_state.bvs_cal = bvs_cal
            st.session_state.r2 = r2
            st.session_state.x_data = x_data
            st.session_state.y_data = y_data
            st.session_state.calibrated = True

    # 3. عرض النتائج
    if st.session_state.get('calibrated', False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Calibrated β₀", f"{st.session_state.b0_cal:.4f}")
        c2.metric("Calibrated β_VS", f"{st.session_state.bvs_cal:.4f}")
        c3.metric("Model Fitness (R²)", f"{st.session_state.r2*100:.2f}%")
        
        x_fit = np.linspace(min(st.session_state.x_data), max(st.session_state.x_data), 100)
        y_fit = np.exp(-st.session_state.b0_cal * x_fit) - st.session_state.bvs_cal * (x_fit**2) + 0.2
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=st.session_state.x_data, y=st.session_state.y_data, mode='markers', name='Experimental Data', marker=dict(size=10, color='black')))
        fig2.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Calibrated Model', line=dict(color='teal', width=3)))
        fig2.update_layout(title="Dynamic Calibration Curve", xaxis_title="Chord Position (x/c)", yaxis_title="Amplitude")
        st.plotly_chart(fig2, use_container_width=True)
        
        # 4. قسم الذكاء الاصطناعي
        st.markdown("---")
        st.subheader("🤖 AI-Powered Analysis")
        
        if st.button("Explain Results with AI"):
            with st.spinner("Claude is analyzing the physics..."):
                prompt = f"You are a fluid dynamics expert. A Ginzburg-Landau model was calibrated against data. Results: R^2: {st.session_state.r2*100:.2f}%, beta_0: {st.session_state.b0_cal:.4f}, beta_VS: {st.session_state.bvs_cal:.4f}. Write a brief, 3-sentence academic analysis explaining these results."
                try:
                    api_key = st.secrets.get("ANTHROPIC_API_KEY")
                    if api_key:
                        client = anthropic.Anthropic(api_key=api_key)
                        response = client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=200,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        ai_response = response.content[0].text
                    else:
                        raise ValueError("No API key")
                except Exception as e:
                    ai_response = f"**Simulated Expert Analysis:** The model achieved an R-squared of **{st.session_state.r2*100:.2f}%**. The calibrated vortex stretching coefficient (β_VS) is **{st.session_state.bvs_cal:.4f}**."
                
                st.info(ai_response)                            
