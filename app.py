import streamlit as st
import numpy as np
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

with tab1:
    st.header("Theoretical Amplitude Evolution")
    model = GLVorticityModel(gamma=gamma, beta_0=beta_0, beta_VS=beta_VS)
    t, A, omega = model.solve()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=A, mode='lines', name='Coupled GL Amplitude |A(t)|', line=dict(color='royalblue', width=3)))
    fig.update_layout(title="Amplitude Evolution Over Time", xaxis_title="Dimensionless Time (T)", yaxis_title="Wave Amplitude |A|")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Ingestion & Automated Calibration")
    
    if st.button("Run Automated Optimization"):
        with st.spinner("Calibrating model against NACA 0012 data..."):
            b0_cal, bvs_cal, r2 = calibrate_model()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Calibrated β₀", f"{b0_cal:.4f}")
            c2.metric("Calibrated β_VS", f"{bvs_cal:.4f}")
            c3.metric("Model Fitness (R²)", f"{r2*100:.2f}%")
            
            x_fit = np.linspace(min(NACA_X), max(NACA_X), 100)
            y_fit = np.exp(-b0_cal * x_fit) + bvs_cal * (x_fit**2) + 0.2
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=NACA_X, y=NACA_AMP, mode='markers', name='NACA 0012 Data', marker=dict(size=10, color='black')))
            fig2.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Calibrated Model', line=dict(color='teal', width=3)))
            fig2.update_layout(title="Calibration Curve", xaxis_title="Chord Position (x/c)", yaxis_title="Amplitude")
            st.plotly_chart(fig2, use_container_width=True)
            
            # ====================================================
            # AI ANALYSIS (Real Claude API Integration with Fallback)
            # ====================================================
            st.markdown("---")
            st.subheader("🤖 AI-Powered Analysis")
            st.write("Let Claude analyze the physics behind the calibrated results.")
            
            if st.button("Explain Results with AI"):
                with st.spinner("Claude is analyzing the physics..."):
                    
                    prompt = f"""
                    You are a fluid dynamics expert. A Ginzburg-Landau model was calibrated against NACA 0012 data.
                    Results:
                    - Model Fitness (R^2): {r2*100:.2f}%
                    - Calibrated beta_0: {b0_cal:.4f}
                    - Calibrated beta_VS (vortex stretching coupling): {bvs_cal:.4f}
                    
                    Write a brief, 3-sentence academic analysis explaining these results.
                    """
                    
                    try:
                        # محاولة استخدام مفتاح API إذا كان موجوداً في Secrets
                        api_key = st.secrets.get("ANTHROPIC_API_KEY")
                        if api_key:
                            client = anthropic.Anthropic(api_key=api_key)
                            response = client.messages.create(
                                model="claude-3-haiku-20240307", # أسرع وأرخص موديل
                                max_tokens=200,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            ai_response = response.content[0].text
                        else:
                            raise ValueError("No API key found")
                            
                    except Exception as e:
                        # الخطة البديلة (Fallback) لكي يعمل التطبيق للزوار بدون مفتاح API
                        ai_response = f"""
                        **Simulated Expert Analysis (No API Key Configured):**
                        The model achieved an R-squared of **{r2*100:.2f}%**, indicating a reasonable fit to the experimental data. 
                        The calibrated vortex stretching coefficient (β_VS) is **{bvs_cal:.4f}**. 
                        This positive value implies that vortex stretching acts as a non-linear damping mechanism, consistent with the Ginzburg-Landau theoretical framework for boundary-layer transition.
                        """
                    
                    st.info(ai_response)
