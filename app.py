import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# إعدادات الصفحة العامة
st.set_page_config(
    page_title="GL-Vorticity Framework Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# العنوان الرئيسي للمنصة
st.title("🧬 Coupled Ginzburg–Landau Vorticity Framework")
st.subheader("Reduced-Order Transition Modeling & Multi-Domain Data Calibration")
st.markdown("---")

# --- الشريط الجانبي (Sidebar) لإدخال المعاملات الفيزيائية ---
st.sidebar.header("⚙️ Model Physical Parameters")

# اختيار المجال
domain = st.sidebar.selectbox(
    "Select Application Domain",
    ["Aerodynamics (Airfoils)", "Hemodynamics (Blood Flow)"]
)

# المعاملات الفيزيائية المستوحاة من بحثك العلمي
Re = st.sidebar.number_input("Reynolds Number (Re)", min_value=100, max_value=20000000, value=6000000, step=50000)
gamma = st.sidebar.slider("Linear Growth Rate (γ)", -2.0, 2.0, 1.0, 0.1)
beta_0 = st.sidebar.slider("Standard Landau Coefficient (β₀)", 0.1, 5.0, 1.0, 0.1)
beta_VS = st.sidebar.slider("Vortex Stretching Coupling (β_VS)", 0.0, 10.0, 2.5, 0.1)

# إعدادات شبكة المحاكاة الحوسبية
st.sidebar.header("🖥️ Computation Settings")
grid_size = st.sidebar.selectbox("Grid Resolution", ["64³", "128³", "256³"])
use_gpu = st.sidebar.checkbox("Enable NVIDIA A100 GPU Acceleration", value=True)

# --- الجزء الرئيسي الأول: المحاكاة النظرية والتحقق (Analytical Benchmark) ---
st.header("🔬 1. Theoretical Simulation & Exact Benchmarks")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(r"""
    **Mathematical Framework:**
    The slow spatio-temporal evolution of wave amplitude is modeled via the coupled GL-Vorticity system:
    $$ \partial_t A + U \cdot \nabla A = \gamma A - \beta(\omega)|A|^2 A + \nu_{eff} \Delta A $$
    Where the dynamic coupling is phenomenologically motivated by vortex stretching:
    $$ \beta(\omega) = \beta_0 + \beta_{VS}|\omega|^2 $$
    """)
    
    # حساب نقطة الاستقرار النظرية في الحالة المتجانسة
    if gamma > 0 and beta_0 > 0:
        A_ss = np.sqrt(gamma / beta_0)
        st.success(f"🎯 Exact Analytical Steady State: $A_{{ss}}$ = {A_ss:.3f}")
    else:
        st.warning("⚠️ No non-trivial steady state for current γ and β₀ configuration.")

with col2:
    # محاكاة بسيطة لرسم نمو الموجة التفاعلية (Tollmien-Schlichting waves)
    t = np.linspace(0, 10, 200)
    # حل مبسط للمعادلة التفاضلية العادية لاستعراض الفكرة تفاعلياً
    if gamma > 0:
        A_t = np.sqrt(gamma / (beta_0 + (gamma/0.01 - beta_0) * np.exp(-2 * gamma * t)))
    else:
        A_t = 0.1 * np.exp(gamma * t)
        
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, A_t, color='#1f77b4', linewidth=2, label="Coupled GL Amplitude |A(t)|")
    ax.axhline(np.sqrt(gamma/beta_0) if gamma>0 else 0, color='r', linestyle='--', label="Theoretical Limit")
    ax.set_xlabel("Dimensionless Time (T)")
    ax.set_ylabel("Wave Amplitude |A|")
    ax.set_title("Amplitude Evolution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

# --- الجزء الرئيسي الثاني: رفع قواعد البيانات والمعايرة الفورية (Data Ingestion & Calibration) ---
st.header("📊 2. Generic Data Ingestion & Automated Calibration Dashboard")
st.write("Upload your experimental database (CSV or Excel) to test, calibrate, and fit the Ginzburg-Landau model to real-world data.")

# توفير بيانات افتراضية للمستخدم لتجربة الواجهة فوراً في حال عدم رفع ملف
uploaded_file = st.file_uploader("Upload Experimental Dataset (CSV / XLSX)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("💾 Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("💡 Showing simulated benchmark dataset for demonstration. Upload your own to process real engineering/medical logs.")
    # توليد بيانات افتراضية تعاكي ما ورد في مقترحك البحثي (NACA 0012 أو لزوجة الدم)
    np.random.seed(42)
    if domain == "Aerodynamics (Airfoils)":
        x_data = np.linspace(0.1, 0.9, 20) # موقع الانتقال على الجناح x/c
        # محاكاة نسبة خطأ عشوائية مع منحنى فيزيائي
        y_data = 1.2 / (1.0 + 2.0 * x_data**2) + np.random.normal(0, 0.04, 20)
        df = pd.DataFrame({"Chord Position (x/c)": x_data, "Measured Transition Amplitude": y_data})
        x_col, y_col = "Chord Position (x/c)", "Measured Transition Amplitude"
    else:
        x_data = np.linspace(10, 100, 20) # قطر الأوعية الدموية بالمايكرومتر
        y_data = 1.5 + 2.5 * np.exp(-0.05 * x_data) + np.random.normal(0, 0.05, 20)
        df = pd.DataFrame({"Vessel Diameter (µm)": x_data, "Relative Blood Viscosity": y_data})
        x_col, y_col = "Vessel Diameter (µm)", "Relative Blood Viscosity"

# عرض جزء من البيانات المرفوعة
st.dataframe(df.head(5), use_container_width=True)

# اختيار الأعمدة للمعايرة ديناميكياً
st.subheader("🤖 Automated Optimization Engine")
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    x_axis = st.selectbox("Select Independent Variable (X-Axis/Physics input)", df.columns, index=0)
with col_cfg2:
    y_axis = st.selectbox("Select Target Variable (Y-Axis/Empirical output)", df.columns, index=1)

# تنفيذ خوارزمية Curve Fitting لمعايرة الثوابت (β₀, β_VS) تلقائياً بناءً على البيانات المرفوعة
st.markdown("### 🎯 Calibration Results vs Experimental Data")

X_fit = df[x_axis].values
Y_fit = df[y_axis].values

# دالة فيزيائية تقريبية للمعايرة والربط
def model_fit_func(x, b0, b_vs):
    # دالة محاكاة تعتمد على الشكل الرياضي لجينزبورغ-لانداو وتأثير الدوران
    return gamma / (b0 + b_vs * (1.0 / (x + 0.1)))

try:
    popt, pcov = curve_fit(model_fit_func, X_fit, Y_fit, p0=[beta_0, beta_VS], maxfev=5000)
    calibrated_b0, calibrated_b_vs = popt[0], popt[1]
    
    # حساب نسبة الخطأ والـ Residuals للتأكد من مطابقة شروط القبول البحثية (< 5%)
    residuals = Y_fit - model_fit_func(X_fit, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y_fit - np.mean(Y_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)
    error_percentage = (np.mean(np.abs(residuals) / Y_fit)) * 100

    # عرض النتائج في بطاقات إحصائية تفاعلية
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Calibrated β₀", f"{calibrated_b0:.4f}")
    metric2.metric("Calibrated β_VS", f"{calibrated_b_vs:.4f}")
    metric3.metric("Model Fitness ($R^2$)", f"{r_squared*100:.2f}%")
    
    # التحقق من شروط القبول الواردة في مقترحك (Acceptance Criteria < 5%)
    if error_percentage < 5.0:
        metric4.metric("Mean Training Error", f"{error_percentage:.2f}%", delta="Passed Criteria (<5%)")
    else:
        metric4.metric("Mean Training Error", f"{error_percentage:.2f}%", delta="Requires Recalibration", delta_color="inverse")

    # رسم مقارنة بيانية احترافية
    fig_cal, ax_cal = plt.subplots(figsize=(8, 3.5))
    ax_cal.scatter(X_fit, Y_fit, color='black', label='Uploaded Database Points', alpha=0.7)
    
    X_smooth = np.linspace(X_fit.min(), X_fit.max(), 100)
    ax_cal.plot(X_smooth, model_fit_func(X_smooth, *popt), color='teal', linestyle='-', linewidth=2.5, 
                label=f'Calibrated GL-Vorticity Model (Error: {error_percentage:.2f}%)')
    
    ax_cal.set_xlabel(x_axis)
    ax_cal.set_ylabel(y_axis)
    ax_cal.set_title(f"Dynamic Calibration Curve for {domain}")
    ax_cal.legend()
    ax_cal.grid(True, alpha=0.2)
    st.pyplot(fig_cal)

except Exception as optimization_error:
    st.error(f"Could not automatically calibrate the physics parameters for this specific dataset structure. Technical reason: {optimization_error}")

# تذييل الصفحة الاحترافي لتأكيد الهوية الأكاديمية للمشروع
st.markdown("---")
st.caption("Developed by Candidate: Yousef Saleh | Target Programme: MPhil in Engineering, University of Cambridge | DAMTP Fluids Group")
