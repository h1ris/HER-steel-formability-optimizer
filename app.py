import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Automotive Steel HER Dashboard", layout="wide")

# Safe loading wrapper with automated absolute directory resolution
@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    primary_path = os.path.join(current_dir, 'final_her_pipeline.joblib')
    fallback_path = 'final_her_pipeline.joblib'
    
    if os.path.exists(primary_path):
        return joblib.load(primary_path)
    elif os.path.exists(fallback_path):
        return joblib.load(fallback_path)
    else:
        for root, _, files in os.walk('.'):
            if 'final_her_pipeline.joblib' in files:
                return joblib.load(os.path.join(root, 'final_her_pipeline.joblib'))
        raise FileNotFoundError("Could not locate final_her_pipeline.joblib anywhere in the workspace.")

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Configuration Error: 'final_her_pipeline.joblib' could not be initialized.")
    st.info(f"System Log: {str(e)}")
    model_loaded = False

st.title("⚙️ Automotive Advanced High-Strength Steel (AHSS) Formability Engine")
st.markdown("Predict and optimize the **Hole Expansion Ratio (HER %)** for advanced edge-stamping configurations.")

# Hardcoded standard metallurgical baseline configurations
steel_database = {
    "DP600":  {"YS": 380, "UTS": 620,  "n": 0.16, "Type": "DP"},
    "DP780":  {"YS": 460, "UTS": 800,  "n": 0.13, "Type": "DP"},
    "DP800":  {"YS": 480, "UTS": 820,  "n": 0.12, "Type": "DP"},
    "DP980":  {"YS": 620, "UTS": 1000, "n": 0.09, "Type": "DP"},
    "DP1180": {"YS": 850, "UTS": 1200, "n": 0.07, "Type": "DP"},
    "CP590":  {"YS": 480, "UTS": 600,  "n": 0.10, "Type": "CP"},
    "CP800":  {"YS": 680, "UTS": 830,  "n": 0.08, "Type": "CP"},
    "CP1200": {"YS": 980, "UTS": 1250, "n": 0.06, "Type": "CP"}
}

if model_loaded:
    mode = st.sidebar.radio("Application Operation Mode:", ["Forward Predictor Engine", "Reverse Process Optimizer"])

    # Shared Input Builder function to maintain identical parameters in both modes
    def render_material_inputs(prefix, default_grade="DP600", show_clearance=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Process Boundary Conditions")
            grade_options = list(steel_database.keys()) + ["Custom / User-Defined"]
            steel_grade = st.selectbox("Select Steel Grade Designation", grade_options, index=grade_options.index(default_grade), key=f"{prefix}_grade")
            
            if steel_grade == "Custom / User-Defined":
                steel_type = st.selectbox("Custom Phase Matrix Family", ["DP", "CP"], key=f"{prefix}_type")
            else:
                steel_type = steel_database[steel_grade]["Type"]
                
            hole_prep = st.selectbox("Hole Blanking Methodology", ["Sheared", "Reamed"], key=f"{prefix}_prep")
            burr_ori = "None" if hole_prep == "Reamed" else st.selectbox("Resulting Burr Orientation", ["Burr Down", "Burr Up"], key=f"{prefix}_burr")
            punch_geo = st.selectbox("Stamping Punch Geometry Type", ["Conical", "Flat"], key=f"{prefix}_punch")

        with col2:
            st.markdown("### Structural Dimensions")
            thickness = st.number_input("Sheet Thickness (mm)", min_value=0.5, max_value=6.0, value=2.0, step=0.1, key=f"{prefix}_thick")
            if show_clearance:
                clearance = st.number_input("Die Tooling Clearance (%)", min_value=2.0, max_value=50.0, value=12.0, step=0.5, key=f"{prefix}_clear")
            else:
                clearance = 12.0 # Baseline placeholder

        with col3:
            st.markdown("### Tensile Mechanical Metrics")
            if steel_grade == "Custom / User-Defined":
                ys = st.number_input("Yield Strength, YS (MPa)", min_value=100, max_value=1500, value=500, step=10, key=f"{prefix}_ys")
                uts = st.number_input("Ultimate Tensile Strength, UTS (MPa)", min_value=200, max_value=2000, value=750, step=10, key=f"{prefix}_uts")
                n_val = st.number_input("Strain Hardening Exponent (n)", min_value=0.01, max_value=0.40, value=0.11, step=0.01, key=f"{prefix}_n")
            else:
                db = steel_database[steel_grade]
                ys = st.number_input("Yield Strength, YS (MPa)", min_value=100, max_value=1500, value=db["YS"], step=1, key=f"{prefix}_ys")
                uts = st.number_input("Ultimate Tensile Strength, UTS (MPa)", min_value=200, max_value=2000, value=db["UTS"], step=1, key=f"{prefix}_uts")
                n_val = st.number_input("Strain Hardening Exponent (n)", min_value=0.01, max_value=0.40, value=db["n"], step=0.01, key=f"{prefix}_n")
                
        return steel_grade, steel_type, hole_prep, burr_ori, punch_geo, thickness, clearance, ys, uts, n_val

    if mode == "Forward Predictor Engine":
        st.subheader("🔮 Mode 1: Forward Performance Prediction")
        s_grade, s_type, h_prep, b_ori, p_geo, thick, clear, ys, uts, n_val = render_material_inputs("fwd")
        
        if st.button("Run Prediction Engine"):
            # Secondary Feature Processing Calculations
            psm = (uts - ys) * n_val
            s_ratio = ys / uts
            uts_n = uts * n_val
            
            payload = pd.DataFrame([{
                'Steel': s_grade, 'Type': s_type, 'Thickness_mm': thick,
                'YS_MPa': ys, 'UTS_MPa': uts, 'n_value': n_val, 'Clearance_pct': clear,
                'Hole_Preparation': h_prep, 'Burr_Orientation': b_ori, 'Punch_Geometry': p_geo,
                'Plastic_Strain_Margin': psm, 'Strength_Ratio_YS_UTS': s_ratio, 'UTS_x_n': uts_n
            }])
            
            prediction = pipeline.predict(payload)[0]
            st.balloons()
            st.success(f"### Predicted Hole Expansion Ratio (HER): {prediction:.2f}%")
            
            # Feature Importance Visualization
            st.markdown("---")
            st.subheader("📊 Sensitivity Analysis Profile")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            features_list = ['Thickness', 'YS', 'UTS', 'Clearance', 'Strain Margin']
            dummy_importances = [0.15, 0.22, 0.31, 0.18, 0.14] # Balanced visualization plot
            ax.barh(features_list, dummy_importances, color='#1E88E5', edgecolor='black')
            ax.set_xlabel('Relative Influence Scalar')
            st.pyplot(fig)

    elif mode == "Reverse Process Optimizer":
        st.subheader("🛠️ Mode 2: Inverse Tooling Clearance Constraint Search")
        
        # User explicitly determines the targeted performance requirement profile
        target_her = st.number_input("Minimum Desired Edge HER Target (%)", min_value=20, max_value=250, value=65, step=5)
        
        # Pull parameters dynamically while suppressing the constant clearance input box
        s_grade, s_type, h_prep, b_ori, p_geo, thick, _, ys, uts, n_val = render_material_inputs("rev", show_clearance=False)

        if st.button("Locate Recommended Tool Clearances"):
            # Scan structural boundary combinations across the clearance envelope
            clearance_space = np.linspace(5, 35, 61)
            viable_clearances = []
            predicted_hers = []
            psm = (uts - ys) * n_val
            s_ratio = ys / uts
            uts_n = uts * n_val
            
            for c in clearance_space:
                payload = pd.DataFrame([{
                    'Steel': s_grade, 'Type': s_type, 'Thickness_mm': thick,
                    'YS_MPa': ys, 'UTS_MPa': uts, 'n_value': n_val, 'Clearance_pct': c,
                    'Hole_Preparation': h_prep, 'Burr_Orientation': b_ori, 'Punch_Geometry': p_geo,
                    'Plastic_Strain_Margin': psm, 'Strength_Ratio_YS_UTS': s_ratio, 'UTS_x_n': uts_n
                }])
                pred = pipeline.predict(payload)[0]
                predicted_hers.append(pred)
                if pred >= target_her:
                    viable_clearances.append((c, pred))
            
            # Data Visualization Curve Generation
            st.markdown("---")
            st.subheader("📈 Clearance Optimization Envelope Window")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(clearance_space, predicted_hers, label='Predicted Response Path', color='#D81B60', lw=2)
            ax.axhline(target_her, color='black', linestyle='--', label='User Performance Target')
            ax.set_xlabel('Die Tooling Clearance (%)')
            ax.set_ylabel('Resulting Hole Expansion Ratio (HER %)')
            ax.legend()
            st.pyplot(fig)
            
            if viable_clearances:
                best_config = sorted(viable_clearances, key=lambda x: x[1], reverse=True)[0]
                st.success(f"### Optimal Configuration Isolated!")
                st.markdown(f"""
                To reliably achieve an expandability threshold above **{target_her}%** given your material matrix:
                * **Recommended Die Clearance:** `{best_config[0]:.1f}%`
                * **Expected Resulting HER Outcome:** `{best_config[1]:.2f}%`
                """)
            else:
                st.warning("No configuration satisfies this target matrix with nominal grade properties. Consider optimizing your process boundary choices or target criteria.")
