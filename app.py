import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Automotive Steel HER Dashboard", layout="wide")

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
    model_loaded = False

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
    mode = st.sidebar.radio("Application Mode:", [
        "🔮 Forward Predictor Engine", 
        "🛠️ Automatic Process Optimizer", 
        "📈 Advanced Material Analytics"
    ])

    def render_material_inputs(prefix, default_grade="DP600", show_clearance=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Boundary Conditions")
            grade_options = list(steel_database.keys()) + ["Custom / User-Defined"]
            steel_grade = st.selectbox("Steel Grade Designation", grade_options, index=grade_options.index(default_grade), key=f"{prefix}_grade")
            steel_type = "CP" if "CP" in steel_grade else "DP" if "DP" in steel_grade else st.selectbox("Custom Matrix", ["DP", "CP"], key=f"{prefix}_type")
            hole_prep = st.selectbox("Hole Blanking Methodology", ["Sheared", "Reamed"], key=f"{prefix}_prep")
            burr_ori = "None" if hole_prep == "Reamed" else st.selectbox("Resulting Burr Orientation", ["Burr Down", "Burr Up"], key=f"{prefix}_burr")
            punch_geo = st.selectbox("Stamping Punch Geometry Type", ["Conical", "Flat"], key=f"{prefix}_punch")

        with col2:
            st.markdown("### Structural Dimensions")
            thickness = st.number_input("Sheet Thickness (mm)", min_value=0.5, max_value=6.0, value=2.0, step=0.1, key=f"{prefix}_thick")
            clearance = st.number_input("Die Tooling Clearance (%)", min_value=2.0, max_value=50.0, value=12.0, step=0.5, key=f"{prefix}_clear") if show_clearance else 12.0

        with col3:
            st.markdown("### Mechanical Metrics")
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

    if mode == "🔮 Forward Predictor Engine":
        st.subheader("🔮 Mode 1: Forward Performance Prediction")
        s_grade, s_type, h_prep, b_ori, p_geo, thick, clear, ys, uts, n_val = render_material_inputs("fwd")
        
        if st.button("Run Prediction Engine"):
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

    elif mode == "🛠️ Automatic Process Optimizer":
        st.subheader("🛠️ Mode 2: Automated Die Clearance Optimization")
        s_grade, s_type, h_prep, b_ori, p_geo, thick, _, ys, uts, n_val = render_material_inputs("opt", show_clearance=False)

        if st.button("Isolate Optimum Manufacturing Window"):
            clearance_space = np.linspace(5, 40, 71)
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
                predicted_hers.append(pipeline.predict(payload)[0])
            
            max_her = max(predicted_hers)
            opt_clearance = clearance_space[np.argmax(predicted_hers)]
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="Maximum Attainable HER", value=f"{max_her:.2f}%")
            with col_res2:
                st.metric(label="Optimum Die Clearance Value", value=f"{opt_clearance:.1f}%")
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(clearance_space, predicted_hers, color='#2E7D32', lw=2.5, label='Clearance Response Curve')
            ax.axvline(opt_clearance, color='#C62828', linestyle='--', label=f'Optimal Window ({opt_clearance:.1f}%)')
            ax.set_xlabel('Die Tooling Clearance (%)')
            ax.set_ylabel('Hole Expansion Ratio (HER %)')
            ax.legend(prop={'size': 8})
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)

    elif mode == "📈 Advanced Material Analytics":
        st.subheader("📊 Mode 3: Interactive Multi-Parameter Sensitivity Analyzer")
        
        # Pull baseline properties
        s_grade, s_type, h_prep, b_ori, p_geo, thick, clear, ys, uts, n_val = render_material_inputs("anl")
        
        st.markdown("---")
        st.markdown("### Choose Parametric Variations to Chart Against HER%")
        
        param_map = {
            "Die Tooling Clearance (%)": np.linspace(5, 40, 30),
            "Sheet Thickness (mm)": np.linspace(1.0, 4.0, 30),
            "Ultimate Tensile Strength (MPa)": np.linspace(500, 1300, 30),
            "Strain Hardening Exponent (n)": np.linspace(0.05, 0.22, 30)
        }
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            var_x = st.selectbox("Select Parameter for X-Axis Variable", list(param_map.keys()), index=0)
        with col_sel2:
            var_y = st.selectbox("Select Parameter for Y-Axis Variable", list(param_map.keys()), index=1)
            
        if st.button("Generate Parametric Field Map"):
            x_space = param_map[var_x]
            y_space = param_map[var_y]
            X, Y = np.meshgrid(x_space, y_space)
            Z = np.zeros_dict = np.zeros(X.shape)
            
            # Fast tracking grid evaluation matrix loop
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # Reassign dynamically based on selection
                    c_loop = X[i, j] if var_x == "Die Tooling Clearance (%)" else Y[i, j] if var_y == "Die Tooling Clearance (%)" else clear
                    t_loop = X[i, j] if var_x == "Sheet Thickness (mm)" else Y[i, j] if var_y == "Sheet Thickness (mm)" else thick
                    uts_loop = X[i, j] if var_x == "Ultimate Tensile Strength (MPa)" else Y[i, j] if var_y == "Ultimate Tensile Strength (MPa)" else uts
                    n_loop = X[i, j] if var_x == "Strain Hardening Exponent (n)" else Y[i, j] if var_y == "Strain Hardening Exponent (n)" else n_val
                    
                    psm_loop = (uts_loop - ys) * n_loop
                    
                    payload = pd.DataFrame([{
                        'Steel': s_grade, 'Type': s_type, 'Thickness_mm': t_loop,
                        'YS_MPa': ys, 'UTS_MPa': uts_loop, 'n_value': n_loop, 'Clearance_pct': c_loop,
                        'Hole_Preparation': h_prep, 'Burr_Orientation': b_ori, 'Punch_Geometry': p_geo,
                        'Plastic_Strain_Margin': psm_loop, 'Strength_Ratio_YS_UTS': ys/uts_loop, 'UTS_x_n': uts_loop*n_loop
                    }])
                    Z[i, j] = pipeline.predict(payload)[0]
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            contour = ax.contourf(X, Y, Z, levels=15, cmap='plasma')
            fig.colorbar(contour, label='Resulting HER %')
            ax.set_xlabel(var_x)
            ax.set_ylabel(var_y)
            st.pyplot(fig)
