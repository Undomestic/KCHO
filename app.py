"""
Streamlit Frontend for KCTHO.
A professional, modernized UI for the Kolmogorov-Chebyshev Topological Health Oracle.

Run with:
    streamlit run backend/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Backend Imports
import config
from data_generation import generate_data_for_condition, smooth_all_variables, get_default_params
from topology import (
    compute_persistence, betti_numbers_at_scale, persistent_entropy
)
from kan import fit_kan_edge_bspline, most_complex_variable
from math_utils import (
    lyapunov_spectrum, correlation_dimension, approximate_entropy,
    ricci_curvature, mean_curvature_trend, box_counting_dimension,
    compute_health_score
)
from clinical import (
    interpret_lyapunov, generate_diagnosis,
    enhanced_report, generate_easy_explanations
)
from treatment_simulator import run_simulation
from example_loader import get_examples, find_closest_example
from persim import plot_diagrams

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KCTHO | Topological Health Oracle",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    /* Headers */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00d2ff;
    }
    /* Buttons */
    .stButton>button {
        background-color: #00d2ff;
        color: #000000;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00b8e6;
        color: #ffffff;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d2ff;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🧬 KCTHO Control")
        st.markdown("Topological Health Oracle")
        st.markdown("---")
        
        condition = st.selectbox(
            "Medical Condition",
            ["sepsis", "cardiac arrest", "hemorrhage", "healthy"],
            index=0
        )
        
        st.markdown("### Configuration")
        use_gemini = st.checkbox("Enable Gemini AI", value=config.USE_GEMINI)
        seed_input = st.number_input("Random Seed (0 = Random)", min_value=0, value=0, step=1)
        seed = None if seed_input == 0 else seed_input
        
        st.markdown("---")
        run_btn = st.button("RUN ANALYSIS", type="primary")
        
        st.markdown("### System Status")
        st.caption("✅ Backend Connected")
        if use_gemini:
            st.caption("✅ Gemini AI Active")
        else:
            st.caption("⚪ Gemini AI Disabled")

    # --- MAIN CONTENT ---
    st.title("Kolmogorov-Chebyshev Topological Health Oracle")
    st.markdown("##### Advanced Topological & Dynamical Analysis of Physiological Signals")

    # Initialize session state for data persistence
    if 'data_generated' not in st.session_state:
        st.session_state['data_generated'] = False

    if run_btn:
        with st.spinner(f"Generating synthetic physiological data for '{condition}'..."):
            # Phase 1: Data Generation
            raw_data = generate_data_for_condition(condition, seed=seed, vary=True)
            smoothed_data = smooth_all_variables(raw_data)
            
            st.session_state['raw_data'] = raw_data
            st.session_state['smoothed_data'] = smoothed_data
            st.session_state['condition'] = condition
            st.session_state['data_generated'] = True
            
            # Run Analysis immediately to populate cache
            point_cloud = np.column_stack([smoothed_data[var] for var in config.PHYSIO_VARS])
            diagrams = compute_persistence(point_cloud)
            betti = betti_numbers_at_scale(diagrams, scale=config.BETTI_SCALE_THRESHOLD)
            
            if len(diagrams) > 1:
                entropy_h1 = persistent_entropy(diagrams[1])
            else:
                entropy_h1 = 0.0
                
            lyap_spec = lyapunov_spectrum(smoothed_data['MAP'],
                                        emb_dim=config.LYAPUNOV_EMBEDDING_DIM,
                                        matrix_dim=config.LYAPUNOV_MATRIX_DIM)
            lyap_max = lyap_spec[0]
            
            corr_dim = correlation_dimension(point_cloud)
            apen = approximate_entropy(smoothed_data['MAP'])
            box_dim = box_counting_dimension(point_cloud)
            
            curvatures = ricci_curvature(point_cloud)
            mean_curv = np.mean(curvatures)
            
            var_pairs = [(v, 'MAP') for v in config.PHYSIO_VARS if v != 'MAP']
            most_var, max_comp, all_comps = most_complex_variable(smoothed_data, var_pairs, method='curvature')
            
            health_score = compute_health_score(
                betti, lyap_max, entropy_h1, corr_dim, apen, box_dim, mean_curv, max_comp
            )
            
            st.session_state['results'] = {
                'betti': betti,
                'entropy_h1': entropy_h1,
                'lyap_max': lyap_max,
                'corr_dim': corr_dim,
                'apen': apen,
                'box_dim': box_dim,
                'mean_curv': mean_curv,
                'most_complex_var': most_var,
                'max_complex': max_comp,
                'all_comps': all_comps,
                'diagnosis': generate_diagnosis(betti, lyap_max, most_var, mean_curv),
                'lyap_interp': interpret_lyapunov(lyap_max),
                'health_score': health_score
            }
            st.session_state['point_cloud'] = point_cloud
            st.session_state['curvatures'] = curvatures
            st.session_state['diagrams'] = diagrams

    if st.session_state['data_generated']:
        # Retrieve data from session state
        raw_data = st.session_state['raw_data']
        smoothed_data = st.session_state['smoothed_data']
        results = st.session_state['results']
        curvatures = st.session_state['curvatures']
        point_cloud = st.session_state['point_cloud']
        
        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vitals & Data", 
            "🌀 Topology & Dynamics", 
            "🧠 KAN Complexity", 
            "📋 Clinical Report",
            "💊 Treatment Simulator"
        ])

        # --- TAB 1: VITALS ---
        with tab1:
            st.subheader("Physiological Time Series")
            st.caption("Comparison of Raw Noisy Data vs. Chebyshev Smoothed Manifold")
            
            fig = go.Figure()
            colors = ['#00d2ff', '#ff0055', '#00ffaa', '#ffaa00']
            
            for i, var in enumerate(config.PHYSIO_VARS):
                # Smoothed only for cleaner UI, or toggle
                fig.add_trace(go.Scatter(
                    x=smoothed_data['time'], y=smoothed_data[var],
                    mode='lines', name=var,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
                # Add raw as faint background
                fig.add_trace(go.Scatter(
                    x=raw_data['time'], y=raw_data[var],
                    mode='lines', name=f"{var} (Raw)",
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.3, showlegend=False
                ))

            fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Time (steps)",
                yaxis_title="Normalized Value",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: TOPOLOGY ---
        with tab2:
            # Top Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Health Score", f"{results['health_score']:.1f}/100", 
                      delta="Low" if results['health_score'] < 50 else "Stable")
            c2.metric("Betti Numbers", f"{results['betti']}", "β0, β1, β2")
            c3.metric("Max Lyapunov (λ)", f"{results['lyap_max']:.3f}", results['lyap_interp'])
            c4.metric("Mean Curvature", f"{results['mean_curv']:.3f}")

            st.markdown("---")
            
            col_left, col_right = st.columns([1.5, 1])
            
            with col_left:
                st.subheader("3D Phase Space Manifold")
                st.caption("Colored by Ricci Curvature")
                
                # Use first 3 vars for 3D projection
                vars_3d = config.PHYSIO_VARS[:3]
                df_3d = pd.DataFrame({
                    vars_3d[0]: smoothed_data[vars_3d[0]],
                    vars_3d[1]: smoothed_data[vars_3d[1]],
                    vars_3d[2]: smoothed_data[vars_3d[2]],
                    'Curvature': curvatures
                })
                
                fig_3d = px.scatter_3d(
                    df_3d, x=vars_3d[0], y=vars_3d[1], z=vars_3d[2],
                    color='Curvature',
                    color_continuous_scale='Viridis',
                    title=f"Phase Space: {vars_3d[0]} vs {vars_3d[1]} vs {vars_3d[2]}"
                )
                fig_3d.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig_3d, use_container_width=True)

            with col_right:
                st.subheader("Topological Features")
                
                # Curvature Trend
                fig_curv = px.line(y=curvatures, title="Ricci Curvature Evolution")
                fig_curv.update_traces(line_color='#00d2ff')
                fig_curv.update_layout(template="plotly_dark", height=300, xaxis_title="Time", yaxis_title="Ricci Scalar")
                st.plotly_chart(fig_curv, use_container_width=True)
                
                # Persistence Diagram (Matplotlib fallback)
                st.markdown("**Persistence Diagrams**")
                fig_pers = plt.figure(figsize=(5, 4), facecolor='#0e1117')
                # Create axis for persim to plot on
                ax = fig_pers.add_subplot(111)
                # We need to update utils.py to accept ax or just use plot_diagrams directly here
                # Using plot_diagrams directly since we imported it
                plot_diagrams(st.session_state['diagrams'], show=False, ax=ax)
                st.pyplot(fig_pers)

        # --- TAB 3: KAN ---
        with tab3:
            st.subheader("Kolmogorov-Arnold Network (KAN) Analysis")
            st.markdown("Identifies the variable contributing most to the system's complexity.")
            
            most_var = results['most_complex_var']
            
            col_k1, col_k2 = st.columns([2, 1])
            
            with col_k1:
                x = smoothed_data[most_var]
                y = smoothed_data['MAP']
                spline = fit_kan_edge_bspline(x, y)
                x_grid = np.linspace(min(x), max(x), config.KAN_N_GRID)
                y_spline = spline(x_grid)
                
                fig_kan = go.Figure()
                fig_kan.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Observed Data', marker=dict(color='gray', opacity=0.5)))
                fig_kan.add_trace(go.Scatter(x=x_grid, y=y_spline, mode='lines', name='Learned Activation φ(x)', line=dict(color='#ff0055', width=4)))
                
                fig_kan.update_layout(
                    title=f"Learned Edge Function: {most_var} → MAP",
                    xaxis_title=most_var,
                    yaxis_title="MAP",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig_kan, use_container_width=True)
            
            with col_k2:
                st.markdown("### Complexity Scores")
                comps = results['all_comps']
                df_comp = pd.DataFrame(list(comps.items()), columns=['Variable', 'Complexity'])
                df_comp = df_comp.sort_values('Complexity', ascending=False)
                
                st.dataframe(
                    df_comp.style.background_gradient(cmap='Reds'),
                    use_container_width=True
                )
                st.info(f"**{most_var}** is the dominant driver of complexity.")

        # --- TAB 4: REPORT ---
        with tab4:
            st.subheader("Clinical Report & AI Insights")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown("#### 📄 Technical Report")
                report_text = enhanced_report(results, st.session_state['condition'])
                st.text_area("Generated Report", report_text, height=400)
            
            with col_r2:
                st.markdown("#### 🤖 Gemini AI Analysis")
                if use_gemini:
                    if st.button("Generate AI Explanation"):
                        with st.spinner("Consulting Gemini..."):
                            # Prepare context
                            last_vitals = {
                                'map': smoothed_data['MAP'][-1],
                                'lactate': smoothed_data['Lactate'][-1],
                                'hrv': smoothed_data['HRV'][-1],
                                'scvo2': smoothed_data['ScvO2'][-1]
                            }
                            examples = get_examples()
                            closest_examples = find_closest_example(last_vitals, examples, n=3) if examples else None
                            
                            metrics_dict_gemini = {
                                **results,
                                'hrv_value': np.mean(smoothed_data['HRV']),
                                'map_value': np.mean(smoothed_data['MAP']),
                                'il6_value': np.mean(smoothed_data['IL6']),
                            }
                            
                            explanation = generate_easy_explanations(
                                results, st.session_state['condition'], metrics_dict_gemini,
                                closest_examples=closest_examples
                            )
                            st.markdown(explanation)
                else:
                    st.warning("Gemini API is disabled. Enable it in the sidebar to see AI insights.")

        # --- TAB 5: SIMULATOR ---
        with tab5:
            st.subheader("What-If Treatment Simulator")
            st.markdown("Simulate interventions and observe topological shifts.")
            
            col_sim1, col_sim2 = st.columns([1, 3])
            
            with col_sim1:
                treatment_opt = st.selectbox(
                    "Select Intervention", 
                    ["antibiotics", "oxygen", "fluids", "anti_inflammatories"]
                )
                sim_btn = st.button("Simulate Treatment", type="secondary")
            
            if sim_btn:
                with st.spinner(f"Simulating effects of {treatment_opt}..."):
                    base_params = get_default_params(st.session_state['condition'], vary=True)
                    treatment_results = run_simulation(st.session_state['condition'], base_params, treatment_opt, seed=seed)
                    
                    # Display Comparison
                    with col_sim2:
                        st.markdown(f"### Results: {treatment_opt.capitalize()}")
                        
                        m1, m2, m3 = st.columns(3)
                        
                        # Helper for delta display
                        def show_delta(label, val_before, val_after, fmt="{:.2f}"):
                            delta = val_after - val_before
                            st.metric(label, fmt.format(val_after), f"{delta:+.2f}")

                        with m1:
                            st.markdown("**Topology**")
                            show_delta("β0 (Components)", results['betti'][0], treatment_results['betti'][0], "{:.0f}")
                            show_delta("β1 (Loops)", results['betti'][1], treatment_results['betti'][1], "{:.0f}")
                        
                        with m2:
                            st.markdown("**Dynamics**")
                            show_delta("Lyapunov λ", results['lyap_max'], treatment_results['lyap_max'])
                            show_delta("Entropy H1", results['entropy_h1'], treatment_results['entropy_h1'])
                        
                        with m3:
                            st.markdown("**Outcome**")
                            st.metric("Diagnosis", treatment_results['diagnosis'])
                            st.metric("Mean Curvature", f"{treatment_results['mean_curv']:.3f}", 
                                      f"{treatment_results['mean_curv'] - results['mean_curv']:+.3f}")

                    if treatment_results['diagnosis'] != results['diagnosis']:
                        st.success(f"Treatment successfully altered the topological state: {treatment_results['diagnosis']}")
                    else:
                        st.info("Topological state remained stable.")

    else:
        # Welcome Screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to KCTHO</h2>
            <p>Select a condition from the sidebar and click <b>RUN ANALYSIS</b> to begin.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()