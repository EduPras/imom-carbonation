"""Global SHAP interpretability tab view component."""

import streamlit as st
from interpretability.visualizer import render_global_shap_plots


def render_shap_tab() -> None:
    """Render global SHAP Interpretability tab."""
    st.markdown("### SHAP Global Interpretability Analysis")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values explain the influence of each concrete mix property and exposure condition "
        "on the final target properties. These plots show whether a feature increases or decreases the target value."
    )

    shap_model_choices = ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]
    selected_shap_model = st.selectbox(
        "Select Model to View SHAP Analysis:",
        shap_model_choices,
        key="shap_model_select",
    )

    output_names = ["Carbonation Depth (mm)", "Cube Compressive Strength (MPa)"]
    selected_target = st.selectbox(
        "Select Target Variable to Analyze:",
        output_names,
        key="shap_target_select",
    )

    render_global_shap_plots(selected_shap_model, selected_target)
