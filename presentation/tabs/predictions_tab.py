"""Interactive Predictions tab view component."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from interpretability.visualizer import render_waterfall_chart
from model.base_predictor import BasePredictor


from pathlib import Path

def render_predictions_tab(
    predictors: list[BasePredictor], input_data: np.ndarray, checkpoints_dir: Path
) -> None:
    """Render Interactive Predictions tab content."""
    st.markdown("### Interactive Predictions")
    st.markdown(
        "The metrics below display predictions for **Carbonation Depth**."
    )

    # Make predictions across all models
    prediction_results = []
    for predictor in predictors:
        name = predictor.get_name()
        preds = predictor.predict(input_data)
        prediction_results.append(
            {
                "Model": name,
                "Carbonation Depth (mm)": float(preds[0, 0] if len(preds.shape) > 1 else preds[0]),
            }
        )

    df_preds = pd.DataFrame(prediction_results)

    # Highlight top model (XGBoost)
    xgb_pred = df_preds[df_preds["Model"] == "XGBoost"].iloc[0]

    st.metric(
        label="Carbonation Depth (mm) — XGBoost [Best]",
        value=f"{xgb_pred['Carbonation Depth (mm)']:.2f} mm",
    )

    # Single-Prediction SHAP Waterfall Analysis
    st.markdown("---")
    st.markdown("### SHAP Analysis")
    st.markdown(
        "Deconstruct the model's prediction for the current sidebar input values into individual feature contributions. "
        "The waterfall plot shows how each property pushes the prediction higher or lower from the dataset baseline."
    )

    wf_model = st.selectbox(
        "Select Model for Waterfall Explanation:",
        ["XGBoost", "LightGBM", "CatBoost", "Random Forest"],
        index=0,
        key="wf_model_select",
    )

    feature_names = [
        "Water absorption (%)",
        "Effective w/b ratio",
        "Fine aggregate (kg/m³)",
        "Gravel content (kg/m³)",
        "RA content (kg/m³)",
        "Superplasticizer (kg/m³)",
        "CO2 conc (%)",
        "Exposure time (days)",
        "Compressive strength (MPa)",
    ]

    if st.session_state["feature_mode"] != "9 Variables (All)":
        feature_names = feature_names[:7]

    render_waterfall_chart(wf_model, input_data, feature_names, checkpoints_dir)

    st.markdown("---")
    st.markdown("#### Comparison Chart")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_preds["Model"],
            y=df_preds["Carbonation Depth (mm)"],
            name="Carbonation Depth (mm)",
            marker_color="#FF4B4B",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Model Predictions Comparison",
        xaxis_title="Machine Learning Models",
        yaxis_title="Predicted Value (mm)",
        legend_title="Outputs",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
