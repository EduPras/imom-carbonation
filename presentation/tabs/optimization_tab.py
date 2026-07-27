"""Mix Optimization Tab view component."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from model.base_predictor import BasePredictor


def render_optimization_tab(predictors: list[BasePredictor], input_data: np.ndarray) -> None:
    """Render the Monte Carlo Explorer tab."""
    st.markdown("### Monte Carlo Explorer")
    st.markdown(
        "This tool simulates thousands of random concrete mix designs while keeping your current **Exposure Time** and **CO₂ Concentration** constant. "
        "It uses the **XGBoost** model to predict performance, allowing you to discover the exact input ranges that lead to optimal concrete."
    )

    # Sidebar / Local controls for the optimization criteria
    col1, col2 = st.columns(2)
    with col1:
        model_names = [p.get_name() for p in predictors]
        # Default to XGBoost if available, else first
        default_index = model_names.index("XGBoost") if "XGBoost" in model_names else 0
        selected_model_name = st.selectbox("Select ML Model for Simulation", model_names, index=default_index)
        
    with col2:
        n_samples = st.slider(
            "Monte Carlo Samples", 
            min_value=1000, 
            max_value=1000000, 
            value=5000, 
            step=5000,
            help="High values increase accuracy but might slow down the visualization. Scatter plots are downsampled above 20,000 points to keep the browser responsive."
        )

    # Feature ranges based on the dataset characteristics
    ranges = {
        "Water absorption (%)": (0.0, 16.58),
        "Effective w/b ratio": (0.25, 1.02),
        "Fine aggregate (kg/m³)": (357.65, 998.0),
        "Gravel content (kg/m³)": (0.0, 689.0),
        "RA content (kg/m³)": (0.0, 357.8),
        "Superplasticizer (kg/m³)": (0.0, 7.31),
    }
    
    # Fixed environmental conditions from user input
    fixed_carbon_conc = input_data[0, 6]
    fixed_exposure = input_data[0, 7]
    
    # Generate random matrix
    np.random.seed(42) # For reproducibility
    X_sim = np.zeros((n_samples, 8))
    X_sim[:, 0] = np.random.uniform(ranges["Water absorption (%)"][0], ranges["Water absorption (%)"][1], n_samples)
    X_sim[:, 1] = np.random.uniform(ranges["Effective w/b ratio"][0], ranges["Effective w/b ratio"][1], n_samples)
    X_sim[:, 2] = np.random.uniform(ranges["Fine aggregate (kg/m³)"][0], ranges["Fine aggregate (kg/m³)"][1], n_samples)
    X_sim[:, 3] = np.random.uniform(ranges["Gravel content (kg/m³)"][0], ranges["Gravel content (kg/m³)"][1], n_samples)
    X_sim[:, 4] = np.random.uniform(ranges["RA content (kg/m³)"][0], ranges["RA content (kg/m³)"][1], n_samples)
    X_sim[:, 5] = np.random.uniform(ranges["Superplasticizer (kg/m³)"][0], ranges["Superplasticizer (kg/m³)"][1], n_samples)
    X_sim[:, 6] = fixed_carbon_conc
    X_sim[:, 7] = fixed_exposure

    # Find the selected predictor
    selected_predictor = next(p for p in predictors if p.get_name() == selected_model_name)
    
    # Predict
    preds = selected_predictor.predict(X_sim) # shape: (n_samples, 2)
    carb_preds = preds[:, 0]
    strength_preds = preds[:, 1]
    
    # Create DataFrame for plotting
    df_sim = pd.DataFrame(X_sim[:, :6], columns=list(ranges.keys()))
    df_sim["Carbonation Depth (mm)"] = carb_preds
    df_sim["Compressive Strength (MPa)"] = strength_preds
    
    # Plot 1: Scatter plot highlighting the optimal zone
    st.markdown("---")
    st.markdown("#### 1. Performance Distribution")
    st.markdown(f"Simulated **{n_samples:,}** random mixes using **{selected_model_name}**.")
    st.info("Draw a box (or lasso) on the scatter plot below to select your desired performance zone! The Parallel Coordinates plot will automatically update to show the recipes for those selected mixes.")
    
    # Downsample logic for plotting
    max_scatter_points = 20000
    if n_samples > max_scatter_points:
        st.markdown(f"*(Scatter plot visually downsampled to {max_scatter_points:,} points for browser performance)*.")
        df_scatter = df_sim.sample(n=max_scatter_points, random_state=42).reset_index(drop=True)
    else:
        df_scatter = df_sim.copy().reset_index(drop=True)

    fig_scatter = px.scatter(
        df_scatter, 
        x="Carbonation Depth (mm)", 
        y="Compressive Strength (MPa)", 
        opacity=0.5,
        hover_data=["Effective w/b ratio", "RA content (kg/m³)"],
        render_mode="webgl"
    )
    
    fig_scatter.update_traces(marker=dict(color="#1F77B4"))
    fig_scatter.update_layout(
        template="plotly_white", 
        height=500,
        dragmode="select"
    )
    
    # Native Streamlit bi-directional selection
    event = st.plotly_chart(fig_scatter, use_container_width=True, on_select="rerun", selection_mode=("box", "lasso"))
    
    # Plot 2: Parallel Coordinates of the OPTIMAL mixes
    st.markdown("---")
    st.markdown("#### 2. Input Variables Leading to Selected Results")
    
    # Check if user selected points
    if event and event.selection and event.selection.points:
        selected_indices = [pt["point_index"] for pt in event.selection.points]
        df_opt = df_scatter.iloc[selected_indices]
        optimal_count = len(df_opt)
        
        st.success(f"You selected **{optimal_count:,}** mixes. The Parallel Coordinates plot below reveals the exact ranges and combinations of ingredients required to achieve this specific performance.")
        
        color_var = st.pills(
            "Color lines by:",
            options=["Compressive Strength (MPa)", "Carbonation Depth (mm)"],
            default="Compressive Strength (MPa)"
        )
        if not color_var:
            color_var = "Compressive Strength (MPa)"
            
        color_scale = "viridis" if "Strength" in color_var else "inferno"
        
        # Parallel coordinates plot
        fig_par = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df_opt[color_var], 
                    colorscale=color_scale, 
                    showscale=True, 
                    cmin=df_opt[color_var].min(), 
                    cmax=df_opt[color_var].max()
                ),
                dimensions=[
                    dict(range=[ranges["Effective w/b ratio"][0], ranges["Effective w/b ratio"][1]], label="w/b ratio", values=df_opt["Effective w/b ratio"]),
                    dict(range=[ranges["RA content (kg/m³)"][0], ranges["RA content (kg/m³)"][1]], label="RA content", values=df_opt["RA content (kg/m³)"]),
                    dict(range=[ranges["Superplasticizer (kg/m³)"][0], ranges["Superplasticizer (kg/m³)"][1]], label="Superplasticizer", values=df_opt["Superplasticizer (kg/m³)"]),
                    dict(range=[ranges["Water absorption (%)"][0], ranges["Water absorption (%)"][1]], label="Water absorption", values=df_opt["Water absorption (%)"]),
                    dict(range=[ranges["Fine aggregate (kg/m³)"][0], ranges["Fine aggregate (kg/m³)"][1]], label="Fine aggregate", values=df_opt["Fine aggregate (kg/m³)"]),
                    dict(range=[ranges["Gravel content (kg/m³)"][0], ranges["Gravel content (kg/m³)"][1]], label="Gravel content", values=df_opt["Gravel content (kg/m³)"]),
                ]
            )
        )
        fig_par.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_par, use_container_width=True)
        
        # Statistical summary of optimal inputs
        st.markdown("**Summary of Mix Ingredients (Min - Max) for your selection:**")
        df_summary = df_opt.iloc[:, :6].describe().loc[['min', 'max']].T
        st.dataframe(df_summary.style.format("{:.2f}"), use_container_width=True)
        
    else:
        st.warning("**No points selected.** Please draw a box or lasso around a cluster of points on the scatter plot above to see the mix proportions required to achieve those results.")
