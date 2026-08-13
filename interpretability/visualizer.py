"""Visualization components for SHAP plots (Waterfall, Beeswarm, Bar)."""

import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st
from pathlib import Path

from .shap_engine import get_waterfall_data, load_global_shap_data


def render_waterfall_chart(
    model_name: str,
    input_data: np.ndarray,
    feature_names: list[str],
    checkpoints_dir: Path = Path("checkpoints_9var")
) -> None:
    """Render single-prediction SHAP Waterfall chart using Plotly."""
    unit = "mm"
    feature_values = input_data[0]

    try:
        base_val, shap_contribs = get_waterfall_data(
            model_name, input_data, checkpoints_dir
        )

        x_labels = ["Dataset Baseline"]
        y_values = [base_val]
        measures = ["absolute"]
        text_labels = [f"{base_val:.2f} {unit}"]

        for fname, fval, sval in zip(feature_names, feature_values, shap_contribs):
            x_labels.append(f"{fname} = {fval}")
            y_values.append(sval)
            measures.append("relative")
            text_labels.append(f"{'+' if sval >= 0 else ''}{sval:.2f} {unit}")

        final_pred = base_val + float(np.sum(shap_contribs))
        x_labels.append(f"Final Prediction ({model_name})")
        y_values.append(final_pred)
        measures.append("total")
        text_labels.append(f"{final_pred:.2f} {unit}")

        fig_wf = go.Figure(
            go.Waterfall(
                name="SHAP Contribution",
                orientation="v",
                measure=measures,
                x=x_labels,
                y=y_values,
                text=text_labels,
                textposition="outside",
                connector={"line": {"color": "rgba(128, 128, 128, 0.5)"}},
                decreasing={"marker": {"color": "#FF4B4B"}},
                increasing={"marker": {"color": "#2CA02C"}},
                totals={"marker": {"color": "#1F77B4"}},
            )
        )

        fig_wf.update_layout(
            title=f"Waterfall Contribution Plot — {model_name} for Carbonation Depth",
            showlegend=False,
            template="plotly_white",
            height=550,
            yaxis_title=f"Value ({unit})",
            xaxis_tickangle=-35,
        )

        st.plotly_chart(fig_wf, use_container_width=True)

    except Exception as e:
        st.error(f"Could not compute SHAP waterfall plot: {e}")


def render_global_shap_plots(model_name: str, checkpoints_dir: Path = Path("checkpoints_9var")) -> None:
    """Render global out-of-fold Beeswarm and Bar SHAP plots."""
    shap_data = load_global_shap_data(model_name, checkpoints_dir)

    if shap_data is None:
        st.warning(
            f"No SHAP data found for **{model_name}** yet. "
            "Please run `python main.py` to generate the SHAP values."
        )
        return

    try:
        shap_values = shap_data["shap_values"]
        X_scaled = shap_data["X_scaled"]
        raw_feature_names = shap_data["feature_names"]
        
        # The raw feature names from the dataset have typos and trailing spaces (e.g. 'RA content .1')
        # We map them to clean names for all SHAP plots if the length matches
        clean_names = [
            "Water absorption (%)",
            "Effective w/b ratio",
            "Fine aggregate (kg/m³)",
            "Gravel content (kg/m³)",
            "RA content (kg/m³)",
            "Superplasticizer (kg/m³)",
            "CO2 concentration (%)",
            "Exposure time (days)",
            "Compressive strength (MPa)"
        ]
        if len(raw_feature_names) == len(clean_names):
            feature_names = clean_names
        elif len(raw_feature_names) == 7:
            feature_names = clean_names[:7]
        else:
            feature_names = raw_feature_names
            
        if len(shap_values.shape) == 3:
            target_shap_values = shap_values[:, :, 0]
        else:
            target_shap_values = shap_values

        # Scale SHAP values back to real physical units (mm)
        import joblib
        scaler_y_path = checkpoints_dir / "scalers" / "fold_1_y.pkl"
        if scaler_y_path.exists():
            scaler_y = joblib.load(scaler_y_path)
            target_shap_values = target_shap_values * scaler_y.scale_[0]

        # Calculate mean absolute SHAP values for sorting
        mean_abs_shap = np.mean(np.abs(target_shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)

        # Summary plot (Beeswarm approximation via Scatter with jitter)
        st.markdown("#### SHAP Summary - Beeswarm Plot")
        st.markdown(
            "Features are ranked by importance (top features have the most impact). "
            "Each dot is a sample. **Red** indicates a high feature value, and **Blue** indicates a low value. "
            "Points to the right push the model prediction **higher**, points to the left push it **lower**."
        )

        fig_beeswarm = go.Figure()
        for i, feat_idx in enumerate(sorted_idx):
            feat_name = feature_names[feat_idx]
            shap_vals = target_shap_values[:, feat_idx]
            feat_vals = X_scaled[:, feat_idx]
            
            # Random jitter to simulate beeswarm spread
            jitter = np.random.uniform(-0.25, 0.25, size=len(shap_vals))
            
            fig_beeswarm.add_trace(go.Scatter(
                x=shap_vals,
                y=np.full(len(shap_vals), i) + jitter,
                mode="markers",
                marker=dict(
                    color=feat_vals,
                    colorscale="RdBu_r",
                    size=6,
                    opacity=0.7,
                    showscale=(i == 0),
                    colorbar=dict(title="Feature Value", thickness=15, len=0.75) if i == 0 else None
                ),
                name=feat_name,
                hoverinfo="text",
                text=[f"{feat_name}<br>Scaled Value: {fv:.2f}<br>SHAP: {sv:.3f}" for sv, fv in zip(shap_vals, feat_vals)]
            ))

        fig_beeswarm.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(sorted_idx))),
                ticktext=[feature_names[idx] for idx in sorted_idx]
            ),
            xaxis=dict(
                title="SHAP Value (Impact on Model Output in mm)",
                showgrid=True,
                gridcolor="#E0E0E0",
                gridwidth=0.8,
                dtick=2.0,
                zeroline=True,
                zerolinecolor="#666666",
                zerolinewidth=1.2,
            ),
            template="plotly_white",
            height=600,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_beeswarm, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Mean Absolute SHAP Value (Feature Importance) Bar Chart")
        st.markdown("This bar chart ranks the features by their average magnitude of impact on the target output.")

        fig_bar = go.Figure(go.Bar(
            x=mean_abs_shap[sorted_idx],
            y=[feature_names[idx] for idx in sorted_idx],
            orientation='h',
            marker_color="#FF4B4B",
            text=[f"{val:.3f}" for val in mean_abs_shap[sorted_idx]],
            textposition='outside'
        ))
        
        fig_bar.update_layout(
            xaxis_title="mean(|SHAP value|) (average impact on carbonation depth in mm)",
            yaxis_title="",
            template="plotly_white",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        


    except Exception as e:
        st.error(f"Failed to render SHAP plots for {model_name}: {e}")
