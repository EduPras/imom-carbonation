"""Visualization components for SHAP plots (Waterfall, Beeswarm, Bar)."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st

from .shap_engine import get_waterfall_data, load_global_shap_data


def render_waterfall_chart(
    model_name: str,
    target_name: str,
    input_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """Render single-prediction SHAP Waterfall chart using Plotly."""
    target_idx = 0 if "Carbonation" in target_name else 1
    unit = "mm" if target_idx == 0 else "MPa"
    feature_values = input_data[0]

    try:
        base_val, shap_contribs = get_waterfall_data(
            model_name, target_idx, input_data
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
            title=f"Waterfall Contribution Plot — {model_name} for {target_name}",
            showlegend=False,
            template="plotly_white",
            height=550,
            yaxis_title=f"Value ({unit})",
            xaxis_tickangle=-35,
        )

        st.plotly_chart(fig_wf, use_container_width=True)

    except Exception as e:
        st.error(f"Could not compute SHAP waterfall plot: {e}")


def render_global_shap_plots(model_name: str, target_name: str) -> None:
    """Render global out-of-fold Beeswarm and Bar SHAP plots."""
    shap_data = load_global_shap_data(model_name)

    if shap_data is None:
        st.warning(
            f"No SHAP data found for **{model_name}** yet. "
            "Please run `python main.py` to generate the SHAP values."
        )
        return

    try:
        shap_values = shap_data["shap_values"]
        X_scaled = shap_data["X_scaled"]
        feature_names = shap_data["feature_names"]
        target_idx = 0 if "Carbonation" in target_name else 1
        target_shap_values = shap_values[:, :, target_idx]

        # Summary plot (Beeswarm)
        st.markdown("#### SHAP Summary - Beeswarm Plot")
        st.markdown(
            "Features are ranked by importance (top features have the most impact). "
            "Each dot is a sample. **Red** indicates a high feature value, and **Blue** indicates a low value. "
            "Points to the right push the model prediction **higher**, points to the left push it **lower**."
        )

        fig_beeswarm, _ = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            target_shap_values,
            features=X_scaled,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig_beeswarm)
        plt.close(fig_beeswarm)

        st.markdown("---")
        st.markdown("#### Mean Absolute SHAP Value (Feature Importance) Bar Chart")
        st.markdown("This bar chart ranks the features by their average magnitude of impact on the target output.")

        fig_bar, _ = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            target_shap_values,
            features=X_scaled,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    except Exception as e:
        st.error(f"Failed to render SHAP plots for {model_name}: {e}")
