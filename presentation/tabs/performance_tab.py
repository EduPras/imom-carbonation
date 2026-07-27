"""Model performance comparison tab view component."""

import pandas as pd
import plotly.express as px
import streamlit as st


def render_performance_tab() -> None:
    """Render 5-Fold CV Performance summary tab."""
    st.markdown("### 📊 5-Fold Cross-Validation Performance Summary")
    st.markdown(
        "Below are the performance statistics compiled across all 5 folds. Tree-based ensemble models perform exceptionally well on this tabular dataset."
    )

    performance_data = {
        "Model": [
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "SVR",
            "Random Forest",
            "MLP",
            "Ridge Regression",
            "TabNet",
        ],
        "Mean R2": [0.9780, 0.9688, 0.9638, 0.9498, 0.9444, 0.8384, 0.4848, -5.8806],
        "R2 Std": [0.0081, 0.0081, 0.0105, 0.0153, 0.0104, 0.0523, 0.0668, 3.9396],
        "Mean MSE": [0.0200, 0.0302, 0.0345, 0.0456, 0.0529, 0.1521, 0.5095, 6.6308],
        "MSE Std": [0.0041, 0.0098, 0.0094, 0.0087, 0.0073, 0.0360, 0.1240, 3.6142],
        "Mean MAE": [0.0740, 0.1005, 0.1176, 0.1196, 0.1604, 0.2769, 0.5274, 1.5107],
        "MAE Std": [0.0034, 0.0096, 0.0130, 0.0080, 0.0097, 0.0255, 0.0488, 0.3390],
    }

    df_perf = pd.DataFrame(performance_data)

    fig_r2 = px.bar(
        df_perf[df_perf["Mean R2"] > 0],
        x="Model",
        y="Mean R2",
        error_y="R2 Std",
        title="Model R-Squared Comparison (Higher is Better)",
        labels={"Mean R2": "R² Score"},
        color="Mean R2",
        color_continuous_scale="Blues",
    )
    fig_r2.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("#### Performance Metrics Table")
    formatted_data = []
    for _, row in df_perf.iterrows():
        formatted_data.append(
            {
                "Model": row["Model"],
                "R² Score": f"{row['Mean R2']:.4f} ± {row['R2 Std']:.4f}",
                "Mean Squared Error (MSE)": f"{row['Mean MSE']:.4f} ± {row['MSE Std']:.4f}",
                "Mean Absolute Error (MAE)": f"{row['Mean MAE']:.4f} ± {row['MAE Std']:.4f}",
            }
        )
    st.table(pd.DataFrame(formatted_data))
