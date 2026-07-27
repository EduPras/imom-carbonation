"""Learning curves tab view component."""

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_learning_tab(checkpoints_dir: Path = Path("checkpoints")) -> None:
    """Render Learning Curves tab component."""
    st.markdown("### 📉 Learning Curves")
    st.markdown(
        "Analyze the training vs. validation loss over time for models that iterate (e.g., boosting rounds or epochs). "
        "A large gap between training and validation typically indicates **overfitting**, while a high plateau indicates **underfitting**."
    )

    model_choices = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
        "TabNet",
        "Random Forest",
        "SVR",
        "Ridge Regression",
    ]
    selected_model = st.selectbox(
        "Select Model to View Learning Curve:", model_choices
    )

    history_file = checkpoints_dir / selected_model / "fold_1_history.json"

    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        if len(train_loss) > 1:
            df_history = pd.DataFrame(
                {
                    "Iteration/Epoch": range(1, len(train_loss) + 1),
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                }
            )

            fig_lc = go.Figure()
            fig_lc.add_trace(
                go.Scatter(
                    x=df_history["Iteration/Epoch"],
                    y=df_history["Training Loss"],
                    mode="lines",
                    name="Training Loss",
                    line=dict(color="#1F77B4"),
                )
            )
            fig_lc.add_trace(
                go.Scatter(
                    x=df_history["Iteration/Epoch"],
                    y=df_history["Validation Loss"],
                    mode="lines",
                    name="Validation Loss",
                    line=dict(color="#FF4B4B"),
                )
            )

            fig_lc.update_layout(
                title=f"{selected_model} - Learning Curve (Fold 1)",
                xaxis_title="Iteration / Epoch",
                yaxis_title="Loss",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig_lc, use_container_width=True)

        elif len(train_loss) == 1:
            st.info(
                f"**{selected_model}** does not train iteratively over epochs. The final training loss was {train_loss[0]:.4f} and validation loss was {val_loss[0]:.4f}."
            )
        else:
            st.warning(
                f"History data for {selected_model} is empty or malformed."
            )

    else:
        st.warning(
            f"No learning curve history found for **{selected_model}**. Ensure you have run the training pipeline with metrics tracking enabled."
        )
