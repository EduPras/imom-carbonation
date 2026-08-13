import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from interpretability.shap_engine import get_waterfall_data

# ------------------------------------------------------------------------------
# 1. LATEX & PLOTLY CONFIGURATION
# ------------------------------------------------------------------------------
# LaTeX typically uses 72.27 points per inch, but Plotly/Kaleido exports PDFs
# assuming 96 pixels per inch (PPI). To prevent LaTeX from scaling your font,
# you MUST generate the figure at the exact physical width of your LaTeX document.

# ---- USER CONFIGURATION ----
# What is your exact \linewidth or \textwidth in LaTeX?
# (e.g., standard single-column is often ~5.5 inches, two-column is ~3.5 inches)
LATEX_LINE_WIDTH_INCHES = 5.5

# Convert inches to Plotly's assumed 96 PPI
PLOTLY_WIDTH = int(LATEX_LINE_WIDTH_INCHES * 96)
# Keep a reasonable aspect ratio for height (e.g. 2.8 - 3.2 inches)
PLOTLY_HEIGHT_SHAP = int(3.2 * 96)
PLOTLY_HEIGHT_LEARNING = int(3.0 * 96)

# Set global font to Carlito, 8pt for all Plotly figures
latex_template = pio.templates["plotly_white"]
latex_template.layout.font = dict(family="Carlito, sans-serif", size=8, color="black")
latex_template.layout.title.font = dict(
    family="Carlito, sans-serif", size=9, color="black"
)
pio.templates.default = latex_template

# Ensure figures directory exists
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Select the model and checkpoint directory to generate figures for (ALL 9 VARIABLES)
MODEL_NAME = "XGBoost"
CHECKPOINTS_DIR = Path("checkpoints_9var")

print(f"Generating LaTeX-ready figures for {MODEL_NAME} ({CHECKPOINTS_DIR.name})...")

# ------------------------------------------------------------------------------
# 2. SHAP GLOBAL PLOTS
# ------------------------------------------------------------------------------


def generate_shap_plots():
    shap_file = CHECKPOINTS_DIR / MODEL_NAME / "shap_data.pkl"
    if not shap_file.exists():
        print(f"Skipping SHAP plots: {shap_file} not found.")
        return

    shap_data = joblib.load(shap_file)
    shap_values = shap_data["shap_values"]
    X_scaled = shap_data["X_scaled"]

    clean_names = [
        "Water absorption (%)",
        "Effective w/b ratio",
        "Fine aggregate (kg/m³)",
        "Gravel content (kg/m³)",
        "RA content (kg/m³)",
        "Superplasticizer (kg/m³)",
        "CO2 concentration (%)",
        "Exposure time (days)",
        "Compressive strength (MPa)",
    ]

    # Adapt feature names based on 7-var or 9-var
    if X_scaled.shape[1] == 7:
        feature_names = clean_names[:7]
    elif X_scaled.shape[1] == 9:
        feature_names = clean_names
    else:
        feature_names = shap_data["feature_names"]

    if len(shap_values.shape) == 3:
        target_shap_values = shap_values[:, :, 0]
    else:
        target_shap_values = shap_values

    # Scale SHAP values to mm
    scaler_y_path = CHECKPOINTS_DIR / "scalers" / "fold_1_y.pkl"
    if scaler_y_path.exists():
        scaler_y = joblib.load(scaler_y_path)
        target_shap_values = target_shap_values * scaler_y.scale_[0]

    mean_abs_shap = np.mean(np.abs(target_shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    # 2A. BEESWARM PLOT (with frequent vertical grid lines for carbonation depth)
    fig_beeswarm = go.Figure()
    for i, feat_idx in enumerate(sorted_idx):
        feat_name = feature_names[feat_idx]
        shap_vals = target_shap_values[:, feat_idx]
        feat_vals = X_scaled[:, feat_idx]

        jitter = np.random.uniform(-0.25, 0.25, size=len(shap_vals))

        fig_beeswarm.add_trace(
            go.Scatter(
                x=shap_vals,
                y=np.full(len(shap_vals), i) + jitter,
                mode="markers",
                marker=dict(
                    color=feat_vals,
                    colorscale="RdBu_r",
                    size=3,  # Smaller dots for LaTeX
                    opacity=0.7,
                    showscale=(i == 0),
                    colorbar=dict(
                        title="Feature Value (Z-Score)",
                        thickness=10,
                        len=0.8,
                        title_font=dict(size=8),
                        tickfont=dict(size=8),
                    )
                    if i == 0
                    else None,
                ),
                showlegend=False,
            )
        )

    fig_beeswarm.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(sorted_idx))),
            ticktext=[feature_names[idx] for idx in sorted_idx],
        ),
        xaxis=dict(
            title="SHAP Value (Impact on Carbonation Depth in mm)",
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=0.8,
            dtick=2.0,  # Vertical lines every 2 mm
            zeroline=True,
            zerolinecolor="#666666",
            zerolinewidth=1.2,
        ),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT_SHAP,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    fig_beeswarm.write_image(FIGURES_DIR / "shap_beeswarm.pdf")
    print(f"Saved {FIGURES_DIR / 'shap_beeswarm.pdf'}")

    # 2B. BAR CHART (GLOBAL FEATURE IMPORTANCE)
    fig_bar = go.Figure(
        go.Bar(
            x=mean_abs_shap[sorted_idx],
            y=[feature_names[idx] for idx in sorted_idx],
            orientation="h",
            marker_color="#FF4B4B",
            text=[f"{val:.2f}" for val in mean_abs_shap[sorted_idx]],
            textposition="outside",
        )
    )

    fig_bar.update_layout(
        xaxis=dict(
            title="mean(|SHAP value|) (average impact in mm)",
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=0.8,
            dtick=1.0,
        ),
        width=PLOTLY_WIDTH,
        height=PLOTLY_HEIGHT_SHAP,
        margin=dict(l=10, r=20, t=10, b=30),
    )
    fig_bar.write_image(FIGURES_DIR / "shap_bar.pdf")
    print(f"Saved {FIGURES_DIR / 'shap_bar.pdf'}")


# ------------------------------------------------------------------------------
# 3. LOCAL SHAP PLOT (EXPLANATION FOR A SINGLE SAMPLE)
# ------------------------------------------------------------------------------


def generate_local_shap_plot():
    # Representative sample with all 9 input features
    sample_input = np.array([[5.9, 0.53, 624.0, 514.0, 546.0, 0.65, 5.2, 208.0, 45.0]])

    clean_names = [
        "Water absorption (%)",
        "Effective w/b ratio",
        "Fine aggregate (kg/m³)",
        "Gravel content (kg/m³)",
        "RA content (kg/m³)",
        "Superplasticizer (kg/m³)",
        "CO2 concentration (%)",
        "Exposure time (days)",
        "Compressive strength (MPa)",
    ]

    try:
        base_val, shap_contribs = get_waterfall_data(
            MODEL_NAME, sample_input, checkpoints_dir=CHECKPOINTS_DIR
        )

        labels = [f"{fname} = {fval}" for fname, fval in zip(clean_names, sample_input[0])]

        # Sort by absolute contribution magnitude
        sorted_idx = np.argsort(np.abs(shap_contribs))
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_contribs = [shap_contribs[i] for i in sorted_idx]
        colors = ["#2CA02C" if val >= 0 else "#FF4B4B" for val in sorted_contribs]

        fig_local = go.Figure(
            go.Bar(
                x=sorted_contribs,
                y=sorted_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{val:+.2f} mm" for val in sorted_contribs],
                textposition="outside",
            )
        )

        final_pred = base_val + float(np.sum(shap_contribs))

        fig_local.update_layout(
            title=dict(
                text=f"Local SHAP Explanation — {MODEL_NAME} (Base: {base_val:.2f} mm → Pred: {final_pred:.2f} mm)",
                font=dict(size=8),
            ),
            xaxis=dict(
                title="SHAP Feature Contribution to Carbonation Depth (mm)",
                showgrid=True,
                gridcolor="#E0E0E0",
                gridwidth=0.8,
                dtick=1.0,
                zeroline=True,
                zerolinecolor="#444444",
                zerolinewidth=1.2,
            ),
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT_SHAP,
            margin=dict(l=10, r=25, t=25, b=30),
        )

        fig_local.write_image(FIGURES_DIR / "shap_local.pdf")
        print(f"Saved {FIGURES_DIR / 'shap_local.pdf'}")

    except Exception as e:
        print(f"Failed to generate local SHAP plot: {e}")


# ------------------------------------------------------------------------------
# 4. LEARNING CURVE PLOT
# ------------------------------------------------------------------------------


def generate_learning_curve():
    history_file = CHECKPOINTS_DIR / MODEL_NAME / "fold_1_history.json"
    if not history_file.exists():
        print(f"Skipping Learning Curve: {history_file} not found.")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if len(train_loss) > 1:
        df_history = pd.DataFrame(
            {
                "Epoch": range(1, len(train_loss) + 1),
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
            }
        )

        fig_lc = go.Figure()
        fig_lc.add_trace(
            go.Scatter(
                x=df_history["Epoch"],
                y=df_history["Training Loss"],
                mode="lines",
                name="Training Loss",
                line=dict(color="#1F77B4", width=1.5),
            )
        )
        fig_lc.add_trace(
            go.Scatter(
                x=df_history["Epoch"],
                y=df_history["Validation Loss"],
                mode="lines",
                name="Validation Loss",
                line=dict(color="#FF4B4B", width=1.5),
            )
        )

        fig_lc.update_layout(
            xaxis=dict(title="Boosting Round / Epoch", showgrid=True, gridcolor="#E0E0E0"),
            yaxis=dict(title="Mean Squared Error", showgrid=True, gridcolor="#E0E0E0"),
            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT_LEARNING,
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        fig_lc.write_image(FIGURES_DIR / "learning_curve.pdf")
        print(f"Saved {FIGURES_DIR / 'learning_curve.pdf'}")


# ------------------------------------------------------------------------------
# EXECUTE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    generate_shap_plots()
    generate_local_shap_plot()
    generate_learning_curve()
    print("All figures successfully generated in PDF format for LaTeX.")
