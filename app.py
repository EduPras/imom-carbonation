import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Set page config for a premium SaaS look
st.set_page_config(
    page_title="Concrete Carbonation & Compressive Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for custom dark theme styling, glassmorphism card design, and custom fonts
st.markdown(
    """
    <style>
    /* Custom fonts and global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek gradient background for title header */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Styled metric cards */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode override for metric cards if browser is in dark mode */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #1e1e1e;
            border-color: #333333;
            color: #ffffff;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🏗️ Concrete Carbonation & Strength Predictor</h1>
        <p class="header-subtitle">Analyze, predict, and compare 8 different Machine Learning models trained with 5-Fold Cross-Validation & Optuna tuning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Import predictor implementations
from model.predictor_implementations import get_all_predictors


# Cache predictors loading to speed up interaction
@st.cache_resource
def load_predictors():
    return get_all_predictors(Path("checkpoints"))


# Try to load predictors
try:
    predictors = load_predictors()
except Exception as e:
    st.error(
        "### ⚠️ No trained model checkpoints found!\n"
        "Please wait for the training process to complete or run the training script in your terminal:\n"
        "```bash\n"
        "uv run python main.py\n"
        "```"
    )
    st.stop()

# Sidebar: Inputs Panel
st.sidebar.markdown("## 📊 Input Parameters")
st.sidebar.markdown(
    "Adjust the properties of the concrete mix and exposure conditions below:"
)

# Define sliders based on actual min/max/mean of the dataset
water_absorption = st.sidebar.slider(
    "Water absorption (%)",
    min_value=0.0,
    max_value=17.0,
    value=5.9,
    step=0.1,
    help="Water absorption percentage of the concrete.",
)

w_b_ratio = st.sidebar.slider(
    "Effective water-to-binder ratio",
    min_value=0.25,
    max_value=1.05,
    value=0.53,
    step=0.01,
    help="Ratio of effective water content to binder content.",
)

fine_agg = st.sidebar.slider(
    "Fine aggregate content (kg/m³)",
    min_value=350,
    max_value=1000,
    value=624,
    step=1,
    help="Amount of fine aggregate (sand) used in the concrete mix.",
)

gravel = st.sidebar.slider(
    "Gravel content (kg/m³)",
    min_value=0,
    max_value=1350,
    value=514,
    step=1,
    help="Amount of gravel used in the concrete mix.",
)

ra_content = st.sidebar.slider(
    "Recycled Aggregate (RA) content (kg/m³)",
    min_value=0,
    max_value=1300,
    value=546,
    step=1,
    help="Amount of recycled aggregate used in the concrete mix.",
)

superplasticizer = st.sidebar.slider(
    "Superplasticizer (kg/m³)",
    min_value=0.0,
    max_value=7.5,
    value=0.65,
    step=0.05,
    help="Water reducer additive used to improve workability.",
)

carbon_conc = st.sidebar.slider(
    "Carbon concentration (%)",
    min_value=0.0,
    max_value=20.0,
    value=5.2,
    step=0.1,
    help="CO2 concentration during the carbonation testing.",
)

exposure_time = st.sidebar.slider(
    "Exposure time (days)",
    min_value=7,
    max_value=3650,
    value=208,
    step=1,
    help="Duration of concrete exposure to CO2 in days.",
)

# Convert inputs to NumPy array for prediction
input_data = np.array(
    [
        [
            water_absorption,
            w_b_ratio,
            fine_agg,
            gravel,
            ra_content,
            superplasticizer,
            carbon_conc,
            exposure_time,
        ]
    ]
)

# Create tabs for interactive predictions and overall model performance
tab_predict, tab_performance = st.tabs(
    ["🔮 Interactive Predictions", "📈 Model Performance comparison"]
)

with tab_predict:
    st.markdown("### Model Predictions Comparison")
    st.markdown(
        "The metrics below display predictions for **Carbonation Depth** and **Cube Compressive Strength**."
    )

    # Make predictions across all 8 models
    prediction_results = []
    for predictor in predictors:
        name = predictor.get_name()
        preds = predictor.predict(input_data)
        # preds is of shape (1, 2) -> [Carbonation depth, Cube strength]
        prediction_results.append(
            {
                "Model": name,
                "Carbonation Depth (mm)": float(preds[0, 0]),
                "Cube Compressive Strength (MPa)": float(preds[0, 1]),
            }
        )

    df_preds = pd.DataFrame(prediction_results)

    # Highlight top/recommended model (XGBoost)
    xgb_pred = df_preds[df_preds["Model"] == "XGBoost"].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Carbonation Depth (mm) — XGBoost [Best]",
            value=f"{xgb_pred['Carbonation Depth (mm)']:.2f} mm",
        )
    with col2:
        st.metric(
            label="Cube Compressive Strength (MPa) — XGBoost [Best]",
            value=f"{xgb_pred['Cube Compressive Strength (MPa)']:.2f} MPa",
        )

    st.markdown("---")

    # Plot predictions comparison
    st.markdown("#### Comparison Chart")

    # Double bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_preds["Model"],
            y=df_preds["Carbonation Depth (mm)"],
            name="Carbonation Depth (mm)",
            marker_color="#FF4B4B",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_preds["Model"],
            y=df_preds["Cube Compressive Strength (MPa)"],
            name="Cube Strength (MPa)",
            marker_color="#1F77B4",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Side-by-Side Model Predictions",
        xaxis_title="Machine Learning Models",
        yaxis_title="Predicted Value",
        legend_title="Outputs",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Predictions Table
    st.markdown("#### Predictions Table")
    st.dataframe(
        df_preds.style.highlight_max(
            subset=["Cube Compressive Strength (MPa)"], color="#d4edda"
        ).highlight_min(subset=["Carbonation Depth (mm)"], color="#f8d7da"),
        use_container_width=True,
    )

with tab_performance:
    st.markdown("### 📊 5-Fold Cross-Validation Performance Summary")
    st.markdown(
        "Below are the performance statistics compiled across all 5 folds. Tree-based ensemble models perform exceptionally well on this tabular dataset."
    )

    # Set up performance data from our cross-validation walkthrough
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

    # Plot R2 comparisons
    fig_r2 = px.bar(
        df_perf[df_perf["Mean R2"] > 0],  # filter out heavily negative TabNet for visualization
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

    # Styled table
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
