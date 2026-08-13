"""Concrete Carbonation & Compressive Strength Predictor Web Application.

Refactored using SOLID and Clean Architecture principles.
"""

from pathlib import Path
import streamlit as st

from model.predictor_implementations import get_all_predictors
from presentation.inputs import (
    get_input_array,
    init_session_state,
    render_home_inputs,
    render_sidebar_inputs,
    check_validation,
)
from presentation.styles import setup_page_styles
from presentation.tabs import (
    render_learning_tab,
    render_performance_tab,
    render_predictions_tab,
    render_shap_tab,
)

# Initialize Page Config & Styles
setup_page_styles()
init_session_state()

# Cache model predictors loading
@st.cache_resource
def load_predictors(mode: str):
    if mode == "9 Variables (All)":
        return get_all_predictors(Path("checkpoints_9var"))
    else:
        return get_all_predictors(Path("checkpoints_7var"))

try:
    predictors = load_predictors(st.session_state["feature_mode"])
except Exception as e:
    st.error(
        "### ⚠️ No trained model checkpoints found!\n"
        "Please wait for the training process to complete or run the training script in your terminal:\n"
        "```bash\n"
        "uv run python main.py\n"
        "```"
    )
    st.stop()

# Run Input Validation Check
is_valid, validation_errors = check_validation()

# View logic based on Session State
if not st.session_state["predicted"]:
    # ---------------------------
    # 1. HOME SCREEN
    # ---------------------------
    st.title("🏗️ Concrete Carbonation & Strength Predictor")
    st.markdown(
        "Analyze, predict, and compare 8 Machine Learning models trained with "
        "5-Fold Cross-Validation & Optuna tuning."
    )
    st.markdown("---")

    render_home_inputs()
    
    if not is_valid:
        st.markdown("<br>", unsafe_allow_html=True)
        st.error("⚠️ **Validation Error: Some inputs are out of the accepted range:**\n" + "\n".join(validation_errors))

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict", type="primary", use_container_width=True, disabled=not is_valid):
            st.session_state["predicted"] = True
            st.rerun()

else:
    # ---------------------------
    # 2. RESULTS SCREEN
    # ---------------------------
    render_sidebar_inputs()
    input_data = get_input_array()
    
    # Optional button to return to home screen
    if st.sidebar.button("← Back to Home"):
        st.session_state["predicted"] = False
        st.rerun()

    if not is_valid:
        st.title("⚠️ Out-of-Bounds Input Detected")
        st.error(
            "Please adjust the parameters in the left sidebar to be within their accepted experimental ranges:\n\n" + 
            "\n".join(validation_errors)
        )
    else:
        # Main Navigation Tabs (Text instead of icons)
        tab_predict, tab_performance, tab_learning, tab_shap = st.tabs(
            [
                "Interactive Predictions",
                "Model Performance",
                "Learning Curves",
                "SHAP Interpretability",
            ]
        )

        checkpoints_dir = Path("checkpoints_9var") if st.session_state["feature_mode"] == "9 Variables (All)" else Path("checkpoints_7var")

        with tab_predict:
            render_predictions_tab(predictors, input_data, checkpoints_dir)

        with tab_performance:
            render_performance_tab()

        with tab_learning:
            render_learning_tab(checkpoints_dir)

        with tab_shap:
            render_shap_tab(checkpoints_dir)
