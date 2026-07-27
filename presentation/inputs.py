"""Input components module for Home Screen and Sidebar."""

import numpy as np
import streamlit as st


def init_session_state() -> None:
    """Initialize default values in session state for inputs."""
    defaults = {
        "water_abs": 5.9,
        "w_b_ratio": 0.53,
        "fine_agg": 624.0,
        "gravel": 514.0,
        "ra_content": 546.0,
        "super_p": 0.65,
        "carbon_conc": 5.2,
        "exp_time": 208.0,
        "predicted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_input_array() -> np.ndarray:
    """Get the current inputs from session state as a NumPy array."""
    return np.array(
        [
            [
                st.session_state["water_abs"],
                st.session_state["w_b_ratio"],
                st.session_state["fine_agg"],
                st.session_state["gravel"],
                st.session_state["ra_content"],
                st.session_state["super_p"],
                st.session_state["carbon_conc"],
                st.session_state["exp_time"],
            ]
        ]
    )


def _render_input_fields(container) -> None:
    """Render the 8 numeric input fields within a specific container (e.g. st or col)."""
    container.number_input(
        "Water absorption (%) [0.0 - 16.58]",
        min_value=0.0,
        max_value=16.58,
        step=0.1,
        key="water_abs",
        help="Water absorption percentage of the concrete.",
    )
    container.number_input(
        "Effective w/b ratio [0.25 - 1.02]",
        min_value=0.25,
        max_value=1.02,
        step=0.01,
        key="w_b_ratio",
        help="Ratio of effective water content to binder content.",
    )
    container.number_input(
        "Fine aggregate (kg/m³) [357.65 - 998.0]",
        min_value=357.65,
        max_value=998.0,
        step=1.0,
        key="fine_agg",
        help="Amount of fine aggregate (sand) used in the concrete mix.",
    )
    container.number_input(
        "Gravel content (kg/m³) [0.0 - 689.0]",
        min_value=0.0,
        max_value=689.0,
        step=1.0,
        key="gravel",
        help="Amount of gravel used in the concrete mix.",
    )
    container.number_input(
        "RA content (kg/m³) [0.0 - 357.8]",
        min_value=0.0,
        max_value=357.8,
        step=1.0,
        key="ra_content",
        help="Amount of recycled aggregate used in the concrete mix.",
    )
    container.number_input(
        "Superplasticizer (kg/m³) [0.0 - 7.31]",
        min_value=0.0,
        max_value=7.31,
        step=0.05,
        key="super_p",
        help="Water reducer additive used to improve workability.",
    )
    container.number_input(
        "Carbon concentration (%) [0.05 - 50.0]",
        min_value=0.05,
        max_value=50.0,
        step=0.1,
        key="carbon_conc",
        help="CO2 concentration during the carbonation testing.",
    )
    container.number_input(
        "Exposure time (days) [7.0 - 3650.0]",
        min_value=7.0,
        max_value=3650.0,
        step=1.0,
        key="exp_time",
        help="Duration of concrete exposure to CO2 in days.",
    )


def render_home_inputs() -> None:
    """Render the inputs in a 2x4 grid on the home screen."""
    st.markdown("### Mix Proportions & Exposure Conditions")
    
    # Create a 4-column layout for the first row of inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.number_input("Water absorption (%) [0.0 - 16.58]", min_value=0.0, max_value=16.58, step=0.1, key="water_abs")
        st.number_input("RA content (kg/m³) [0.0 - 357.8]", min_value=0.0, max_value=357.8, step=1.0, key="ra_content")
    with col2:
        st.number_input("Effective w/b ratio [0.25 - 1.02]", min_value=0.25, max_value=1.02, step=0.01, key="w_b_ratio")
        st.number_input("Superplasticizer (kg/m³) [0.0 - 7.31]", min_value=0.0, max_value=7.31, step=0.05, key="super_p")
    with col3:
        st.number_input("Fine aggregate (kg/m³) [357.65 - 998.0]", min_value=357.65, max_value=998.0, step=1.0, key="fine_agg")
        st.number_input("Carbon concentration (%) [0.05 - 50.0]", min_value=0.05, max_value=50.0, step=0.1, key="carbon_conc")
    with col4:
        st.number_input("Gravel content (kg/m³) [0.0 - 689.0]", min_value=0.0, max_value=689.0, step=1.0, key="gravel")
        st.number_input("Exposure time (days) [7.0 - 3650.0]", min_value=7.0, max_value=3650.0, step=1.0, key="exp_time")


def render_sidebar_inputs() -> None:
    """Render the inputs vertically in the sidebar."""
    st.sidebar.markdown("## Input Parameters")
    st.sidebar.markdown("Adjust the properties to recalculate predictions:")
    _render_input_fields(st.sidebar)


def check_validation() -> tuple[bool, list[str]]:
    """Verify that all inputs in session state fall within their validated ranges."""
    ranges = {
        "water_abs": (0.0, 16.58, "Water absorption"),
        "w_b_ratio": (0.25, 1.02, "Effective w/b ratio"),
        "fine_agg": (357.65, 998.0, "Fine aggregate"),
        "gravel": (0.0, 689.0, "Gravel content"),
        "ra_content": (0.0, 357.8, "RA content"),
        "super_p": (0.0, 7.31, "Superplasticizer"),
        "carbon_conc": (0.05, 50.0, "Carbon concentration"),
        "exp_time": (7.0, 3650.0, "Exposure time"),
    }
    
    valid = True
    errors = []
    for key, (vmin, vmax, name) in ranges.items():
        val = st.session_state.get(key)
        if val is not None and (val < vmin or val > vmax):
            valid = False
            errors.append(f"• **{name}** must be between {vmin} and {vmax}")
            
    return valid, errors

