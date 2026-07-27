"""UI styling and global page setup for Streamlit."""

import streamlit as st


def setup_page_styles() -> None:
    """Set up Streamlit page configuration and custom CSS."""
    st.set_page_config(
        page_title="Concrete Carbonation & Compressive Strength Predictor",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Green Predict Button override for primary buttons */
        div.stButton > button[kind="primary"] {
            background-color: #28a745 !important;
            border-color: #28a745 !important;
            color: white !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #218838 !important;
            border-color: #1e7e34 !important;
        }

        /* Native HTML5 validation styling for out-of-range inputs */
        input:out-of-range {
            border: 2px solid #FF4B4B !important;
            background-color: rgba(255, 75, 75, 0.1) !important;
            color: #FF4B4B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
