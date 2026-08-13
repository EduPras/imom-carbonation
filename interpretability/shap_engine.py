"""Engine for loading and calculating SHAP values for model interpretability."""

from pathlib import Path
import joblib
import numpy as np
import shap


def get_waterfall_data(
    model_name: str,
    X_input: np.ndarray,
    checkpoints_dir: Path = Path("checkpoints_9var"),
) -> tuple[float, np.ndarray]:
    """Calculate single-instance SHAP values in target physical units.

    Args:
        model_name: Name of the tree-based model (e.g. 'XGBoost', 'LightGBM').
        X_input: Array of shape (1, num_features) with feature values.
        checkpoints_dir: Path to checkpoints directory.

    Returns:
        tuple containing:
            - base_val_real (float): Expected baseline value in real units.
            - shap_real (np.ndarray): Feature contribution array in real units.
    """
    model_path = checkpoints_dir / model_name / "fold_1.joblib"
    scaler_x_path = checkpoints_dir / "scalers" / "fold_1_X.pkl"
    scaler_y_path = checkpoints_dir / "scalers" / "fold_1_y.pkl"

    model = joblib.load(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    X_scaled = scaler_x.transform(X_input)
    raw_model = getattr(model, "model", model)

    if hasattr(raw_model, "estimators_"):
        estimator = raw_model.estimators_[0]
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_scaled)
        base_val_scaled = explainer.expected_value
        if isinstance(base_val_scaled, (np.ndarray, list)):
            base_val_scaled = base_val_scaled[0]
        shap_real = sv[0] * scaler_y.scale_[0]
        base_val_real = (
            base_val_scaled * scaler_y.scale_[0]
            + scaler_y.mean_[0]
        )
    else:
        explainer = shap.TreeExplainer(raw_model)
        sv = explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            # For some multi-class or older SHAP versions
            sv = sv[0]
            
        # sv is expected to be shape (n_samples, n_features) or (n_samples, n_features, 1)
        if len(sv.shape) == 3:
            shap_real = sv[0, :, 0] * scaler_y.scale_[0]
        else:
            shap_real = sv[0] * scaler_y.scale_[0]
            
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

        base_val_real = (
            base_val * scaler_y.scale_[0]
            + scaler_y.mean_[0]
        )

    return float(base_val_real), shap_real


def load_global_shap_data(
    model_name: str, checkpoints_dir: Path = Path("checkpoints_9var")
) -> dict | None:
    """Load out-of-fold global SHAP dataset pickle file.

    Args:
        model_name: Name of the tree-based model.
        checkpoints_dir: Path to checkpoints directory.

    Returns:
        dict containing shap_values, X_scaled, feature_names, output_names if exists, else None.
    """
    shap_file = checkpoints_dir / model_name / "shap_data.pkl"
    if shap_file.exists():
        return joblib.load(shap_file)
    return None
