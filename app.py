from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

APP_NAME = "Predictive Maintenance System"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
).resolve()

TASK_A_MODEL_PATH = MODEL_DIR / "xgb_failure_detection_model.pkl"
TASK_B_MODEL_PATH = MODEL_DIR / "ovr_failure_mode_model.pkl"

# Comma-separated environment variable:
# FEATURE_NAMES="temperature,pressure,humidity,vibration,rotation_speed"
FEATURE_NAMES_ENV = os.getenv("FEATURE_NAMES", "").strip()

# Optional display names for Task B classes:
# FAILURE_MODE_NAMES="0:Normal,1:Overheat,2:Mechanical Failure"
FAILURE_MODE_NAMES_ENV = os.getenv("FAILURE_MODE_NAMES", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("predictive-maintenance")


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helper functions
# ============================================================

def parse_feature_names_from_env() -> list[str]:
    """
    Read feature names from FEATURE_NAMES environment variable.
    Example:
        FEATURE_NAMES="temperature,pressure,vibration,speed"
    """
    if not FEATURE_NAMES_ENV:
        return []

    features = [
        feature.strip()
        for feature in FEATURE_NAMES_ENV.split(",")
        if feature.strip()
    ]

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(features))


def parse_failure_mode_names() -> dict[Any, str]:
    """
    Parse optional FAILURE_MODE_NAMES environment variable.

    Example:
        FAILURE_MODE_NAMES="0:Overheat,1:Mechanical Failure,2:Power Failure"
    """
    mapping: dict[Any, str] = {}

    if not FAILURE_MODE_NAMES_ENV:
        return mapping

    for item in FAILURE_MODE_NAMES_ENV.split(","):
        if ":" not in item:
            continue

        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not value:
            continue

        # Try integer class labels first.
        try:
            parsed_key: Any = int(key)
        except ValueError:
            parsed_key = key

        mapping[parsed_key] = value

    return mapping


def get_model_feature_names(model: Any) -> list[str]:
    """
    Try to retrieve feature names from a trained model/pipeline.
    Falls back to FEATURE_NAMES environment variable.
    """

    # sklearn / xgboost models trained using a pandas DataFrame
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None:
        return [str(name) for name in feature_names]

    # Some estimators may expose names through an underlying estimator.
    named_steps = getattr(model, "named_steps", None)

    if named_steps:
        for _, estimator in reversed(list(named_steps.items())):
            estimator_features = getattr(
                estimator,
                "feature_names_in_",
                None,
            )

            if estimator_features is not None:
                return [str(name) for name in estimator_features]

    # Environment fallback
    env_features = parse_feature_names_from_env()

    if env_features:
        return env_features

    return []


def validate_models_exist() -> None:
    """Fail early with a clear message if artifacts are missing."""
    missing = []

    if not TASK_A_MODEL_PATH.exists():
        missing.append(str(TASK_A_MODEL_PATH))

    if not TASK_B_MODEL_PATH.exists():
        missing.append(str(TASK_B_MODEL_PATH))

    if missing:
        raise FileNotFoundError(
            "Required model artifacts were not found:\n"
            + "\n".join(missing)
        )


# ============================================================
# Model loading
# ============================================================

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """
    Load both serialized ML models once per Streamlit process.
    """
    validate_models_exist()

    logger.info("Loading Task A model from %s", TASK_A_MODEL_PATH)
    task_a_model = joblib.load(TASK_A_MODEL_PATH)

    logger.info("Loading Task B model from %s", TASK_B_MODEL_PATH)
    task_b_model = joblib.load(TASK_B_MODEL_PATH)

    logger.info("ML models loaded successfully")

    return task_a_model, task_b_model


# ============================================================
# Prediction helpers
# ============================================================

def normalize_prediction(prediction: Any) -> Any:
    """
    Convert numpy/scalar prediction into a normal Python scalar.
    """
    if isinstance(prediction, np.ndarray):
        prediction = prediction.reshape(-1)[0]

    if isinstance(prediction, np.generic):
        return prediction.item()

    return prediction


def get_failure_mode_label(predicted_class: Any, model: Any) -> str:
    """
    Convert Task B predicted class into a human-readable label.
    Priority:
        1. FAILURE_MODE_NAMES environment variable
        2. model.classes_ mapping
        3. raw predicted class
    """
    failure_mode_mapping = parse_failure_mode_names()

    if predicted_class in failure_mode_mapping:
        return failure_mode_mapping[predicted_class]

    classes = getattr(model, "classes_", None)

    if classes is not None:
        for class_value in classes:
            if class_value == predicted_class:
                return str(class_value)

    return str(predicted_class)


def validate_input_dataframe(
    input_df: pd.DataFrame,
    expected_features: list[str],
) -> tuple[bool, str]:
    """
    Validate feature names, count and numeric values.
    """
    if input_df.empty:
        return False, "No input features were provided."

    missing = [
        feature
        for feature in expected_features
        if feature not in input_df.columns
    ]

    if missing:
        return False, f"Missing features: {', '.join(missing)}"

    extra = [
        feature
        for feature in input_df.columns
        if feature not in expected_features
    ]

    if extra:
        input_df.drop(columns=extra, inplace=True)

    # Force the exact training order.
    input_df = input_df[expected_features]

    if input_df.isnull().any().any():
        return False, "Input contains missing/null values."

    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        return False, "Input contains NaN or infinite values."

    return True, ""


def run_inference(
    input_df: pd.DataFrame,
    task_a_model: Any,
    task_b_model: Any,
) -> dict[str, Any]:
    """
    Sequential inference:

        Task A
          |
          ├── Class 0 -> Normal
          |
          └── Class 1 -> Task B -> Failure Mode
    """

    logger.info("Running Task A failure detection")

    task_a_prediction = normalize_prediction(
        task_a_model.predict(input_df)
    )

    # Convert numpy integer/float classes into Python scalars.
    try:
        task_a_prediction = int(task_a_prediction)
    except (TypeError, ValueError):
        pass

    result: dict[str, Any] = {
        "failure_detected": task_a_prediction == 1,
        "task_a_prediction": task_a_prediction,
        "failure_mode": None,
    }

    # --------------------------------------------------------
    # Class 0 -> System Normal
    # --------------------------------------------------------
    if task_a_prediction == 0:
        logger.info("Task A result: NO FAILURE")

        return result

    # --------------------------------------------------------
    # Class 1 -> Run Task B
    # --------------------------------------------------------
    if task_a_prediction == 1:
        logger.warning(
            "Task A result: FAILURE DETECTED. Running Task B."
        )

        task_b_prediction = normalize_prediction(
            task_b_model.predict(input_df)
        )

        failure_mode_label = get_failure_mode_label(
            task_b_prediction,
            task_b_model,
        )

        result["task_b_prediction"] = task_b_prediction
        result["failure_mode"] = failure_mode_label

        logger.warning(
            "Task B result: failure_mode=%s",
            failure_mode_label,
        )

        return result

    # Unexpected class
    raise ValueError(
        f"Unexpected Task A prediction: {task_a_prediction}. "
        "Expected 0 or 1."
    )


# ============================================================
# UI
# ============================================================

def render_header() -> None:
    st.title("⚙️ Predictive Maintenance System")
    st.caption(
        "Sequential ML inference for machine failure detection "
        "and failure-mode classification."
    )


def render_model_info(task_a_model: Any, task_b_model: Any) -> None:
    with st.sidebar:
        st.header("Model Information")

        st.write("**Task A**")
        st.write("Binary Failure Detection")

        st.write("**Task B**")
        st.write("Failure Mode Classification")

        st.divider()

        st.write("**Task A Model**")
        st.code(type(task_a_model).__name__)

        st.write("**Task B Model**")
        st.code(type(task_b_model).__name__)


def render_input_form(feature_names: list[str]) -> pd.DataFrame | None:
    st.subheader("Machine Sensor Input")

    if not feature_names:
        st.error(
            "Feature names could not be determined from the model. "
            "Set the FEATURE_NAMES environment variable."
        )

        st.code(
            'FEATURE_NAMES="temperature,pressure,vibration,speed"',
            language="bash",
        )

        return None

    st.info(
        f"Detected {len(feature_names)} input features. "
        "Enter the values used by the trained model."
    )

    values: dict[str, float] = {}

    with st.form("prediction_form"):

        columns = st.columns(2)

        for index, feature_name in enumerate(feature_names):
            with columns[index % 2]:
                values[feature_name] = st.number_input(
                    label=feature_name,
                    value=0.0,
                    step=0.01,
                    format="%.6f",
                    key=f"feature_{feature_name}",
                )

        submitted = st.form_submit_button(
            "🔍 Analyze Machine",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return None

    return pd.DataFrame([values], columns=feature_names)


def render_normal_result() -> None:
    st.success(
        "✅ SYSTEM NORMAL",
        icon="✅",
    )

    st.markdown(
        """
        ### No Failure Detected

        The failure detection model classified the machine as:

        **Class 0 — Normal**
        """
    )


def render_failure_result(result: dict[str, Any]) -> None:
    st.error(
        "🚨 FAILURE DETECTED",
        icon="🚨",
    )

    failure_mode = result.get("failure_mode", "Unknown")

    st.markdown(
        f"""
        ### Machine Failure Warning

        **Failure Detection:** Class 1

        **Failure Mode:** `{failure_mode}`
        """
    )

    st.warning(
        "Immediate machine inspection is recommended.",
        icon="⚠️",
    )


# ============================================================
# Main application
# ============================================================

def main() -> None:

    render_header()

    try:
        task_a_model, task_b_model = load_models()
    except FileNotFoundError as exc:
        logger.exception("Model files are missing.")
        st.error("Model artifacts are missing.")
        st.code(str(exc))
        st.stop()

    except Exception as exc:
        logger.exception("Model loading failed.")
        st.error(
            "The ML models could not be loaded. "
            "Check artifact compatibility and application logs."
        )
        st.exception(exc)
        st.stop()

    render_model_info(task_a_model, task_b_model)

    feature_names = get_model_feature_names(task_a_model)

    input_df = render_input_form(feature_names)

    if input_df is None:
        return

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    is_valid, validation_error = validate_input_dataframe(
        input_df,
        feature_names,
    )

    if not is_valid:
        st.error(f"Invalid input: {validation_error}")
        logger.warning(
            "Input validation failed: %s",
            validation_error,
        )
        return

    # --------------------------------------------------------
    # Sequential inference
    # --------------------------------------------------------

    with st.spinner("Running prediction..."):
        try:
            result = run_inference(
                input_df=input_df,
                task_a_model=task_a_model,
                task_b_model=task_b_model,
            )

        except Exception as exc:
            logger.exception("Inference failed.")
            st.error(
                "Prediction failed. Verify that the supplied feature "
                "values and model versions match the training pipeline."
            )
            st.exception(exc)
            return

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.divider()
    st.subheader("Prediction Result")

    if result["failure_detected"]:
        render_failure_result(result)
    else:
        render_normal_result()

    # --------------------------------------------------------
    # Debug information for development
    # --------------------------------------------------------

    with st.expander("Prediction details"):
        st.json(
            {
                "task_a_prediction": result.get("task_a_prediction"),
                "task_b_prediction": result.get("task_b_prediction"),
                "failure_mode": result.get("failure_mode"),
            }
        )


if __name__ == "__main__":
    main()