import pandas as pd

import app


class DummyOneVsRestModel:
    def predict_proba(self, input_df):
        return [[0.05, 0.90, 0.60, 0.15, 0.10]]


def test_validate_input_dataframe_supports_categorical_feature_columns():
    input_df = pd.DataFrame(
        [{"Type": "H", "Air temperature [K]": 300.0}]
    )

    is_valid, error = app.validate_input_dataframe(
        input_df,
        ["Type", "Air temperature [K]"],
        {"Type": ["H", "L", "M"]},
    )

    assert is_valid is True
    assert error == ""


def test_get_predicted_failure_modes_from_task_b_returns_ai4i_labels():
    labels = app.get_predicted_failure_modes_from_task_b(
        DummyOneVsRestModel(),
        pd.DataFrame({"Type": ["H"]}),
        threshold=0.5,
    )

    assert labels == ["HDF", "PWF"]
