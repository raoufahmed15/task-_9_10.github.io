import pandas as pd

import app


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
