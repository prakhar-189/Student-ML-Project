"""Unit tests for CustomData -> DataFrame conversion (no model needed)."""
from src.pipeline.predict_pipeline import CustomData

EXPECTED_COLUMNS = [
    "gender", "race_ethnicity", "parental_level_of_education", "lunch",
    "test_preparation_course", "writing_score", "reading_score",
]


def _sample():
    return CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        writing_score=72,
        reading_score=70,
    )


def test_dataframe_has_single_row_and_expected_columns():
    df = _sample().get_data_as_dataframe()
    assert df.shape[0] == 1
    assert list(df.columns) == EXPECTED_COLUMNS


def test_dataframe_preserves_values():
    df = _sample().get_data_as_dataframe()
    assert df.loc[0, "gender"] == "female"
    assert df.loc[0, "reading_score"] == 70
