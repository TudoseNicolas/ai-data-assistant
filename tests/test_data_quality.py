import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_quality as dq  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("1.099,00 lei", 1099.0),
    ("1,099.00", 1099.0),
    ("89,50", 89.5),
    ("12,480", 12480.0),
    ("1.234.567", 1234567.0),
    ("€ 25", 25.0),
    ("-3.5", -3.5),
    (42, 42.0),
])
def test_parse_number_reads_common_formats(raw, expected):
    assert dq.parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["Monitor 27\"", "07.02.2020", "2025-03-02", "abc", "", None])
def test_parse_number_rejects_text_and_dates(raw):
    assert dq.parse_number(raw) is None


def test_normalize_text_ignores_case_spaces_and_diacritics():
    assert dq.normalize_text("  BUCUREȘTI ") == dq.normalize_text("bucuresti")


@pytest.fixture
def messy():
    return pd.DataFrame({
        "City": ["Bucharest", "bucharest ", "Cluj", "Cluj", "Bucharest", "Iasi", "Iasi", "Bucharest"],
        "Price": ["1.099,00 lei", "89,00 lei", "149,00 lei", "149,00 lei", "89,00 lei", "10,00 lei", "20,00 lei", "30,00 lei"],
        "Category": ["A", None, "B", "B", "A", "C", " ", "A"],
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"],
    })


def test_analyze_finds_every_kind_of_problem(messy):
    kinds = {issue.kind for issue in dq.analyze(messy)}
    assert kinds == {"duplicates", "numeric_text", "missing", "date_text", "inconsistent_text"}


def test_blank_strings_count_as_missing(messy):
    missing = next(i for i in dq.analyze(messy) if i.id == "missing:Category")
    assert missing.count == 2


def test_clean_fixes_selected_problems(messy):
    issues = dq.analyze(messy)
    cleaned, log = dq.clean(messy, issues, [i.id for i in issues])
    assert cleaned.duplicated().sum() == 0
    assert pd.api.types.is_numeric_dtype(cleaned["Price"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
    assert cleaned["Category"].isna().sum() == 0
    assert set(cleaned["City"]) == {"Bucharest", "Cluj", "Iasi"}
    assert dq.analyze(cleaned) == []
    assert len(log) == 5


def test_clean_leaves_unselected_problems(messy):
    issues = dq.analyze(messy)
    cleaned, _ = dq.clean(messy, issues, ["duplicates"])
    assert not pd.api.types.is_numeric_dtype(cleaned["Price"])


def test_clean_does_not_change_the_original(messy):
    before = messy.copy()
    issues = dq.analyze(messy)
    dq.clean(messy, issues, [i.id for i in issues])
    pd.testing.assert_frame_equal(messy, before)


def test_quality_score_improves_after_cleaning(messy):
    issues = dq.analyze(messy)
    cleaned, _ = dq.clean(messy, issues, [i.id for i in issues])
    assert dq.quality_score(cleaned) == 100 > dq.quality_score(messy)


def test_free_text_column_is_not_flagged_as_inconsistent():
    df = pd.DataFrame({"Comment": [f"note number {i}" for i in range(20)]})
    assert dq.analyze(df) == []
