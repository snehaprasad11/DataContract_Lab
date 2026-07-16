import pandas as pd
import pytest

from drift_engine import (
    build_suggestions,
    compare_schemas,
    compute_quality_score,
    detect_categorical_drift,
    detect_missing_value_drift,
    detect_numeric_distribution_drift,
    format_profile_for_display,
    generate_summary,
    profile_dataframe,
)


def make_profile(rows):
    return pd.DataFrame(rows).set_index("column")


@pytest.fixture
def baseline_df():
    return pd.DataFrame({
        "id": range(1, 11),
        "status": ["active"] * 7 + ["inactive"] * 3,
        "amount": [10, 12, 11, 13, 10, 12, 11, 13, 12, 11],
        "region": ["west", "west", "east", "east", "west", "west", "east", "east", "west", "west"],
    })


@pytest.fixture
def new_df():
    return pd.DataFrame({
        "id": range(1, 11),
        "status": ["inactive"] * 7 + ["active"] * 3,
        "amount": [110, 112, 111, 113, 110, 112, 111, 113, 112, 111],
        "extra_col": range(1, 11),
    })


# --- profile_dataframe ---

def test_profile_dataframe_numeric_column_gets_stats():
    df = pd.DataFrame({"amount": [10, 20, 30, 40, None]})
    profile = profile_dataframe(df)
    assert profile.loc["amount", "null_pct"] == 20.0
    assert profile.loc["amount", "min"] == 10
    assert profile.loc["amount", "max"] == 40
    assert profile.loc["amount", "top_categories"] is None


def test_profile_dataframe_categorical_column_gets_top_categories():
    df = pd.DataFrame({"status": ["a", "a", "a", "b"]})
    profile = profile_dataframe(df)
    assert profile.loc["status", "mean"] is None
    assert profile.loc["status", "top_categories"] == {"a": 3, "b": 1}


def test_format_profile_for_display_does_not_crash_on_mixed_types(baseline_df):
    profile = profile_dataframe(baseline_df)
    display = format_profile_for_display(profile)
    assert display.loc["amount", "null_pct"] == "0.0%"
    assert display.loc["status", "min"] == "—"


# --- compare_schemas ---

def test_compare_schemas_detects_added_and_removed_columns():
    baseline = make_profile([
        {"column": "a", "dtype": "int64", "sample_values": [1, 2, 3]},
        {"column": "b", "dtype": "int64", "sample_values": [4, 5, 6]},
    ])
    new = make_profile([
        {"column": "a", "dtype": "int64", "sample_values": [1, 2, 3]},
        {"column": "c", "dtype": "int64", "sample_values": [7, 8, 9]},
    ])
    diff = compare_schemas(baseline, new)
    assert diff["added"] == ["c"]
    assert diff["removed"] == ["b"]
    assert diff["dtype_changes"] == []


def test_compare_schemas_detects_dtype_change():
    baseline = make_profile([{"column": "a", "dtype": "int64", "sample_values": [1, 2]}])
    new = make_profile([{"column": "a", "dtype": "object", "sample_values": ["1", "2"]}])
    diff = compare_schemas(baseline, new)
    assert diff["dtype_changes"] == [{"column": "a", "old_dtype": "int64", "new_dtype": "object"}]


def test_compare_schemas_flags_likely_rename_when_values_overlap():
    baseline = make_profile([{"column": "customer_id", "dtype": "int64", "sample_values": [101, 102, 103]}])
    new = make_profile([{"column": "cust_id", "dtype": "int64", "sample_values": [101, 999, 998]}])
    diff = compare_schemas(baseline, new)
    assert diff["possible_renames"] == [{"old_name": "customer_id", "new_name": "cust_id"}]
    assert diff["removed"] == ["customer_id"]
    assert diff["added"] == ["cust_id"]


def test_compare_schemas_no_rename_when_values_dont_overlap():
    baseline = make_profile([{"column": "region", "dtype": "object", "sample_values": ["west", "east"]}])
    new = make_profile([{"column": "extra_col", "dtype": "int64", "sample_values": [1, 2, 3]}])
    diff = compare_schemas(baseline, new)
    assert diff["possible_renames"] == []


# --- detect_missing_value_drift ---

def test_detect_missing_value_drift_flags_increase_past_threshold():
    baseline = profile_dataframe(pd.DataFrame({"col": range(10)}))
    new = profile_dataframe(pd.DataFrame({"col": [1, 2, 3] + [None] * 7}))
    drifted = detect_missing_value_drift(baseline, new)
    assert len(drifted) == 1
    assert drifted[0]["column"] == "col"
    assert drifted[0]["change"] == 70.0


def test_detect_missing_value_drift_ignores_small_changes():
    baseline = profile_dataframe(pd.DataFrame({"col": range(10)}))
    new = profile_dataframe(pd.DataFrame({"col": range(10)}))
    assert detect_missing_value_drift(baseline, new) == []


# --- detect_categorical_drift ---

def test_detect_categorical_drift_flags_top_value_change(baseline_df, new_df):
    baseline_profile = profile_dataframe(baseline_df)
    new_profile = profile_dataframe(new_df)
    drifted = detect_categorical_drift(baseline_profile, new_profile)
    assert drifted == [{"column": "status", "baseline_top_value": "active", "new_top_value": "inactive"}]


def test_detect_categorical_drift_no_flag_when_top_value_same(baseline_df):
    profile = profile_dataframe(baseline_df)
    assert detect_categorical_drift(profile, profile) == []


# --- detect_numeric_distribution_drift ---

def test_detect_numeric_distribution_drift_flags_clear_shift(baseline_df, new_df):
    drifted = detect_numeric_distribution_drift(baseline_df, new_df)
    columns_flagged = [d["column"] for d in drifted]
    assert "amount" in columns_flagged
    amount_result = next(d for d in drifted if d["column"] == "amount")
    assert amount_result["p_value"] < 0.05


def test_detect_numeric_distribution_drift_no_flag_on_identical_distributions(baseline_df):
    assert detect_numeric_distribution_drift(baseline_df, baseline_df.copy()) == []


def test_detect_numeric_distribution_drift_skips_tiny_samples():
    baseline = pd.DataFrame({"amount": [1]})
    new = pd.DataFrame({"amount": [100]})
    assert detect_numeric_distribution_drift(baseline, new) == []


# --- compute_quality_score ---

def test_compute_quality_score_applies_expected_deductions():
    schema_diff = {"added": [], "removed": ["a", "b"], "dtype_changes": [{"column": "x"}], "possible_renames": []}
    missing_drift = [{"column": "y"}]
    categorical_drift = [{"column": "z"}, {"column": "w"}]
    numeric_drift = []
    score = compute_quality_score(schema_diff, missing_drift, categorical_drift, numeric_drift)
    assert score == 100 - (2 * 10) - (1 * 5) - (1 * 5) - (2 * 3)


def test_compute_quality_score_excludes_renamed_columns_from_penalty():
    schema_diff = {
        "added": ["b"], "removed": ["a"], "dtype_changes": [],
        "possible_renames": [{"old_name": "a", "new_name": "b"}],
    }
    score = compute_quality_score(schema_diff, [], [], [])
    assert score == 100


def test_compute_quality_score_never_goes_below_zero():
    schema_diff = {"added": [], "removed": [f"col{i}" for i in range(20)], "dtype_changes": [], "possible_renames": []}
    score = compute_quality_score(schema_diff, [], [], [])
    assert score == 0


# --- generate_summary ---

def test_generate_summary_reports_no_differences_when_clean():
    schema_diff = {"added": [], "removed": [], "dtype_changes": [], "possible_renames": []}
    summary = generate_summary(schema_diff, [], [], [], 100)
    assert "No meaningful differences" in summary
    assert "100/100" in summary


def test_generate_summary_mentions_drifted_columns():
    schema_diff = {"added": ["new_col"], "removed": [], "dtype_changes": [], "possible_renames": []}
    summary = generate_summary(schema_diff, [], [], [], 90)
    assert "new_col" in summary


# --- build_suggestions ---

def test_build_suggestions_empty_when_no_drift():
    schema_diff = {"added": [], "removed": [], "dtype_changes": [], "possible_renames": []}
    assert build_suggestions(schema_diff, [], [], []) == []


def test_build_suggestions_flags_unexplained_removed_columns():
    schema_diff = {"added": [], "removed": ["legacy_col"], "dtype_changes": [], "possible_renames": []}
    suggestions = build_suggestions(schema_diff, [], [], [])
    assert any("legacy_col" in s for s in suggestions)
