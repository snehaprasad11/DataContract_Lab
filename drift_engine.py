import pandas as pd
from scipy import stats


def profile_dataframe(df):
    profiles = []
    for column in df.columns:
        series = df[column]
        profile = {
            "column": column,
            "dtype": str(series.dtype),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique_count": series.nunique(),
            "sample_values": series.dropna().unique()[:3].tolist(),
        }

        if pd.api.types.is_numeric_dtype(series):
            profile["min"] = round(series.min(), 2)
            profile["max"] = round(series.max(), 2)
            profile["mean"] = round(series.mean(), 2)
            profile["median"] = round(series.median(), 2)
            profile["std"] = round(series.std(), 2)
            profile["top_categories"] = None
        else:
            profile["min"] = None
            profile["max"] = None
            profile["mean"] = None
            profile["median"] = None
            profile["std"] = None
            profile["top_categories"] = series.value_counts().head(3).to_dict()

        profiles.append(profile)

    return pd.DataFrame(profiles).set_index("column")


def format_profile_for_display(profile_df):
    display_df = profile_df.copy()
    display_df["sample_values"] = display_df["sample_values"].apply(
        lambda values: ", ".join(str(v) for v in values)
    )
    display_df["top_categories"] = display_df["top_categories"].apply(
        lambda categories: ", ".join(f"{k}: {v}" for k, v in categories.items()) if categories else "—"
    )
    display_df["null_pct"] = display_df["null_pct"].apply(lambda x: f"{x:.1f}%")
    for col in ["min", "max", "mean", "median", "std"]:
        display_df[col] = display_df[col].apply(lambda x: "—" if pd.isna(x) else f"{x:g}")
    return display_df


def compare_schemas(baseline_profile, new_profile):
    baseline_cols = set(baseline_profile.index)
    new_cols = set(new_profile.index)

    added = sorted(new_cols - baseline_cols)
    removed = sorted(baseline_cols - new_cols)
    common = sorted(baseline_cols & new_cols)

    dtype_changes = []
    for col in common:
        old_dtype = baseline_profile.loc[col, "dtype"]
        new_dtype = new_profile.loc[col, "dtype"]
        if old_dtype != new_dtype:
            dtype_changes.append({"column": col, "old_dtype": old_dtype, "new_dtype": new_dtype})

    possible_renames = []
    for removed_col in removed:
        removed_values = set(baseline_profile.loc[removed_col, "sample_values"])
        for added_col in added:
            added_values = set(new_profile.loc[added_col, "sample_values"])
            if removed_values and removed_values & added_values:
                possible_renames.append({"old_name": removed_col, "new_name": added_col})

    return {
        "added": added,
        "removed": removed,
        "dtype_changes": dtype_changes,
        "possible_renames": possible_renames,
    }


def detect_missing_value_drift(baseline_profile, new_profile, threshold=10):
    common = sorted(set(baseline_profile.index) & set(new_profile.index))
    drifted = []
    for col in common:
        old_pct = baseline_profile.loc[col, "null_pct"]
        new_pct = new_profile.loc[col, "null_pct"]
        change = new_pct - old_pct
        if abs(change) >= threshold:
            drifted.append({
                "column": col,
                "baseline_null_pct": old_pct,
                "new_null_pct": new_pct,
                "change": round(change, 2),
            })
    return drifted


def detect_categorical_drift(baseline_df, new_df, alpha=0.05, min_observations=5, max_categories=50):
    common_cols = sorted(set(baseline_df.columns) & set(new_df.columns))
    drifted = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(baseline_df[col]) or pd.api.types.is_numeric_dtype(new_df[col]):
            continue

        baseline_counts = baseline_df[col].dropna().value_counts()
        new_counts = new_df[col].dropna().value_counts()
        categories = sorted(set(baseline_counts.index) | set(new_counts.index))

        if len(categories) < 2 or len(categories) > max_categories:
            continue
        if baseline_counts.sum() < min_observations or new_counts.sum() < min_observations:
            continue

        observed = [
            [baseline_counts.get(cat, 0) for cat in categories],
            [new_counts.get(cat, 0) for cat in categories],
        ]

        try:
            statistic, p_value, _, _ = stats.chi2_contingency(observed)
        except ValueError:
            continue

        if p_value < alpha:
            drifted.append({
                "column": col,
                "chi2_statistic": round(statistic, 4),
                "p_value": round(p_value, 4),
                "baseline_top_value": baseline_counts.idxmax(),
                "new_top_value": new_counts.idxmax(),
            })
    return drifted


def detect_numeric_distribution_drift(baseline_df, new_df, alpha=0.05):
    common_cols = sorted(set(baseline_df.columns) & set(new_df.columns))
    drifted = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(baseline_df[col]) and pd.api.types.is_numeric_dtype(new_df[col]):
            baseline_values = baseline_df[col].dropna()
            new_values = new_df[col].dropna()
            if len(baseline_values) < 2 or len(new_values) < 2:
                continue
            statistic, p_value = stats.ks_2samp(baseline_values, new_values)
            if p_value < alpha:
                drifted.append({
                    "column": col,
                    "ks_statistic": round(statistic, 4),
                    "p_value": round(p_value, 4),
                })
    return drifted


def compute_quality_score(schema_diff, missing_drift, categorical_drift, numeric_drift):
    score = 100

    renamed_old_names = {rename["old_name"] for rename in schema_diff["possible_renames"]}
    unexplained_removed = [col for col in schema_diff["removed"] if col not in renamed_old_names]

    score -= len(unexplained_removed) * 10
    score -= len(schema_diff["dtype_changes"]) * 5
    score -= len(missing_drift) * 5
    score -= len(categorical_drift) * 3
    score -= len(numeric_drift) * 5

    return max(0, min(100, score))


def generate_summary(schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score):
    sentences = []

    renamed_old_names = {r["old_name"] for r in schema_diff["possible_renames"]}
    unexplained_removed = [c for c in schema_diff["removed"] if c not in renamed_old_names]

    if schema_diff["possible_renames"]:
        renames_text = "; ".join(f"'{r['old_name']}' → '{r['new_name']}'" for r in schema_diff["possible_renames"])
        sentences.append(f"{len(schema_diff['possible_renames'])} column(s) appear to have been renamed: {renames_text}.")

    if schema_diff["added"]:
        sentences.append(f"{len(schema_diff['added'])} new column(s) were added: {', '.join(schema_diff['added'])}.")

    if unexplained_removed:
        sentences.append(f"{len(unexplained_removed)} column(s) were removed with no clear replacement: {', '.join(unexplained_removed)}.")

    if schema_diff["dtype_changes"]:
        dtype_text = ", ".join(f"{d['column']} ({d['old_dtype']} → {d['new_dtype']})" for d in schema_diff["dtype_changes"])
        sentences.append(f"{len(schema_diff['dtype_changes'])} column(s) changed data type: {dtype_text}.")

    if missing_drift:
        missing_text = ", ".join(f"{d['column']} ({d['baseline_null_pct']}% → {d['new_null_pct']}%)" for d in missing_drift)
        sentences.append(f"Missing values increased notably in: {missing_text}.")

    if categorical_drift:
        cat_text = ", ".join(f"{d['column']} (p={d['p_value']})" for d in categorical_drift)
        sentences.append(f"Categorical distribution shifted significantly in: {cat_text}.")

    if numeric_drift:
        num_text = ", ".join(f"{d['column']} (p={d['p_value']})" for d in numeric_drift)
        sentences.append(f"Numeric distribution shifted significantly in: {num_text}.")

    if not sentences:
        sentences.append("No meaningful differences were detected between the two files.")

    sentences.append(f"Overall data quality score: {quality_score}/100.")

    return " ".join(sentences)


def build_suggestions(schema_diff, missing_drift, categorical_drift, numeric_drift):
    suggestions = []
    renamed_old_names = {r["old_name"] for r in schema_diff["possible_renames"]}
    unexplained_removed = [c for c in schema_diff["removed"] if c not in renamed_old_names]

    if unexplained_removed:
        suggestions.append(
            f"Column(s) {', '.join(unexplained_removed)} disappeared with no clear replacement. "
            "Confirm this is intentional before downstream consumers break."
        )
    if schema_diff["added"]:
        suggestions.append(
            f"New column(s) {', '.join(schema_diff['added'])} appeared. "
            "Make sure any downstream schema or contract is updated to expect them."
        )
    if schema_diff["dtype_changes"]:
        cols = ", ".join(d["column"] for d in schema_diff["dtype_changes"])
        suggestions.append(f"Data type changed for: {cols}. Check that downstream code doesn't assume the old type.")
    if missing_drift:
        cols = ", ".join(d["column"] for d in missing_drift)
        suggestions.append(
            f"Missing values increased notably in: {cols}. Review the upstream source and consider a "
            "null-handling strategy before this data is consumed further."
        )
    if categorical_drift:
        cols = ", ".join(d["column"] for d in categorical_drift)
        suggestions.append(f"The distribution of values shifted significantly in: {cols}. Confirm this reflects a real business change, not a labeling or encoding bug.")
    if numeric_drift:
        cols = ", ".join(d["column"] for d in numeric_drift)
        suggestions.append(
            f"Numeric distribution shifted significantly in: {cols}. Validate whether this is an expected "
            "change (e.g. a pricing update) or a pipeline bug."
        )

    return suggestions
