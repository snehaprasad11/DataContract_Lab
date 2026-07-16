import io
import os
from urllib.parse import quote_plus

import certifi
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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

load_dotenv()

st.set_page_config(page_title="DataContract Lab", page_icon="🔬", layout="wide")

HACKER_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #05070a 0%, #0b0f10 55%, #060a08 100%) !important;
}
[data-testid="stMarkdownContainer"] h1 {
    text-shadow: 0 0 10px rgba(57, 255, 122, 0.6);
    letter-spacing: 1px;
}
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    text-shadow: 0 0 6px rgba(57, 255, 122, 0.4);
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(57, 255, 122, 0.5) !important;
    border-radius: 6px;
}
[data-testid="stAlert"] {
    border-left: 4px solid #39ff7a;
}
table, table td, table th {
    color: #39ff7a !important;
    font-family: "Courier New", Consolas, monospace !important;
    border-color: rgba(57, 255, 122, 0.25) !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(57, 255, 122, 0.25);
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background-color: #ff3b3b !important;
    border-color: #ff3b3b !important;
}
button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background-color: #ff6161 !important;
    border-color: #ff6161 !important;
}
.typewriter-title {
    font-family: "Courier New", Consolas, monospace;
    color: #39ff7a;
    text-shadow: 0 0 10px rgba(57, 255, 122, 0.6);
    font-size: 2.4rem;
    font-weight: 700;
    display: inline-block;
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid #39ff7a;
    width: 22ch;
    animation: typing 4s steps(22, end) infinite, blink-caret 0.75s step-end infinite;
    margin-bottom: 0.5rem;
}
@keyframes typing {
    0% { width: 0ch; }
    60% { width: 22ch; }
    100% { width: 22ch; }
}
@keyframes blink-caret {
    from, to { border-color: transparent; }
    50% { border-color: #39ff7a; }
}
</style>
"""
st.markdown(HACKER_CSS, unsafe_allow_html=True)
st.markdown('<div class="typewriter-title">🔬 DataContract Lab</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload your **baseline** file (the old/known-good version).\n"
        "2. Upload your **new** file (the version you want to check).\n"
        "3. Click **Run Comparison**.\n\n"
        "Supports `.csv`, `.xlsx`, and `.json` — you can even mix formats between the two files."
    )

st.write("Compare two versions of a dataset and catch schema drift, missing data, and distribution shifts — before they break something downstream.")


def load_dataframe(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    elif filename.endswith(".json"):
        return pd.read_json(uploaded_file)
    else:
        st.error(f"Unsupported file type: {filename}")
        return None

def plot_null_pct_comparison(baseline_profile, new_profile):
    common_cols = sorted(set(baseline_profile.index) & set(new_profile.index))
    chart_df = pd.DataFrame({
        "column": common_cols * 2,
        "file": ["baseline"] * len(common_cols) + ["new"] * len(common_cols),
        "null_pct": (
            [baseline_profile.loc[c, "null_pct"] for c in common_cols]
            + [new_profile.loc[c, "null_pct"] for c in common_cols]
        ),
    })
    fig = px.bar(
        chart_df, x="column", y="null_pct", color="file", barmode="group",
        title="Missing values (%) by column", color_discrete_map={"baseline": "#39ff7a", "new": "#ff3b3b"},
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#05070a", plot_bgcolor="#05070a", font_color="#39ff7a",
    )
    return fig


def plot_distribution_overlay(baseline_df, new_df, column):
    hist_df = pd.concat([
        pd.DataFrame({"value": baseline_df[column].dropna(), "file": "baseline"}),
        pd.DataFrame({"value": new_df[column].dropna(), "file": "new"}),
    ])
    fig = px.histogram(
        hist_df, x="value", color="file", barmode="overlay", opacity=0.6,
        title=f"Distribution of '{column}'", color_discrete_map={"baseline": "#39ff7a", "new": "#ff3b3b"},
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#05070a", plot_bgcolor="#05070a", font_color="#39ff7a",
    )
    return fig


def build_markdown_report(schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score, summary_text):
    lines = ["# DataContract Lab Report", "", f"**Quality score:** {quality_score}/100", "", "## Summary", summary_text, ""]

    lines.append("## Schema comparison")
    lines.append(f"- Added columns: {', '.join(schema_diff['added']) or 'none'}")
    lines.append(f"- Removed columns: {', '.join(schema_diff['removed']) or 'none'}")
    if schema_diff["possible_renames"]:
        renames = "; ".join(f"{r['old_name']} → {r['new_name']}" for r in schema_diff["possible_renames"])
        lines.append(f"- Possible renames: {renames}")
    if schema_diff["dtype_changes"]:
        dtypes = "; ".join(f"{d['column']} ({d['old_dtype']} → {d['new_dtype']})" for d in schema_diff["dtype_changes"])
        lines.append(f"- Data type changes: {dtypes}")
    lines.append("")

    lines.append("## Missing-value drift")
    if missing_drift:
        for d in missing_drift:
            lines.append(f"- {d['column']}: {d['baseline_null_pct']}% → {d['new_null_pct']}%")
    else:
        lines.append("None detected.")
    lines.append("")

    lines.append("## Categorical drift")
    if categorical_drift:
        for d in categorical_drift:
            lines.append(
                f"- {d['column']}: chi-square p-value={d['p_value']} "
                f"(top value: '{d['baseline_top_value']}' → '{d['new_top_value']}')"
            )
    else:
        lines.append("None detected.")
    lines.append("")

    lines.append("## Numeric distribution drift")
    if numeric_drift:
        for d in numeric_drift:
            lines.append(f"- {d['column']}: KS statistic={d['ks_statistic']}, p-value={d['p_value']}")
    else:
        lines.append("None detected.")

    return "\n".join(lines)


BRAND_GREEN = colors.HexColor("#1f8a4c")
BRAND_DARK = colors.HexColor("#0b1f14")
FLAG_RED = colors.HexColor("#c62828")
FLAG_AMBER = colors.HexColor("#b8860b")


def _pdf_safe(text):
    return str(text).replace("→", "->")


def _draw_letterhead(canvas_obj, doc):
    canvas_obj.saveState()
    width, height = LETTER

    canvas_obj.setFillColor(BRAND_DARK)
    canvas_obj.rect(0, height - 0.65 * inch, width, 0.65 * inch, fill=True, stroke=False)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Courier-Bold", 15)
    canvas_obj.drawString(0.5 * inch, height - 0.43 * inch, "DataContract Lab")
    canvas_obj.setFont("Courier", 9)
    canvas_obj.drawRightString(width - 0.5 * inch, height - 0.43 * inch, "Data Quality Diagnostic Report")

    canvas_obj.setFillColor(colors.grey)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(0.5 * inch, 0.4 * inch, "DataContract Lab — automated diagnostic, not a substitute for human review")
    canvas_obj.drawRightString(width - 0.5 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas_obj.restoreState()


def _styled_table(rows):
    table = Table(rows, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]
    for row_idx in range(1, len(rows)):
        if rows[row_idx][-1] == "ABNORMAL":
            style.append(("TEXTCOLOR", (-1, row_idx), (-1, row_idx), FLAG_RED))
            style.append(("FONTNAME", (-1, row_idx), (-1, row_idx), "Helvetica-Bold"))
        elif rows[row_idx][-1] == "INFO":
            style.append(("TEXTCOLOR", (-1, row_idx), (-1, row_idx), FLAG_AMBER))
    table.setStyle(TableStyle(style))
    return table


def build_pdf_report(schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score, summary_text, baseline_name, new_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=1.0 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], textColor=BRAND_DARK)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], textColor=BRAND_GREEN, spaceBefore=14)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=14)
    suggestion_style = ParagraphStyle("Suggestion", parent=styles["BodyText"], leftIndent=12, leading=14)

    if quality_score >= 90:
        score_color = BRAND_GREEN
    elif quality_score >= 70:
        score_color = FLAG_AMBER
    else:
        score_color = FLAG_RED

    story = [
        Paragraph("Data Quality Diagnostic Report", title_style),
        Paragraph(f"Baseline file: <b>{baseline_name}</b> &nbsp;&nbsp;&nbsp; New file: <b>{new_name}</b>", body_style),
        Spacer(1, 10),
    ]

    score_table = Table([["Overall Quality Score", f"{quality_score} / 100"]], colWidths=[3 * inch, 3 * inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), score_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 13),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Overall Assessment", section_style))
    story.append(Paragraph(_pdf_safe(summary_text), body_style))

    story.append(PageBreak())

    story.append(Paragraph("Diagnostic Panels", title_style))
    story.append(Paragraph(
        "Each panel below lists what was checked, the result found, and the expected/normal range — read the same way you'd read a lab test panel.",
        body_style,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Schema Panel", section_style))
    renamed_old_names = {r["old_name"] for r in schema_diff["possible_renames"]}
    unexplained_removed = [c for c in schema_diff["removed"] if c not in renamed_old_names]
    schema_rows = [["Test", "Result", "Normal Range", "Flag"]]
    schema_rows.append(["Added columns", ", ".join(schema_diff["added"]) or "none", "0", "ABNORMAL" if schema_diff["added"] else "NORMAL"])
    schema_rows.append(["Removed columns (unexplained)", ", ".join(unexplained_removed) or "none", "0", "ABNORMAL" if unexplained_removed else "NORMAL"])
    renames_text = "; ".join(f"{r['old_name']} -> {r['new_name']}" for r in schema_diff["possible_renames"])
    schema_rows.append(["Renamed columns", renames_text or "none", "—", "INFO" if schema_diff["possible_renames"] else "NORMAL"])
    dtype_text = ", ".join(d["column"] for d in schema_diff["dtype_changes"])
    schema_rows.append(["Data type changes", dtype_text or "none", "0", "ABNORMAL" if schema_diff["dtype_changes"] else "NORMAL"])
    story.append(_styled_table(schema_rows))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Missing-Value Panel", section_style))
    if missing_drift:
        rows = [["Column", "Baseline", "New", "Flag"]]
        for d in missing_drift:
            rows.append([d["column"], f"{d['baseline_null_pct']}%", f"{d['new_null_pct']}%", "ABNORMAL"])
        story.append(_styled_table(rows))
    else:
        story.append(Paragraph("No significant missing-value drift detected.", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Categorical Drift Panel", section_style))
    if categorical_drift:
        rows = [["Column", "Baseline top value", "New top value", "p-value", "Flag"]]
        for d in categorical_drift:
            rows.append([d["column"], str(d["baseline_top_value"]), str(d["new_top_value"]), str(d["p_value"]), "ABNORMAL"])
        story.append(_styled_table(rows))
    else:
        story.append(Paragraph("No significant categorical drift detected.", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Numeric Distribution Panel", section_style))
    if numeric_drift:
        rows = [["Column", "KS statistic", "p-value", "Flag"]]
        for d in numeric_drift:
            rows.append([d["column"], str(d["ks_statistic"]), str(d["p_value"]), "ABNORMAL"])
        story.append(_styled_table(rows))
    else:
        story.append(Paragraph("No significant numeric distribution drift detected.", body_style))

    story.append(PageBreak())

    story.append(Paragraph("Suggestions", title_style))
    suggestions = build_suggestions(schema_diff, missing_drift, categorical_drift, numeric_drift)
    if suggestions:
        for i, suggestion in enumerate(suggestions, start=1):
            story.append(Paragraph(f"{i}. {suggestion}", suggestion_style))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No action needed — this dataset looks consistent with the baseline.", body_style))

    doc.build(story, onFirstPage=_draw_letterhead, onLaterPages=_draw_letterhead)
    buffer.seek(0)
    return buffer


def generate_ollama_explanation(summary_text, model="llama3.2"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": (
                    "You are explaining a data quality report to a non-technical teammate. "
                    "Rewrite the following findings as 2-3 short, friendly sentences:\n\n" + summary_text
                ),
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["response"], None
    except requests.exceptions.ConnectionError:
        return None, "Couldn't reach Ollama — make sure it's installed and running (`ollama serve`), with a model pulled (e.g. `ollama pull llama3`), then try again."
    except Exception as e:
        return None, f"Ollama request failed: {e}"


def get_setting(key, default=None):
    """Read config with a clear priority: Streamlit Cloud secrets (st.secrets) win,
    then local environment variables (from .env), then the default. This lets the same
    code run locally off a .env file and in the cloud off Streamlit's Secrets box."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # st.secrets raises if there is no secrets file at all (normal when running locally)
        pass
    return os.environ.get(key, default)


def _db_url(db_user, db_password, db_host, db_port, db_name=""):
    # URL-encode the credentials so passwords containing @ : / etc. (common with
    # auto-generated cloud passwords) don't corrupt the connection string.
    return (
        f"mysql+pymysql://{quote_plus(db_user)}:{quote_plus(db_password)}"
        f"@{db_host}:{db_port}/{db_name}"
    )


@st.cache_resource
def get_engine():
    db_host = get_setting("DB_HOST", "localhost")
    db_port = get_setting("DB_PORT", "3306")
    db_user = get_setting("DB_USER", "root")
    db_password = get_setting("DB_PASSWORD", "")
    db_name = get_setting("DB_NAME", "datacontract_lab")

    # Hosted MySQL providers (e.g. TiDB Cloud) require TLS. Set DB_SSL=true to enable it;
    # the CA bundle from certifi covers their public certificate chains.
    use_ssl = str(get_setting("DB_SSL", "false")).strip().lower() in ("1", "true", "yes")
    connect_args = {"ssl": {"ca": certifi.where()}} if use_ssl else {}

    # Auto-create the database for local setups. On hosted providers the database
    # already exists and the account may not be allowed to create one, so a failure
    # here is non-fatal — we fall through to connecting to the existing database.
    try:
        server_engine = create_engine(_db_url(db_user, db_password, db_host, db_port), connect_args=connect_args)
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            conn.commit()
    except Exception:
        pass

    engine = create_engine(_db_url(db_user, db_password, db_host, db_port, db_name), connect_args=connect_args)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                baseline_filename VARCHAR(255) NOT NULL,
                new_filename VARCHAR(255) NOT NULL,
                quality_score INT NOT NULL,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

    return engine


def save_scan(engine, baseline_filename, new_filename, quality_score, summary_text):
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO scans (baseline_filename, new_filename, quality_score, summary)
                VALUES (:baseline_filename, :new_filename, :quality_score, :summary)
            """),
            {
                "baseline_filename": baseline_filename,
                "new_filename": new_filename,
                "quality_score": quality_score,
                "summary": summary_text,
            },
        )
        conn.commit()


def load_scan_history(engine, limit=20):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT baseline_filename, new_filename, quality_score, created_at
                FROM scans ORDER BY created_at DESC LIMIT :limit
            """),
            {"limit": limit},
        )
        rows = result.fetchall()
    return pd.DataFrame(rows, columns=["baseline_filename", "new_filename", "quality_score", "created_at"])


engine = get_engine()

# Section headers are numbered dynamically (rather than hard-coded) so that hiding an
# optional section — e.g. the Ollama panel when ENABLE_OLLAMA=false — doesn't leave a
# visible gap like "8 ... 10". The counter resets on every rerun since the whole script
# re-executes top to bottom.
_section_n = 0


def numbered(title):
    global _section_n
    _section_n += 1
    return f"{_section_n}. {title}"


st.header(numbered("Upload your datasets"))
col1, col2 = st.columns(2)

with col1:
    baseline_file = st.file_uploader(
        "📂 Baseline dataset",
        type=["csv", "xlsx", "json"],
        key="baseline",
        help="The original/older version you're comparing against.",
    )

with col2:
    new_file = st.file_uploader(
        "📁 New dataset",
        type=["csv", "xlsx", "json"],
        key="new",
        help="The newer version you want to check for drift.",
    )

st.divider()

if baseline_file and new_file:
    if st.button("Run Comparison", type="primary"):
        st.session_state.scan_saved = False
        progress = st.progress(0, text="Reading baseline file...")
        st.session_state.baseline_df = load_dataframe(baseline_file)

        progress.progress(40, text="Reading new file...")
        st.session_state.new_df = load_dataframe(new_file)

        if st.session_state.baseline_df is not None and st.session_state.new_df is not None:
            progress.progress(70, text="Profiling both datasets...")
            st.session_state.baseline_profile = profile_dataframe(st.session_state.baseline_df)
            st.session_state.new_profile = profile_dataframe(st.session_state.new_df)

        progress.progress(100, text="Done!")
        progress.empty()
else:
    # If either file is removed from the picker, drop any results from a previous
    # comparison so the panels below don't keep showing stale data (and so the PDF
    # export doesn't crash trying to read a now-missing file's name).
    for stale_key in ("baseline_df", "new_df", "baseline_profile", "new_profile", "scan_saved"):
        st.session_state.pop(stale_key, None)
    st.info("Upload both a baseline and a new file to enable the comparison.")

if "baseline_df" in st.session_state and "new_df" in st.session_state:
    baseline_df = st.session_state.baseline_df
    new_df = st.session_state.new_df

    if baseline_df is not None and new_df is not None:
        st.success("Both files loaded.")
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.subheader("Baseline preview")
            st.table(baseline_df.head())
        with preview_col2:
            st.subheader("New preview")
            st.table(new_df.head())
    if "baseline_profile" in st.session_state and "new_profile" in st.session_state:
        st.header(numbered("Column profiles"))
        st.caption("Think of this as each file's ID card: type, how much of it is missing in action, how many distinct values it holds, and its vital stats. No comparing yet — just getting acquainted.")
        st.subheader("Baseline profile")
        st.table(format_profile_for_display(st.session_state.baseline_profile))
        st.subheader("New profile")
        st.table(format_profile_for_display(st.session_state.new_profile))

        st.header(numbered("Schema comparison"))
        st.caption("The lineup: which columns showed up new, which ones skipped town, which ones just changed their name to dodge recognition, and which ones swapped their entire identity (data type).")
        schema_diff = compare_schemas(st.session_state.baseline_profile, st.session_state.new_profile)

        if schema_diff["possible_renames"]:
            for rename in schema_diff["possible_renames"]:
                st.warning(f"Possible rename: '{rename['old_name']}' → '{rename['new_name']}' (same values, different column name)")

        if schema_diff["added"]:
            st.info(f"Added columns: {', '.join(schema_diff['added'])}")

        if schema_diff["removed"]:
            st.error(f"Removed columns: {', '.join(schema_diff['removed'])}")

        if schema_diff["dtype_changes"]:
            st.warning("Data type changes detected:")
            st.table(pd.DataFrame(schema_diff["dtype_changes"]).set_index("column"))

        if not (schema_diff["added"] or schema_diff["removed"] or schema_diff["dtype_changes"]):
            st.success("No schema changes detected — column names and types match.")

        st.header(numbered("Missing-value & categorical drift"))
        st.caption("For columns that stuck around in both files: are they ghosting you more often now (missing values), or did their whole spread of answers shift, not just the most popular one?")
        missing_drift = detect_missing_value_drift(st.session_state.baseline_profile, st.session_state.new_profile)
        categorical_drift = detect_categorical_drift(st.session_state.baseline_df, st.session_state.new_df)

        if missing_drift:
            st.warning("Missing-value drift detected:")
            st.table(pd.DataFrame(missing_drift).set_index("column"))
        else:
            st.success("No significant missing-value drift detected.")

        if categorical_drift:
            st.warning("Statistically significant categorical drift detected (chi-square test, p < 0.05):")
            st.table(pd.DataFrame(categorical_drift).set_index("column"))
        else:
            st.success("No significant categorical drift detected.")

        st.plotly_chart(
            plot_null_pct_comparison(st.session_state.baseline_profile, st.session_state.new_profile),
            use_container_width=True,
        )

        st.header(numbered("Numeric distribution drift"))
        st.caption("Runs an actual statistics test (Kolmogorov–Smirnov — yes, that's really its name) on shared number columns, because averages can lie. This checks if the whole shape of the numbers moved, not just where they like to hang out on average.")
        numeric_drift = detect_numeric_distribution_drift(st.session_state.baseline_df, st.session_state.new_df)

        if numeric_drift:
            st.warning("Statistically significant distribution shifts detected (Kolmogorov–Smirnov test, p < 0.05):")
            st.table(pd.DataFrame(numeric_drift).set_index("column"))
            for drift in numeric_drift:
                st.plotly_chart(
                    plot_distribution_overlay(st.session_state.baseline_df, st.session_state.new_df, drift["column"]),
                    use_container_width=True,
                )
        else:
            st.success("No significant numeric distribution drift detected.")

        st.header(numbered("Data quality score"))
        st.caption("Every offense uncovered above — a column that vanished, one that snuck in wearing a disguise, missing data gone AWOL, numbers acting shifty — gets tallied into one report card out of 100. Lower score, dirtier data.")
        quality_score = compute_quality_score(schema_diff, missing_drift, categorical_drift, numeric_drift)

        if quality_score >= 90:
            st.success(f"Quality score: {quality_score}/100 — looks good.")
        elif quality_score >= 70:
            st.warning(f"Quality score: {quality_score}/100 — some drift detected, review the sections above.")
        else:
            st.error(f"Quality score: {quality_score}/100 — significant drift detected, investigate before trusting this data.")

        st.header(numbered("Summary"))
        st.caption("Same verdict, translated from 'tables and red boxes' into full sentences — the case file written up in plain English, no detective badge required to read it.")
        summary_text = generate_summary(schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score)
        st.write(summary_text)

        if not st.session_state.get("scan_saved"):
            save_scan(engine, baseline_file.name, new_file.name, quality_score, summary_text)
            st.session_state.scan_saved = True

        st.header(numbered("Cleaning suggestions"))
        st.caption("Concrete next steps for each issue found — what to double-check, who to ask, and what might break downstream if you don't. These also appear in the exported PDF report.")
        suggestions = build_suggestions(schema_diff, missing_drift, categorical_drift, numeric_drift)
        if suggestions:
            for i, suggestion in enumerate(suggestions, start=1):
                st.warning(f"**{i}.** {suggestion}")
        else:
            st.success("No action needed — this dataset looks consistent with the baseline.")

        # The Ollama button only makes sense where a local Ollama is reachable. On a hosted
        # deployment it never will be, so set ENABLE_OLLAMA=false in the cloud secrets to hide it.
        ollama_enabled = str(get_setting("ENABLE_OLLAMA", "true")).strip().lower() in ("1", "true", "yes")
        if ollama_enabled:
            st.header(numbered("AI explanation (optional)"))
            st.caption("Ask a locally-running Ollama model to rewrite the summary above in even friendlier language. Nothing leaves your machine — if Ollama isn't running, this just politely says so instead of doing anything scary.")
            if st.button("Explain with local LLM"):
                with st.spinner("Asking your local model..."):
                    explanation, error = generate_ollama_explanation(summary_text)
                if explanation:
                    st.write(explanation)
                else:
                    st.warning(error)

        st.header(numbered("Export report"))
        st.caption("Get a full diagnostic report, lab-report style, with the DataContract Lab letterhead on every page — or a lighter plain Markdown version.")
        report_markdown = build_markdown_report(schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score, summary_text)
        pdf_buffer = build_pdf_report(
            schema_diff, missing_drift, categorical_drift, numeric_drift, quality_score, summary_text,
            baseline_file.name, new_file.name,
        )
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download PDF report", data=pdf_buffer, file_name="datacontract_lab_report.pdf", mime="application/pdf",
            )
        with export_col2:
            st.download_button(
                "Download Markdown report", data=report_markdown, file_name="datacontract_lab_report.md", mime="text/markdown",
            )

st.divider()
st.header("Scan history")
st.caption("Every comparison you've run gets saved to MySQL — this is the paper trail, pulled fresh from the database on every page load.")
history_df = load_scan_history(engine)
if history_df.empty:
    st.info("No scans saved yet — run a comparison above to start building history.")
else:
    st.table(history_df)