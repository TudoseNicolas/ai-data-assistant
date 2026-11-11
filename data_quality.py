"""Data quality checks and cleaning (plain pandas, no AI)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

# A text column is treated as "numbers stored as text" / "dates stored as text"
# when at least this share of its non-empty values can be converted.
PARSE_THRESHOLD = 0.9

# Matches values such as "1.099,00 lei", "1,099.00", "€ 25", "12 %", "-3.5".
_NUMBER_LIKE = re.compile(
    r"^\s*(?:lei|ron|eur|usd|€|\$|£)?\s*[-+]?\d[\d.,\s]*\s*(?:lei|ron|eur|usd|€|\$|£|%)?\s*$",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_FULL_DATE = re.compile(r"^\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})(\s.*)?$")


def _to_dates(series: pd.Series) -> pd.Series:
    """Parse dates; dd.mm.yyyy (Romanian style) is read day-first, ISO dates stay as they are."""
    return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)


@dataclass
class Issue:
    """One problem found in the data."""

    id: str
    kind: str  # duplicates | numeric_text | date_text | missing | inconsistent_text
    severity: str  # high | medium | low
    title: str
    description: str
    column: str | None = None
    count: int = 0


# helpers


def is_text_column(series: pd.Series) -> bool:
    """True for text columns (object dtype in pandas 2, "str" dtype in pandas 3)."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def parse_number(value) -> float | None:
    """Convert values like '1.099,00 lei' or '1,099.00' to a float. None if not a number."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or _FULL_DATE.match(text) or not _NUMBER_LIKE.match(text):
        return None
    text = re.sub(r"[^\d,.\-+]", "", text)
    if "," in text and "." in text:
        # The separator that appears last is the decimal separator.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        whole, _, decimals = text.rpartition(",")
        if text.count(",") == 1 and len(decimals) in (1, 2):
            text = f"{whole}.{decimals}"  # "89,50" -> decimal comma
        else:
            text = text.replace(",", "")  # "12,480" -> thousands separator
    elif text.count(".") > 1:
        text = text.replace(".", "")  # "1.234.567" -> thousands separators
    try:
        return float(text)
    except ValueError:
        return None


def normalize_text(value) -> str:
    """Lower-case, trimmed, single-spaced, without diacritics ('București ' -> 'bucuresti')."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def _blank_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """Treat empty or whitespace-only text cells as missing values."""
    out = df.copy()
    for col in out.columns:
        if is_text_column(out[col]):
            blank = out[col].astype("string").str.strip().eq("").fillna(False).astype(bool)
            out[col] = out[col].mask(blank)
    return out


def _numeric_text_columns(df: pd.DataFrame) -> dict[str, int]:
    """Text columns whose values are really numbers -> number of cells affected."""
    found = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty or not is_text_column(df[col]):
            continue
        parsed = series.map(parse_number)
        if parsed.notna().mean() >= PARSE_THRESHOLD:
            found[col] = int(len(series))
    return found


def _date_text_columns(df: pd.DataFrame, skip: set[str]) -> dict[str, int]:
    """Text columns whose values are really dates -> number of cells affected."""
    found = {}
    for col in df.columns:
        if col in skip or not is_text_column(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty or series.str.contains(_DATE_LIKE).mean() < PARSE_THRESHOLD:
            continue
        if _to_dates(series).notna().mean() >= PARSE_THRESHOLD:
            found[col] = int(len(series))
    return found


def _canonical_spellings(series: pd.Series) -> dict[str, str]:
    """Map every spelling variant to the most common (trimmed) spelling of its group."""
    values = series.dropna().astype(str)
    if values.empty:
        return {}
    keys = values.map(normalize_text)
    trimmed = values.str.strip()
    canonical = {}
    for key, group in trimmed.groupby(keys):
        canonical[key] = group.value_counts().index[0]
    return {value: canonical[normalize_text(value)] for value in values.unique()}


def _inconsistent_text_columns(df: pd.DataFrame, skip: set[str]) -> dict[str, tuple[int, list[str]]]:
    """Text columns with spelling variants -> (cells affected, example variants)."""
    found = {}
    for col in df.columns:
        if col in skip or not is_text_column(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty or series.nunique() > 0.5 * len(series):
            continue  # free text, not a category
        mapping = _canonical_spellings(series)
        changed = series[series.map(mapping) != series]
        if not changed.empty:
            examples = list(dict.fromkeys(changed.head(50)))[:3]
            found[col] = (int(len(changed)), examples)
    return found


# public API


def column_type(series: pd.Series) -> str:
    """A friendly type label for the Overview tab."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if is_text_column(series):
        frame = series.to_frame()
        if _numeric_text_columns(frame):
            return "text (should be number)"
        if _date_text_columns(frame, set()):
            return "text (should be date)"
    return "text"


def overview(df: pd.DataFrame) -> dict:
    """Headline numbers for the Overview tab."""
    clean = _blank_to_na(df)
    cells = max(df.shape[0] * df.shape[1], 1)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_pct": float(clean.isna().sum().sum() / cells * 100),
        "duplicates": int(df.duplicated().sum()),
        "types": {col: column_type(df[col]) for col in df.columns},
    }


def analyze(df: pd.DataFrame) -> list[Issue]:
    """Find data quality problems, most serious first."""
    issues: list[Issue] = []
    data = _blank_to_na(df)
    rows = max(len(data), 1)

    duplicates = int(data.duplicated().sum())
    if duplicates:
        issues.append(Issue(
            id="duplicates", kind="duplicates", severity="high",
            title=f"{duplicates:,} duplicate rows",
            description="Identical rows that would inflate totals. Fix: keep the first, remove the rest.",
            count=duplicates,
        ))

    numeric_text = _numeric_text_columns(data)
    for col, count in numeric_text.items():
        example = data[col].dropna().astype(str).iloc[0]
        issues.append(Issue(
            id=f"numeric_text:{col}", kind="numeric_text", severity="high", column=col,
            title=f"“{col}” stored as text",
            description=f"Values like “{example}” can’t be summed or averaged. Fix: convert to numbers.",
            count=count,
        ))

    date_text = _date_text_columns(data, skip=set(numeric_text))
    for col, count in date_text.items():
        issues.append(Issue(
            id=f"date_text:{col}", kind="date_text", severity="low", column=col,
            title=f"“{col}” stored as text instead of dates",
            description="Real dates make trends by month or year possible. Fix: convert to dates.",
            count=count,
        ))

    for col in data.columns:
        missing = int(data[col].isna().sum())
        if not missing:
            continue
        pct = missing / rows * 100
        severity = "high" if pct >= 20 else "medium" if pct >= 1 else "low"
        if col in date_text or pd.api.types.is_datetime64_any_dtype(data[col]):
            fix = "remove the rows without a date"
        elif col in numeric_text or pd.api.types.is_numeric_dtype(data[col]):
            fix = "fill with the column’s median"
        else:
            fix = "fill with “Unknown” so the rows stay in the analysis"
        issues.append(Issue(
            id=f"missing:{col}", kind="missing", severity=severity, column=col,
            title=f"{missing:,} missing values in “{col}” ({pct:.1f}%)",
            description=f"Fix: {fix}.",
            count=missing,
        ))

    inconsistent = _inconsistent_text_columns(data, skip=set(numeric_text) | set(date_text))
    for col, (count, examples) in inconsistent.items():
        shown = ", ".join(f"“{e}”" for e in examples)
        issues.append(Issue(
            id=f"inconsistent_text:{col}", kind="inconsistent_text", severity="low", column=col,
            title=f"Inconsistent spelling in “{col}”",
            description=f"{count:,} cells such as {shown} are spelled differently from the rest of "
                        "their group. Fix: trim spaces and use one spelling.",
            count=count,
        ))

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda issue: order[issue.severity])


def clean(df: pd.DataFrame, issues: list[Issue], selected_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the selected fixes and return (cleaned copy, log of what changed)."""
    chosen = [issue for issue in issues if issue.id in set(selected_ids)]
    out = _blank_to_na(df)
    log: list[str] = []

    # spelling first, so near-duplicates become exact duplicates
    for issue in (i for i in chosen if i.kind == "inconsistent_text"):
        mapping = _canonical_spellings(out[issue.column])
        out[issue.column] = out[issue.column].map(lambda v: mapping.get(v, v) if pd.notna(v) else v)
        log.append(f"Standardised spelling in “{issue.column}”.")

    for issue in (i for i in chosen if i.kind == "numeric_text"):
        out[issue.column] = pd.to_numeric(out[issue.column].map(parse_number), errors="coerce")
        log.append(f"Converted “{issue.column}” to numbers.")
    for issue in (i for i in chosen if i.kind == "date_text"):
        out[issue.column] = _to_dates(out[issue.column])
        log.append(f"Converted “{issue.column}” to dates.")

    for issue in (i for i in chosen if i.kind == "missing"):
        col = issue.column
        if pd.api.types.is_datetime64_any_dtype(out[col]) or col in {
            i.column for i in issues if i.kind == "date_text"
        }:
            before = len(out)
            out = out[out[col].notna()]
            log.append(f"Removed {before - len(out):,} rows without “{col}”.")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
            log.append(f"Filled missing “{col}” with the median.")
        else:
            out[col] = out[col].fillna("Unknown")
            log.append(f"Filled missing “{col}” with “Unknown”.")

    if any(i.kind == "duplicates" for i in chosen):
        before = len(out)
        out = out.drop_duplicates()
        log.append(f"Removed {before - len(out):,} duplicate rows.")

    return out.reset_index(drop=True), log


def quality_score(df: pd.DataFrame) -> int:
    """A simple 0–100 score: start at 100 and subtract for each kind of problem."""
    issues = analyze(df)
    info = overview(df)
    rows = max(info["rows"], 1)
    score = 100.0
    score -= min(info["missing_pct"] * 2, 30)
    score -= min(info["duplicates"] / rows * 100 * 2, 30)
    score -= 6 * sum(1 for i in issues if i.kind == "numeric_text")
    score -= 2 * sum(1 for i in issues if i.kind in ("date_text", "inconsistent_text"))
    return int(max(0, min(100, round(score))))
