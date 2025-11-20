import json
import re
from typing import Any, Dict, Optional, List

import pandas as pd
import altair as alt
import streamlit as st


# Pattern for capturing the ```chart ... ``` JSON block
CHART_BLOCK_PATTERN = re.compile(
    r"```chart\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def strip_chart_blocks(answer: str) -> str:
    """
    Remove the ```chart ...``` JSON block from the answer text
    so the user does not see raw JSON in the chat.
    """
    if not answer:
        return ""
    cleaned = CHART_BLOCK_PATTERN.sub("", answer)
    return cleaned.strip()


def _extract_chart_spec(answer: str) -> Optional[Dict[str, Any]]:
    """
    Extract the JSON chart spec from the ```chart``` code block.
    """
    if not answer:
        return None

    match = CHART_BLOCK_PATTERN.search(answer)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        spec = json.loads(raw)
        if isinstance(spec, dict):
            return spec
    except json.JSONDecodeError:
        st.warning("Could not parse chart JSON.")
        return None

    return None


def _build_chart(spec: Dict[str, Any]):
    """
    Build an Altair chart from the parsed spec.
    Supports: line, bar, pie
    """
    data: List[Dict[str, Any]] = spec.get("data") or []
    if not data:
        st.warning("Chart spec contains no data.")
        return None

    df = pd.DataFrame(data)

    chart_type = (spec.get("type") or "bar").lower()
    x_field = spec.get("xField")
    y_field = spec.get("yField")
    series_field = spec.get("seriesField")

    if not x_field or not y_field:
        st.warning("Chart spec missing xField or yField.")
        return None

    if x_field not in df.columns or y_field not in df.columns:
        st.warning("Chart spec references columns not found in data.")
        return None

    # ------------------ LINE ------------------
    if chart_type == "line":
        enc = {
            "x": alt.X(x_field, title=x_field),
            "y": alt.Y(y_field, title=y_field),
        }
        if series_field and series_field in df.columns:
            enc["color"] = alt.Color(series_field, title=series_field)

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(**enc)
            .properties(width="container", height=400)
        )

    # ------------------ BAR ------------------
    elif chart_type == "bar":
        enc = {
            "x": alt.X(x_field, title=x_field),
            "y": alt.Y(y_field, title=y_field),
        }
        if series_field and series_field in df.columns:
            enc["color"] = alt.Color(series_field, title=series_field)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(**enc)
            .properties(width="container", height=400)
        )

    # ------------------ PIE ------------------
    elif chart_type == "pie":
        total = df[y_field].sum()

        # Add percentage column
        if total and total != 0:
            df["__pct"] = df[y_field] / total * 100

        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(
                theta=alt.Theta(field=y_field, type="quantitative"),
                color=alt.Color(field=x_field, type="nominal"),
                tooltip=[
                    x_field,
                    alt.Tooltip(y_field, format=",.1f"),
                    alt.Tooltip("__pct", title="Share (%)", format=".1f"),
                ],
            )
            .properties(width=400, height=400)
        )

    # ------------------ FALLBACK ------------------
    else:
        st.warning(f"Unknown chart type '{chart_type}', using bar chart instead.")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(x_field),
                y=alt.Y(y_field),
            )
            .properties(width="container", height=400)
        )

    return chart


def render_chart_from_answer(answer: str) -> None:
    """
    1. Extract the chart JSON from the assistant's reply
    2. Build Altair chart
    3. Render it in Streamlit
    """
    spec = _extract_chart_spec(answer)
    if not spec:
        return

    chart = _build_chart(spec)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
