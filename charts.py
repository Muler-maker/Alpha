import json
import re
from typing import List, Dict, Any

import altair as alt
import pandas as pd

# Regex to find ```chart ... ``` blocks
_CHART_BLOCK_RE = re.compile(
    r"```chart\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def strip_chart_blocks(answer: str) -> str:
    """
    Remove all ```chart ... ``` blocks from the answer text.
    Used so the user only sees the explanation + tables, not the JSON.
    """
    if not answer:
        return answer
    return _CHART_BLOCK_RE.sub("", answer).strip()


def extract_chart_blocks(answer: str) -> List[Dict[str, Any]]:
    """
    Extract and parse all chart JSON blocks from the answer.
    Each block is expected to be a JSON object describing the chart.
    """
    if not answer:
        return []

    blocks: List[Dict[str, Any]] = []
    for match in _CHART_BLOCK_RE.finditer(answer):
        raw = match.group(1).strip()
        try:
            spec = json.loads(raw)
            if isinstance(spec, dict):
                blocks.append(spec)
        except Exception:
            continue
    return blocks


def _build_chart_from_spec(spec: Dict[str, Any]):
    """
    Build an Altair chart from the JSON spec produced by _build_chart_block
    in query_engine.py.

    Expected spec shape:
      {
        "type": "bar" | "line" | "pie",
        "xField": "...",
        "yField": "...",
        "seriesField": "... or None",
        "aggregation": "...",
        "group_by": [...],
        "data": [{...}, {...}, ...]
      }
    """
    data = spec.get("data") or []
    if not data:
        return None

    df = pd.DataFrame(data)

    chart_type = (spec.get("type") or "bar").lower()
    x_field = spec.get("xField")
    y_field = spec.get("yField")
    series_field = spec.get("seriesField")

    if not x_field or not y_field:
        # Not enough info to build a chart
        return None

    # Pie chart: special handling
    if chart_type == "pie":
        angle = alt.Theta(y_field, type="quantitative")
        color_field = series_field or x_field
        color = alt.Color(color_field, type="nominal")
        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(angle=angle, color=color, tooltip=list(df.columns))
        )
        return chart

    # Common encodings for bar / line
    x_enc = alt.X(x_field, type="nominal", sort="-y")
    y_enc = alt.Y(y_field, type="quantitative")
    color_enc = alt.Color(series_field, type="nominal") if series_field else None

    if chart_type == "line":
        base = alt.Chart(df).mark_line()
    else:
        # default to bar
        base = alt.Chart(df).mark_bar()

    if color_enc is not None:
        chart = base.encode(x=x_enc, y=y_enc, color=color_enc, tooltip=list(df.columns))
    else:
        chart = base.encode(x=x_enc, y=y_enc, tooltip=list(df.columns))

    return chart.properties(width="container")


def render_chart_from_answer(answer: str) -> List[alt.Chart]:
    """
    Parse the answer, build charts from any chart blocks, and return
    a list of Altair charts. The caller is responsible for displaying them.

    Example usage in Streamlit:

        charts = render_chart_from_answer(raw_answer)
        for ch in charts:
            st.altair_chart(ch, use_container_width=True)
    """
    charts: List[alt.Chart] = []
    specs = extract_chart_blocks(answer)
    for spec in specs:
        ch = _build_chart_from_spec(spec)
        if ch is not None:
            charts.append(ch)
    return charts
