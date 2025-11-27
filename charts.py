import json
import re
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# Detect ```chart { ... } ``` blocks in the assistant's answer
CHART_BLOCK_PATTERN = re.compile(
    r"```chart\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_chart_specs(answer: str) -> List[Dict[str, Any]]:
    """Extract JSON chart specs from ```chart ...``` code fences."""
    specs: List[Dict[str, Any]] = []
    if not answer:
        return specs

    for match in CHART_BLOCK_PATTERN.finditer(answer):
        raw_json = match.group(1).strip()
        try:
            spec = json.loads(raw_json)
            if isinstance(spec, dict):
                specs.append(spec)
        except Exception:
            # ignore broken specs
            continue

    return specs


def _build_chart(spec: Dict[str, Any]) -> Optional[alt.Chart]:
    """Build an Altair chart from a spec produced by _build_chart_block."""
    data = spec.get("data")
    if not isinstance(data, list) or not data:
        return None

    df = pd.DataFrame(data)
    chart_type = (spec.get("type") or "bar").lower()
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    title = spec.get("title")

    if not x or not y or x not in df.columns or y not in df.columns:
        return None

    # Build chart by type
    if chart_type == "pie":
        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(
                theta=alt.Theta(field=y, type="quantitative"),
                color=alt.Color(field=x, type="nominal"),
                tooltip=[c for c in df.columns],
            )
        )
    elif chart_type == "line":
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(x),
                y=alt.Y(y),
                tooltip=[c for c in df.columns],
                # if a color field was provided and exists, use it; else a fixed color
                color=color if color and color in df.columns else alt.value("#4A2E88"),
            )
        )
    else:  # default = bar
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(x, sort="-y"),
                y=alt.Y(y),
                tooltip=[c for c in df.columns],
                color=color if color and color in df.columns else alt.value("#4A2E88"),
            )
        )

    if title:
        chart = chart.properties(title=title)

    return chart


def render_chart_from_answer(answer: str) -> None:
    """
    Look for ```chart { ... } ``` blocks in the answer, build Altair charts,
    and render them directly in Streamlit.
    """
    specs = _extract_chart_specs(answer)
    if not specs:
        return

    for spec in specs:
        chart = _build_chart(spec)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)