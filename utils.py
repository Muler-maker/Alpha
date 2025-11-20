# utils.py

import json
import re
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI

# Create a single shared client (uses OPENAI_API_KEY from your .env)
client = OpenAI()

# ---- 1. SYSTEM PROMPT FOR THE INTERPRETER ----

INTERPRETER_SYSTEM_PROMPT = """
You are a translation layer between a business user and a pandas DataFrame.

Your job:
- Read the user's question about Isotopia's orders/projections/events.
- Output a SINGLE JSON object describing how to query the data.
- Do NOT include any explanation, only JSON.

Available data concepts (columns may exist with these or similar names):
- Measures (numeric, can be aggregated):
  - "Total_mCi"        : total ordered activity in mCi
  - "Proj_Projection mCi" : projected activity in mCi (from projection table)

- Typical dimensions / filters:
  - "Customer"
  - "Distributor"
  - "Country"
  - "Product"
  - "Year"
  - "Week"
  - "Week number for Activity vs Projection"
  - "Week of supply"
  - "Shipping Status" (or similar)
  - "Meta_Event" (major events / metadata)

OUTPUT SCHEMA (very important):

{
  "operation": "aggregate",          // "aggregate" or "detail"
  "measure": "Total_mCi",            // which numeric column to use (or null if none)
  "aggregation": "sum",              // "sum", "count", "avg"
  "filters": {
     "Customer": ["DSD"],            // lists or scalars for exact matches
     "Year": 2025,
     "Shipping Status": [
       "Shipped",
       "Partially shipped",
       "Shipped and arrived late",
       "Order being processed"
     ]
  },
  "group_by": []                     // list of columns to group by, may be empty
}

Guidelines:
- When the user asks "how many mCi / what is the amount / volume", assume:
    measure = "Total_mCi", aggregation = "sum", operation = "aggregate".
- When the user asks "how many orders" (count of rows) use:
    measure = null, aggregation = "count".
- If they ask to see a breakdown (e.g., "by country", "per product"):
    fill "group_by" with the requested dimensions.
- Always include appropriate filters:
    - Year if the question specifies a year (e.g., 2025).
    - Customer / Distributor / Country when mentioned.
- If shipping status is not specified but the question is about orders volume,
  default to these statuses when they exist:
  ["Shipped", "Partially shipped", "Shipped and arrived late", "Order being processed"].
- Be strict with JSON: no comments, no trailing commas, no extra text.
"""


# ---- 2. SMALL HELPER: CLEAN JSON TEXT ----

def _extract_json(text: str) -> str:
    """
    Remove ```json ... ``` fences if the model returns them,
    and return the bare JSON string.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


# ---- 3. BUILD QUERY SPEC WITH GPT ----

def build_query_spec(question: str) -> Dict[str, Any]:
    """
    Call GPT to translate the natural language question into
    a structured JSON query spec.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = resp.choices[0].message.content or ""
    json_text = _extract_json(raw)
    spec = json.loads(json_text)
    return spec


# ---- 4. EXECUTE QUERY SPEC ON A DATAFRAME ----

def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    filtered = df.copy()
    for col, value in filters.items():
        if col not in filtered.columns:
            # Silently ignore filters for columns that don't exist in the DF
            continue

        if isinstance(value, list):
            filtered = filtered[filtered[col].isin(value)]
        else:
            filtered = filtered[filtered[col] == value]
    return filtered


def _aggregate(df: pd.DataFrame, measure: str | None,
               aggregation: str, group_by: List[str]) -> pd.DataFrame:
    if group_by:
        if measure is None:
            # count rows per group
            result = df.groupby(group_by).size().reset_index(name="count")
        else:
            if measure not in df.columns:
                raise ValueError(f"Measure column '{measure}' not found in data.")
            agg_map = {
                "sum": "sum",
                "avg": "mean",
                "mean": "mean",
                "count": "count",
            }
            agg_func = agg_map.get(aggregation.lower(), "sum")
            result = (
                df.groupby(group_by)[measure]
                .agg(agg_func)
                .reset_index()
            )
    else:
        if measure is None:
            # count all rows
            result_value = len(df)
            result = pd.DataFrame({"count": [result_value]})
        else:
            if measure not in df.columns:
                raise ValueError(f"Measure column '{measure}' not found in data.")
            agg_map = {
                "sum": df[measure].sum,
                "avg": df[measure].mean,
                "mean": df[measure].mean,
                "count": df[measure].count,
            }
            func = agg_map.get(aggregation.lower(), df[measure].sum)
            result_value = func()
            result = pd.DataFrame({measure: [result_value]})
    return result


# ---- 5. HIGH-LEVEL FUNCTION USED BY THE APP ----

def answer_question_from_df(question: str, df: pd.DataFrame) -> str:
    """
    Main entry point for the Streamlit app.

    1) Ask GPT to build a query spec.
    2) Apply filters and aggregations on the consolidated dataframe.
    3) Return a human-readable answer with the numeric result
       and a short explanation of how it was computed.
    """
    spec = build_query_spec(question)

    operation = spec.get("operation", "aggregate")
    measure = spec.get("measure")
    aggregation = spec.get("aggregation", "sum")
    filters = spec.get("filters", {}) or {}
    group_by = spec.get("group_by", []) or []

    # 1. Filter the dataframe
    df_filtered = _apply_filters(df, filters)

    if operation == "aggregate":
        result_df = _aggregate(df_filtered, measure, aggregation, group_by)
    else:
        # For now we treat any non-aggregate as "detail" and just return a count
        result_df = pd.DataFrame({"count": [len(df_filtered)]})

    # Build a simple textual explanation
    filter_descriptions = []
    for col, val in filters.items():
        filter_descriptions.append(f"{col} = {val}")
    filter_text = ", ".join(filter_descriptions) if filter_descriptions else "no filters"

    # Format result
    if result_df.shape[0] == 1 and result_df.shape[1] == 1:
        # Single scalar
        col = result_df.columns[0]
        value = result_df.iloc[0, 0]
        answer_text = (
            f"Based on the current data, the result is **{value:,.0f} {col}** "
            f"after applying {filter_text}."
        )
    else:
        # Small table summary
        answer_text = (
            "Here is a summary based on your question "
            f"(filters applied: {filter_text}):\n\n"
        )
        answer_text += result_df.to_markdown(index=False)

    return answer_text
