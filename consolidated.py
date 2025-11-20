# consolidated.py

import pandas as pd


def prefix_cols(df: pd.DataFrame, prefix: str, exclude=None) -> pd.DataFrame:
    """
    Add a prefix to all columns except those in `exclude`.
    Useful to avoid name collisions when merging.
    """
    if df is None:
        return None

    if exclude is None:
        exclude = []

    new_cols = {}
    for col in df.columns:
        if col in exclude:
            new_cols[col] = col
        else:
            new_cols[col] = f"{prefix}{col}"
    return df.rename(columns=new_cols)


def build_consolidated_df(
    orders_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an order-level consolidated dataframe for Alpha.

    Design:
    - Base table is orders_df (already preprocessed by preprocess_orders).
      → One row per order.
    - Metadata is aggregated to one row per (Year, Week of supply)
      and then joined onto orders by (Year, Week).
    - Projection is aggregated to one row per (Year, Updated week number,
      Distributor) and then joined onto orders by
      (Year, Week number for Activity vs Projection, Distributor).

    This function does NOT modify the source data in Sheets/Airtable – it only
    creates an in-memory dataframe for the app.
    """
    if orders_df is None or orders_df.empty:
        raise ValueError("Orders dataframe is empty in build_consolidated_df().")

    # ------------------------------------------------------------------ #
    # 0) Base: orders (fact table)
    # ------------------------------------------------------------------ #
    merged = orders_df.copy()

    # Sanity: keys we expect from preprocess_orders
    for col in ["Year", "Week", "Distributor"]:
        if col not in merged.columns:
            raise ValueError(f"Orders data is missing '{col}' column.")

    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce")
    merged["Week"] = pd.to_numeric(merged["Week"], errors="coerce")
    merged = merged.dropna(subset=["Year", "Week"])
    merged["Year"] = merged["Year"].astype(int)
    merged["Week"] = merged["Week"].astype(int)

    # ------------------------------------------------------------------ #
    # 1) METADATA — one row per (Year, Week of supply)
    # ------------------------------------------------------------------ #
    if meta_df is not None and not meta_df.empty:
        meta = meta_df.copy()

        if "Year" not in meta.columns:
            raise ValueError("Metadata sheet is missing 'Year' column.")
        if "Week of supply" not in meta.columns:
            raise ValueError("Metadata sheet is missing 'Week of supply' column.")

        meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")
        meta["Week of supply"] = pd.to_numeric(meta["Week of supply"], errors="coerce")
        meta = meta.dropna(subset=["Year", "Week of supply"])
        meta["Year"] = meta["Year"].astype(int)
        meta["Week of supply"] = meta["Week of supply"].astype(int)

        group_keys = ["Year", "Week of supply"]
        agg_dict = {}

        for col in meta.columns:
            if col in group_keys:
                continue

            col_series = meta[col]

            if pd.api.types.is_bool_dtype(col_series):
                agg_dict[col] = "any"
            elif pd.api.types.is_numeric_dtype(col_series):
                # Sum numeric metadata if it exists
                agg_dict[col] = "sum"
            else:
                # Concatenate unique non-empty strings
                def _concat_unique(x):
                    vals = {str(v).strip() for v in x.dropna() if str(v).strip()}
                    return "; ".join(sorted(vals)) if vals else ""
                agg_dict[col] = _concat_unique

        meta_by_week = meta.groupby(group_keys, as_index=False).agg(agg_dict)

        # Prefix metadata columns to avoid clashes
        meta_prefixed = prefix_cols(
            meta_by_week,
            prefix="Meta_",
            exclude=["Year", "Week of supply"],
        )

        # Join onto orders by (Year, Week)
        merged = merged.merge(
            meta_prefixed,
            how="left",
            left_on=["Year", "Week"],
            right_on=["Year", "Week of supply"],
        )

    # ------------------------------------------------------------------ #
    # 2) PROJECTION — one row per (Year, Updated week, Distributor)
    # ------------------------------------------------------------------ #
    if proj_df is not None and not proj_df.empty:
        proj = proj_df.copy()

        if "Year" not in proj.columns:
            raise ValueError("Projection sheet is missing 'Year' column.")
        if "Updated week number" not in proj.columns:
            raise ValueError("Projection sheet is missing 'Updated week number' column.")

        # Try to find distributor column name in projection
        proj_dist_col = None
        for cand in [
            "Distributing company (from Company name)",
            "Distributor",
            "Distributing company",
        ]:
            if cand in proj.columns:
                proj_dist_col = cand
                break

        if proj_dist_col is None:
            raise ValueError(
                "Projection sheet is missing distributor column "
                "('Distributing company (from Company name)' or 'Distributor')."
            )

        proj["Year"] = pd.to_numeric(proj["Year"], errors="coerce")
        proj["Updated week number"] = pd.to_numeric(
            proj["Updated week number"], errors="coerce"
        )
        proj = proj.dropna(subset=["Year", "Updated week number", proj_dist_col])
        proj["Year"] = proj["Year"].astype(int)
        proj["Updated week number"] = proj["Updated week number"].astype(int)

        proj = proj.rename(
            columns={
                "Updated week number": "ProjWeek",
                proj_dist_col: "Distributor",
            }
        )

        proj_group_keys = ["Year", "ProjWeek", "Distributor"]
        proj_agg_dict = {}

        for col in proj.columns:
            if col in proj_group_keys:
                continue

            col_series = proj[col]
            if pd.api.types.is_bool_dtype(col_series):
                proj_agg_dict[col] = "any"
            elif pd.api.types.is_numeric_dtype(col_series):
                proj_agg_dict[col] = "sum"
            else:
                def _concat_unique(x):
                    vals = {str(v).strip() for v in x.dropna() if str(v).strip()}
                    return "; ".join(sorted(vals)) if vals else ""
                proj_agg_dict[col] = _concat_unique

        proj_by_key = proj.groupby(proj_group_keys, as_index=False).agg(proj_agg_dict)

        proj_prefixed = prefix_cols(
            proj_by_key,
            prefix="Proj_",
            exclude=["Year", "ProjWeek", "Distributor"],
        )

        if "Week number for Activity vs Projection" not in merged.columns:
            raise ValueError(
                "Orders data is missing 'Week number for Activity vs Projection' "
                "column needed to join projections."
            )

        # Convert to numeric but KEEP rows even if this field is empty.
        # Orders without this week will still appear; they just won't get projections.
        merged["Week number for Activity vs Projection"] = pd.to_numeric(
            merged["Week number for Activity vs Projection"], errors="coerce"
        )

        merged = merged.merge(
            proj_prefixed,
            how="left",
            left_on=["Year", "Week number for Activity vs Projection", "Distributor"],
            right_on=["Year", "ProjWeek", "Distributor"],
        )


    # Row count should still be == number of orders
    return merged
