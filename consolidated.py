def build_consolidated_df(
    orders_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an order-level consolidated dataframe for Alpha.

    Notes:
    - Orders are the fact table (1 row per order).
    - Metadata is aggregated to one row per (Year, Week of supply) and left-joined onto orders by (Year, Week).
    - Projections are aggregated to one row per (Year, ProjWeek, Distributor, Catalogue) and left-joined onto orders.
      If the strict (with Catalogue) join fails for some rows, we fill missing projections using a looser join
      (Year, ProjWeek, Distributor) to prevent false "projections not available" due to catalogue name mismatches.
    """
    if orders_df is None or orders_df.empty:
        raise ValueError("Orders dataframe is empty in build_consolidated_df().")

    # ------------------------------------------------------------------ #
    # 0) Base: orders (fact table)
    # ------------------------------------------------------------------ #
    merged = orders_df.copy()

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
                agg_dict[col] = "sum"
            else:
                def _concat_unique(x):
                    vals = {str(v).strip() for v in x.dropna() if str(v).strip()}
                    return "; ".join(sorted(vals)) if vals else ""
                agg_dict[col] = _concat_unique

        meta_by_week = meta.groupby(group_keys, as_index=False).agg(agg_dict)

        meta_prefixed = prefix_cols(
            meta_by_week,
            prefix="Meta_",
            exclude=["Year", "Week of supply"],
        )

        merged = merged.merge(
            meta_prefixed,
            how="left",
            left_on=["Year", "Week"],
            right_on=["Year", "Week of supply"],
        )

    # ------------------------------------------------------------------ #
    # 2) PROJECTION — strict join (with catalogue) + fallback join (without)
    # ------------------------------------------------------------------ #
    if proj_df is not None and not proj_df.empty:
        proj = proj_df.copy()

        if "Year" not in proj.columns:
            raise ValueError("Projection sheet is missing 'Year' column.")
        if "Updated week number" not in proj.columns:
            raise ValueError("Projection sheet is missing 'Updated week number' column.")

        # Find distributor column name in projection
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
        proj["Updated week number"] = pd.to_numeric(proj["Updated week number"], errors="coerce")
        proj = proj.dropna(subset=["Year", "Updated week number", proj_dist_col])
        proj["Year"] = proj["Year"].astype(int)
        proj["Updated week number"] = proj["Updated week number"].astype(int)

        proj = proj.rename(
            columns={
                "Updated week number": "ProjWeek",
                proj_dist_col: "Distributor",
            }
        )

        if "Week number for Activity vs Projection" not in merged.columns:
            raise ValueError(
                "Orders data is missing 'Week number for Activity vs Projection' "
                "column needed to join projections."
            )

        merged["Week number for Activity vs Projection"] = pd.to_numeric(
            merged["Week number for Activity vs Projection"], errors="coerce"
        )

        # Normalize join keys (requires your _norm_key() to exist in scope)
        merged["Distributor"] = _norm_key(merged["Distributor"])
        proj["Distributor"] = _norm_key(proj["Distributor"])

        have_catalogue = ("Catalogue description (sold as)" in merged.columns) and (
            "Catalogue description (sold as)" in proj.columns
        )
        if have_catalogue:
            merged["Catalogue description (sold as)"] = _norm_key(merged["Catalogue description (sold as)"])
            proj["Catalogue description (sold as)"] = _norm_key(proj["Catalogue description (sold as)"])

        # -------------------------
        # Strict aggregation (with catalogue if available)
        # -------------------------
        strict_keys = ["Year", "ProjWeek", "Distributor"]
        if have_catalogue:
            strict_keys.append("Catalogue description (sold as)")

        proj_agg_dict = {}
        for col in proj.columns:
            if col in strict_keys:
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

        proj_strict = proj.groupby(strict_keys, as_index=False).agg(proj_agg_dict)

        proj_prefixed = prefix_cols(
            proj_strict,
            prefix="Proj_",
            exclude=strict_keys,
        )

        if have_catalogue:
            merged = merged.merge(
                proj_prefixed,
                how="left",
                left_on=["Year", "Week number for Activity vs Projection", "Distributor", "Catalogue description (sold as)"],
                right_on=["Year", "ProjWeek", "Distributor", "Catalogue description (sold as)"],
            )
        else:
            merged = merged.merge(
                proj_prefixed,
                how="left",
                left_on=["Year", "Week number for Activity vs Projection", "Distributor"],
                right_on=["Year", "ProjWeek", "Distributor"],
            )

        # -------------------------
        # Fallback fill (ignore catalogue mismatches)
        # Only runs if catalogue exists and strict join left gaps.
        # -------------------------
        strict_amount_col = "Proj_Proj_Amount"  # because prefix_cols prefixes "Proj_" onto original "Proj_Amount"
        if have_catalogue and strict_amount_col in merged.columns:
            missing_mask = merged[strict_amount_col].isna()
            if missing_mask.any() and "Proj_Amount" in proj.columns:
                proj_loose = proj.groupby(["Year", "ProjWeek", "Distributor"], as_index=False).agg({"Proj_Amount": "sum"})
                proj_loose_prefixed = prefix_cols(
                    proj_loose,
                    prefix="ProjLoose_",
                    exclude=["Year", "ProjWeek", "Distributor"],
                )

                fill = merged.loc[missing_mask, ["Year", "Week number for Activity vs Projection", "Distributor"]].merge(
                    proj_loose_prefixed,
                    how="left",
                    left_on=["Year", "Week number for Activity vs Projection", "Distributor"],
                    right_on=["Year", "ProjWeek", "Distributor"],
                )

                merged.loc[missing_mask, strict_amount_col] = fill["ProjLoose_Proj_Amount"].values

    return merged

