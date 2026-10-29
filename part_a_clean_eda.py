"""Clean historical parcels, build the modelling table, and export delay evidence."""
from pathlib import Path

import pandas as pd

from src.reporting import write_checks, write_csv
from src.shipment_prep import FEATURES, build_features, clean_shipments


def main():
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    dedup = raw.drop_duplicates()
    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = table["delivery_hours"].gt(table["promised_hours"]).astype(int)
    assert len(table) == len(clean) == 1200
    Path("data").mkdir(exist_ok=True)
    table.to_csv("data/modelling_table.csv", index=False)

    q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    checks = {
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(dedup),
        "rows_after_dedup": len(dedup),
        "missing_distance_km_before": dedup["distance_km"].isna().sum(),
        "missing_parcel_weight_before": dedup["parcel_weight_kg"].isna().sum(),
        "sentinel_weight_rows": dedup["parcel_weight_kg"].gt(100).sum(),
        "missing_declared_value_before": dedup["declared_value"].isna().sum(),
        "missing_payment_mode_before": dedup["payment_mode"].isna().sum(),
        "service_type_spellings_before": dedup["service_type"].nunique(),
        "service_type_spellings_after": clean["service_type"].nunique(),
        **{"median_" + column: value for column, value in medians.items()},
        "rows_after_hub_merge": len(table),
        "nulls_in_features_and_target": table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),
        "delayed_count": table["is_delayed"].sum(), "delay_rate": table["is_delayed"].mean(),
        "hours_q1": q1, "hours_q3": q3,
        "hours_lower_bound": lower, "hours_upper_bound": upper,
        "hours_outlier_count": (~table["delivery_hours"].between(lower, upper)).sum(),
        "declared_value_mean_inr": table["declared_value_inr"].mean(),
        "declared_value_median_inr": table["declared_value_inr"].median(),
        "log_declared_value_mean": table["log_declared_value"].mean(),
        "log_declared_value_median": table["log_declared_value"].median(),
    }
    write_checks(checks, "outputs/part_a_checks.csv")
    grouped = table.assign(distance_band=pd.cut(
        table["distance_km"], bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"], right=True,
    ))
    groups = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        result = grouped.groupby(dimension, observed=True).agg(
            shipments=("shipment_id", "size"), delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"),
        ).reset_index().rename(columns={dimension: "group"})
        result.insert(0, "dimension", dimension)
        groups.append(result)
    write_csv(pd.concat(groups, ignore_index=True), "outputs/part_a_delay_by_group.csv")
    correlations = table.select_dtypes(include="number").corr().rename_axis("feature").reset_index()
    write_csv(correlations, "outputs/part_a_correlations.csv")
    print(f"Part A: {len(table)} shipments; {table.is_delayed.sum()} delayed")


if __name__ == "__main__":
    main()
