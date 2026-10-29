"""Turn a batch of new parcels into deterministic operational actions."""
import joblib
import numpy as np
import pandas as pd

from src.reporting import write_checks, write_csv
from src.shipment_prep import FEATURES, build_features, clean_shipments


def main():
    raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    if feature_order != FEATURES:
        raise ValueError("Stored feature order differs from the trained feature contract")
    medians = pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
    threshold = float(pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Stored threshold must be between zero and one")
    clean, _ = clean_shipments(raw, medians=medians)
    features = build_features(clean, hubs)
    model = joblib.load("models/delay_model.pkl")
    probabilities = model.predict_proba(features[feature_order])[:, 1]
    result = features[["shipment_id", "hub_id"]].copy()
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
    result = result.merge(segments, on="hub_id", how="left", sort=False, validate="many_to_one")
    if not result["segment"].isin(["Standard", "Watch", "Escalate"]).all():
        raise ValueError("Every scored hub must have a recognized segment")
    result["delay_probability"] = probabilities.round(4)
    result["flagged"] = result["delay_probability"].ge(threshold).astype(int)
    result["action"] = np.select(
        [result["flagged"].eq(1) & result["segment"].eq("Escalate"), result["flagged"].eq(1)],
        ["Escalate to hub manager", "Proactive customer call"], default="No action",
    )
    assert result["shipment_id"].tolist() == raw.drop_duplicates()["shipment_id"].tolist()
    assert not result.isna().any().any()
    write_csv(result, "outputs/new_shipment_actions.csv")
    write_checks({"new_shipments": len(result), "flagged": result.flagged.sum(),
                  "escalations": result.action.eq("Escalate to hub manager").sum(),
                  "proactive_calls": result.action.eq("Proactive customer call").sum(),
                  "no_action": result.action.eq("No action").sum()},
                 "outputs/new_shipment_action_summary.csv")
    print(f"Scored {len(result)} parcels; {result.flagged.sum()} flagged; "
          f"{result.action.eq('Escalate to hub manager').sum()} hub-manager escalations")


if __name__ == "__main__":
    main()
