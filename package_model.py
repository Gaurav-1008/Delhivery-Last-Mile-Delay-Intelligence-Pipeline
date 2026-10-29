"""Rebuild the chosen training-only classifier and save its scoring artifacts."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from part_c_classification import make_classifier, split_classification
from src.reporting import write_checks, write_csv
from src.shipment_prep import FEATURES, clean_shipments


def main():
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    table = pd.read_csv("data/modelling_table.csv")
    x_train, _, x_test, y_train, _, _ = split_classification(table)
    params = {"n_estimators": int(config["n_estimators"]),
              "max_depth": None if config["max_depth"] == "unlimited" else int(config["max_depth"]),
              "min_samples_leaf": int(config["min_samples_leaf"])}
    assert int(config["selector_k"]) == 7
    model = make_classifier(params).fit(x_train, y_train)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/delay_model.pkl")
    write_csv(pd.DataFrame({"threshold": [config["threshold"]]}), "models/delay_threshold.csv")
    write_csv(pd.DataFrame({"feature": FEATURES}), "models/feature_order.csv")
    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
    write_csv(pd.DataFrame(medians.items(), columns=["column", "median"]), "models/impute_values.csv")
    reloaded = joblib.load("models/delay_model.pkl")
    difference = np.abs(model.predict_proba(x_test)[:, 1] - reloaded.predict_proba(x_test)[:, 1]).max()
    assert difference <= 1e-9
    write_checks({"test_rows_scored": len(x_test), "reload_max_abs_probability_diff": difference,
                  "threshold": float(config["threshold"])}, "outputs/part_d_reload_check.csv")
    print(f"Packaged classifier; reload probability difference {difference:.4f}")


if __name__ == "__main__":
    main()
