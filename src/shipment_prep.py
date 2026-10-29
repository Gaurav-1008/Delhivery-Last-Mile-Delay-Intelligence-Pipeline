"""Shared dispatch-time cleaning and features for training and batch scoring."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
IMPUTE_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].str.strip().str.title()
    if not clean["service_type"].isin(PROMISED_HOURS).all():
        raise ValueError("Missing or unrecognized service_type")
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    if not clean["payment_mode"].isin(["COD", "Prepaid", "Unknown"]).all():
        raise ValueError("Unrecognized payment_mode")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="raise",
    )
    clean = clean.drop(columns="declared_value")
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"], format="%Y-%m-%d %H:%M", errors="raise")
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {column: float(clean[column].median()) for column in IMPUTE_COLUMNS}
    if set(medians) != set(IMPUTE_COLUMNS) or not all(
        np.isfinite(value) and value > 0 for value in medians.values()
    ):
        raise ValueError("Expected three finite, positive imputation medians")
    medians = {column: float(medians[column]) for column in IMPUTE_COLUMNS}
    clean[IMPUTE_COLUMNS] = clean[IMPUTE_COLUMNS].fillna(medians)
    if clean["shipment_id"].isna().any() or clean["shipment_id"].duplicated().any():
        raise ValueError("Shipment IDs must be present and unique after exact deduplication")
    if clean["dispatch_time"].isna().any() or (clean[IMPUTE_COLUMNS] <= 0).any().any():
        raise ValueError("Dispatch timestamps and positive shipment measurements are required")
    return clean, medians


def build_features(clean, hubs):
    features = clean.merge(hubs, on="hub_id", how="left", sort=False, validate="many_to_one")
    if features["hub_city"].isna().any():
        raise ValueError("Every shipment must reference a known hub")
    features["promised_hours"] = features["service_type"].map(PROMISED_HOURS).astype(int)
    features["is_cod"] = features["payment_mode"].eq("COD").astype(int)
    features["is_weekend"] = features["dispatch_time"].dt.dayofweek.ge(5).astype(int)
    features["dispatch_hour"] = features["dispatch_time"].dt.hour.astype(int)
    features["log_declared_value"] = np.log(features["declared_value_inr"])
    if not np.isfinite(features[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Model features must all be finite and non-missing")
    return features
