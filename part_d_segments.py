"""Cluster hubs into three operating playbooks and compare them with tier labels."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.reporting import write_csv

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]


def main():
    table = pd.read_csv("data/modelling_table.csv")
    profile = table.groupby("hub_id", sort=True).agg(
        shipments=("shipment_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index()
    write_csv(profile, "outputs/hub_profile.csv")
    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    i, j = np.triu_indices(len(profile), k=1)
    distances = pairwise_distances(scaled)[i, j]
    separation = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        same = labels[i] == labels[j]
        separation.append({"K": k, "across_over_within": distances[~same].mean() / distances[same].mean(),
                           "cluster_sizes": "-".join(map(str, sorted(np.bincount(labels), reverse=True)))})
        if k == 3:
            profile["cluster"] = labels
    write_csv(pd.DataFrame(separation), "outputs/part_d_separation.csv")
    cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    names = dict(zip(cluster_order, ["Standard", "Watch", "Escalate"]))
    profile["segment"] = profile["cluster"].map(names)
    hubs = pd.read_csv("data/hubs.csv")
    segments = profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one",
    )
    write_csv(segments, "outputs/hub_segments.csv")
    segment_profile = profile.groupby("segment").agg(
        hubs=("hub_id", "size"), **{column: (column, "mean") for column in PROFILE_COLUMNS},
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()
    write_csv(segment_profile, "outputs/segment_profile.csv")
    implied = segments["hub_tier"].map({1: "Standard", 2: "Watch", 3: "Escalate"})
    mismatch = segments.loc[segments["segment"].ne(implied)]
    write_csv(mismatch, "outputs/tier_vs_segment_mismatch.csv")
    assert len(profile) == 24 and profile["shipments"].eq(50).all()
    print(f"Part D: {len(profile)} hubs; {len(mismatch)} tier/segment mismatches")


if __name__ == "__main__":
    main()
