"""Run after the six pipeline stages; checks use computed artifacts and brief references."""
from pathlib import Path
import importlib.metadata
import re
import subprocess
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

from part_c_classification import make_classifier, split_classification
from src.reporting import classification_metrics
from src.shipment_prep import FEATURES, IMPUTE_COLUMNS, NOISE_COLUMNS, build_features, clean_shipments


def read(name):
    return pd.read_csv("outputs/" + name + ".csv")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv("data/raw_shipments.csv")
        cls.hubs = pd.read_csv("data/hubs.csv")
        cls.table = pd.read_csv("data/modelling_table.csv")

    def test_cleaning_and_stored_medians(self):
        original = self.raw.copy(deep=True)
        clean, medians = clean_shipments(self.raw)
        pd.testing.assert_frame_equal(self.raw, original)
        self.assertEqual(len(clean), 1200)
        self.assertEqual(set(medians), set(IMPUTE_COLUMNS))
        np.testing.assert_allclose([medians[column] for column in IMPUTE_COLUMNS], [314.95, 1.83, 1121.5], atol=1e-10)
        self.assertFalse(clean[IMPUTE_COLUMNS].isna().any().any())
        self.assertLessEqual(clean.parcel_weight_kg.max(), 40)
        self.assertEqual(set(clean.service_type), {"Express", "Standard", "Economy"})
        self.assertEqual(clean.payment_mode.eq("Unknown").sum(), 18)
        new = pd.read_csv("data/new_shipments.csv")
        stored = pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
        scored, reused = clean_shipments(new, medians=stored)
        np.testing.assert_allclose([stored[column] for column in IMPUTE_COLUMNS],
                                   [medians[column] for column in IMPUTE_COLUMNS], atol=1e-10)
        self.assertEqual(reused, stored)
        for raw_column, clean_column in [("distance_km", "distance_km"), ("parcel_weight_kg", "parcel_weight_kg"),
                                          ("declared_value", "declared_value_inr")]:
            self.assertTrue(scored.loc[new[raw_column].isna(), clean_column].eq(stored[clean_column]).all())

    def test_features_are_independent_of_delivery_outcome(self):
        clean, _ = clean_shipments(self.raw)
        historical = build_features(clean, self.hubs)
        no_outcome = build_features(clean.drop(columns="delivery_hours"), self.hubs)
        changed_outcome = build_features(clean.assign(delivery_hours=-999), self.hubs)
        pd.testing.assert_frame_equal(historical[FEATURES], no_outcome[FEATURES])
        pd.testing.assert_frame_equal(historical[FEATURES], changed_outcome[FEATURES])
        self.assertEqual(historical.shipment_id.tolist(), clean.shipment_id.tolist())

    def test_invalid_input_fails_before_scoring(self):
        sample = self.raw.iloc[:3].copy()
        for column, bad_value in [("service_type", "Drone"), ("payment_mode", "cash"),
                                   ("declared_value", "₹oops"), ("declared_value", "₹0"),
                                   ("distance_km", -1), ("dispatch_time", "invalid")]:
            with self.subTest(column=column, value=bad_value), self.assertRaises(ValueError):
                clean_shipments(sample.assign(**{column: bad_value}))
        with self.assertRaises(ValueError):
            clean_shipments(sample, medians={"distance_km": 100.0})
        clean, _ = clean_shipments(sample)
        with self.assertRaises(ValueError):
            build_features(clean.assign(hub_id="MISSING"), self.hubs)
        with self.assertRaises(ValueError):
            build_features(clean, pd.concat([self.hubs, self.hubs.iloc[:1]]))
        with self.assertRaises(ValueError):
            clean_shipments(sample.assign(shipment_id="DUPLICATE"))

    def test_part_a_reference_checks(self):
        expected = {
            "raw_rows": 1224, "duplicate_rows_removed": 24, "rows_after_dedup": 1200,
            "missing_distance_km_before": 36, "missing_parcel_weight_before": 24,
            "sentinel_weight_rows": 5, "missing_declared_value_before": 12,
            "missing_payment_mode_before": 18, "service_type_spellings_before": 12,
            "service_type_spellings_after": 3, "median_distance_km": 314.95,
            "median_parcel_weight_kg": 1.83, "median_declared_value_inr": 1121.5,
            "rows_after_hub_merge": 1200, "nulls_in_features_and_target": 0,
            "delayed_count": 278, "delay_rate": 0.2317, "hours_q1": 43.26, "hours_q3": 72.56,
            "hours_lower_bound": -0.71, "hours_upper_bound": 116.52, "hours_outlier_count": 1,
            "declared_value_mean_inr": 1533.05, "declared_value_median_inr": 1121.5,
            "log_declared_value_mean": 7.028, "log_declared_value_median": 7.022,
        }
        actual = read("part_a_checks").set_index("metric").value.to_dict()
        self.assertEqual(set(actual), set(expected))
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(actual[metric], value, delta=0.01 if not float(value).is_integer() else 0)
        self.assertEqual(len(self.table), 1200)
        self.assertTrue(self.table.shipment_id.is_unique)
        self.assertFalse(self.table[FEATURES + ["delivery_hours", "is_delayed"]].isna().any().any())
        self.assertTrue(self.table.is_delayed.eq(self.table.delivery_hours > self.table.promised_hours).all())
        groups = read("part_a_delay_by_group")
        self.assertEqual(groups.shipments.tolist(), [315, 243, 642, 400, 400, 400, 274, 563, 275, 88])
        self.assertEqual(groups.delayed.tolist(), [27, 119, 132, 16, 86, 176, 6, 77, 120, 75])
        np.testing.assert_allclose(groups.delay_rate, groups.delayed / groups.shipments, atol=0.00005)
        correlations = read("part_a_correlations").set_index("feature")
        for a, b, expected_corr in [("delivery_hours", "promised_hours", 0.837),
                                     ("delivery_hours", "distance_km", 0.401),
                                     ("is_delayed", "distance_km", 0.546)]:
            self.assertAlmostEqual(correlations.loc[a, b], expected_corr, delta=0.01)

    def test_regression_acceptance(self):
        checks = read("part_b_checks").set_index("metric").value
        self.assertEqual(checks[["train_rows", "val_rows", "test_rows"]].tolist(), [840, 180, 180])
        self.assertLess(checks.train_scaled_max_abs_mean, 1e-9)
        self.assertGreater(checks.test_scaled_max_abs_mean, 0.01)
        metrics = read("part_b_metrics")
        self.assertEqual(len(metrics), 6)
        self.assertEqual(metrics.split.tolist(), ["val"] * 5 + ["test"])
        np.testing.assert_allclose(metrics.RMSE, np.sqrt(metrics.MSE), atol=0.001)
        validation = metrics[metrics.split.eq("val")].set_index("model")
        self.assertTrue(-0.1 <= validation.loc["mean_baseline", "R2"] <= 0.02)
        for name, minimum in [("linear_regression", 0.88), ("gradient_boosting", 0.88),
                              ("random_forest", 0.86), ("decision_tree", 0.82)]:
            self.assertGreaterEqual(validation.loc[name, "R2"], minimum)
        self.assertEqual(metrics.iloc[-1].model, validation.drop(index="mean_baseline").RMSE.idxmin())
        self.assertGreaterEqual(metrics.iloc[-1].R2, 0.88)
        depths = read("part_b_tree_depth")
        self.assertEqual(len(depths), 8)
        self.assertTrue((depths.train_rmse.diff().dropna() <= 0).all())
        self.assertLessEqual(depths.iloc[0].val_rmse - depths.iloc[0].train_rmse, 2.5)
        self.assertGreaterEqual(depths.iloc[-1].val_rmse - depths.iloc[-1].train_rmse, 5)
        self.assertGreaterEqual(depths.iloc[0].val_rmse - depths.val_rmse.min(), 4)
        self.assertAlmostEqual(validation.loc["decision_tree", "RMSE"], depths.val_rmse.min(), delta=0.001)
        rates = read("part_b_gb_learning_rate").set_index("learning_rate")
        self.assertGreater(rates.loc[0.1, "val_r2"], rates.loc[1.0, "val_r2"])
        self.assertGreater(rates.loc[1.0, "val_r2"], rates.loc[0.01, "val_r2"])
        self.assertGreaterEqual(rates.loc[0.1, "val_r2"], 0.9)
        self.assertLessEqual(rates.loc[0.01, "val_r2"], 0.8)
        cv = read("part_b_cv")
        self.assertEqual(len(cv), 2)
        self.assertTrue(cv.mean_r2.ge(0.9).all() and cv.std_r2.le(0.03).all())

    def test_classification_acceptance_and_confusion_matrices(self):
        x_train, x_val, x_test, y_train, y_val, y_test = split_classification(self.table)
        self.assertEqual([len(y_train), len(y_val), len(y_test)], [840, 180, 180])
        self.assertEqual([y_train.sum(), y_val.sum(), y_test.sum()], [195, 42, 41])
        self.assertFalse(set(x_train.index) & set(x_val.index) or set(x_train.index) & set(x_test.index)
                         or set(x_val.index) & set(x_test.index))
        models = read("part_c_models")
        self.assertEqual(len(models), 4)
        self.assertEqual(models.iloc[0][["TN", "FP", "FN", "TP"]].tolist(), [138, 0, 42, 0])
        for frame in [models, read("part_c_thresholds"), read("part_c_final_test")]:
            for _, row in frame.iterrows():
                self.assertEqual(row.TN + row.FP + row.FN + row.TP, 180)
                self.assertAlmostEqual(row.accuracy, (row.TN + row.TP) / 180, delta=0.00005)
                self.assertAlmostEqual(row.precision, row.TP / max(row.TP + row.FP, 1), delta=0.00005)
                self.assertAlmostEqual(row.recall, row.TP / max(row.TP + row.FN, 1), delta=0.00005)
                self.assertAlmostEqual(row.f1, 2 * row.TP / max(2 * row.TP + row.FP + row.FN, 1), delta=0.00005)
        self.assertTrue(models.iloc[1:].accuracy.gt(models.iloc[0].accuracy).all())
        self.assertTrue(models.iloc[1:].f1.ge(0.6).all())
        scores = read("part_c_feature_scores")
        self.assertEqual(scores.feature.tolist(), FEATURES)
        self.assertEqual(set(scores[scores.f_rank.le(7)].feature),
                         {"distance_km", "hub_tier", "hub_daily_capacity", "n_stops", "promised_hours", "rain_flag", "sale_event_flag"})
        noise = scores[scores.feature.isin(NOISE_COLUMNS)]
        self.assertTrue(noise.f_rank.ge(8).all() and noise.f_score.lt(10).all())
        self.assertGreater(scores[scores.f_rank.le(7)].f_score.min(), 30)
        self.assertTrue(0.15 <= noise.rf_importance.sum() <= 0.30)
        self.assertEqual(read("part_c_selectk").k.tolist(), [5, 7, 10, 12, 17])
        tuning = read("part_c_tuning")
        self.assertEqual(tuning.candidates.tolist(), [27, 9, 1])
        self.assertEqual(tuning.total_fits.tolist(), [135, 45, 1])
        self.assertTrue(tuning.iloc[:2].best_cv_accuracy.between(0.86, 0.93).all())
        self.assertGreaterEqual(tuning.iloc[0].best_cv_accuracy, tuning.iloc[1].best_cv_accuracy)
        self.assertTrue(tuning.val_f1.ge(0.65).all())
        priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
        winner = min(tuning.to_dict("records"), key=lambda row: (-row["val_f1"], priority[row["search"]]))
        config = read("part_c_final_config").iloc[0]
        self.assertEqual(config.chosen_from, winner["search"])
        thresholds = read("part_c_thresholds")
        self.assertEqual(thresholds.threshold.tolist(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertTrue(thresholds.recall.diff().dropna().le(0).all())
        eligible = thresholds[thresholds.recall.ge(0.8)]
        self.assertEqual(config.threshold, eligible.threshold.max() if len(eligible) else 0.1)
        final = read("part_c_final_test").iloc[0]
        self.assertEqual(final.threshold, config.threshold)
        self.assertTrue(final.f1 >= 0.6 and final.recall >= 0.55 and final.precision >= 0.45)
        model = joblib.load("models/delay_model.pkl")
        predicted = model.predict_proba(x_test)[:, 1] >= config.threshold
        for name, value in classification_metrics(y_test, predicted).items():
            self.assertAlmostEqual(final[name], value, delta=0.00005)
        params = {"n_estimators": int(config.n_estimators),
                  "max_depth": None if config.max_depth == "unlimited" else int(config.max_depth),
                  "min_samples_leaf": int(config.min_samples_leaf)}
        rebuilt = make_classifier(params).fit(x_train, y_train)
        np.testing.assert_array_equal(model.predict_proba(x_test), rebuilt.predict_proba(x_test))

    def test_hub_segmentation_acceptance(self):
        profile = read("hub_profile").set_index("hub_id")
        self.assertEqual(profile.index.tolist(), sorted(self.hubs.hub_id))
        self.assertTrue(profile.shipments.eq(50).all())
        for hub, expected in {"HUB01": [45.685, 0.02, 185.061, 54.70],
                              "HUB13": [66.074, 0.46, 473.320, 83.92],
                              "HUB17": [60.851, 0.46, 443.285, 84.18]}.items():
            np.testing.assert_allclose(profile.loc[hub].iloc[1:].to_numpy(float), expected, atol=0.01)
        separation = read("part_d_separation")
        self.assertEqual(separation.cluster_sizes.tolist(), ["15-9", "9-8-7", "8-7-5-4", "8-7-4-3-2"])
        np.testing.assert_allclose(separation.across_over_within, [2.894, 5.393, 6.305, 6.887], atol=0.01)
        segments = read("segment_profile")
        self.assertEqual(segments.segment.tolist(), ["Standard", "Watch", "Escalate"])
        self.assertEqual(segments.hubs.tolist(), [7, 8, 9])
        self.assertTrue(segments.delay_rate.diff().dropna().gt(0).all())
        mismatch = read("tier_vs_segment_mismatch").set_index("hub_id")
        self.assertEqual(mismatch.segment.to_dict(), {"HUB07": "Watch", "HUB13": "Escalate", "HUB15": "Escalate", "HUB22": "Watch"})
        reload_checks = read("part_d_reload_check").set_index("metric").value
        self.assertEqual(reload_checks.test_rows_scored, 180)
        self.assertLessEqual(reload_checks.reload_max_abs_probability_diff, 1e-9)

    def test_actions_probabilities_and_determinism(self):
        path = Path("outputs/new_shipment_actions.csv")
        before = path.read_bytes()
        summary_path = Path("outputs/new_shipment_action_summary.csv")
        summary_before = summary_path.read_bytes()
        for _ in range(2):
            subprocess.run([sys.executable, "predict_delay.py"], check=True, capture_output=True, text=True)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(summary_before, summary_path.read_bytes())
        actions = pd.read_csv(path)
        raw = pd.read_csv("data/new_shipments.csv")
        self.assertEqual(actions.shipment_id.tolist(), raw.shipment_id.tolist())
        self.assertEqual(len(actions), 48)
        self.assertTrue(actions.shipment_id.is_unique)
        self.assertFalse(actions.isna().any().any())
        threshold = pd.read_csv("models/delay_threshold.csv").threshold.iloc[0]
        np.testing.assert_array_equal(actions.flagged, actions.delay_probability.ge(threshold).astype(int))
        expected = np.where(actions.flagged.eq(0), "No action", np.where(
            actions.segment.eq("Escalate"), "Escalate to hub manager", "Proactive customer call"))
        np.testing.assert_array_equal(actions.action, expected)
        self.assertEqual(pd.read_csv("models/feature_order.csv").feature.tolist(), FEATURES)
        medians = pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
        clean, _ = clean_shipments(raw, medians=medians)
        features = build_features(clean, self.hubs)
        model = joblib.load("models/delay_model.pkl")
        np.testing.assert_allclose(actions.delay_probability, model.predict_proba(features[FEATURES])[:, 1], atol=0.00005)

    def test_artifact_contracts_and_readme_links(self):
        required = [
            "data/hubs.csv", "data/raw_shipments.csv", "data/new_shipments.csv", "data/modelling_table.csv",
            "models/delay_model.pkl", "models/delay_threshold.csv", "models/feature_order.csv", "models/impute_values.csv",
            *["outputs/" + name + ".csv" for name in [
                "part_a_checks", "part_a_delay_by_group", "part_a_correlations", "part_b_checks", "part_b_metrics",
                "part_b_tree_depth", "part_b_gb_learning_rate", "part_b_cv", "part_c_models", "part_c_feature_scores",
                "part_c_selectk", "part_c_tuning", "part_c_thresholds", "part_c_final_test", "part_c_final_config",
                "hub_profile", "part_d_separation", "hub_segments", "segment_profile", "tier_vs_segment_mismatch",
                "part_d_reload_check", "new_shipment_actions",
            ]],
        ]
        self.assertTrue(all(Path(path).is_file() for path in required))
        for path in [*Path("outputs").glob("*.csv"), *Path("models").glob("*.csv")]:
            self.assertFalse(any(column.startswith("Unnamed:") for column in pd.read_csv(path).columns))
            for decimal in re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", path.read_text()):
                self.assertEqual(len(decimal.split(".")[1]), 4, (path, decimal))
        requirements = dict(line.split("==") for line in Path("requirements.txt").read_text().splitlines())
        self.assertEqual(set(requirements), {"pandas", "numpy", "scikit-learn", "joblib"})
        for package, version in requirements.items():
            self.assertEqual(importlib.metadata.version(package), version)
        readme = Path("README.md").read_text()
        for target in re.findall(r"\]\(([^)]+)\)", readme):
            self.assertTrue(Path(target).is_file(), target)
        self.assertIn(f"**{read('part_c_checks').set_index('metric').loc['noise_importance_share', 'value']:.4f}**", readme)


if __name__ == "__main__":
    unittest.main()
