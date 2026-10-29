"""Select a delay classifier and recall-constrained alert threshold on validation."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.reporting import classification_metrics, write_checks, write_csv
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def split_classification(table):
    x_train, x_temp, y_train, y_temp = train_test_split(
        table[FEATURES], table["is_delayed"], test_size=0.3, random_state=42,
        stratify=table["is_delayed"],
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_classifier(params):
    return Pipeline([("selector", SelectKBest(f_classif, k=7)),
                     ("model", RandomForestClassifier(random_state=42, **params))])


def main():
    table = pd.read_csv("data/modelling_table.csv")
    x_train, x_val, x_test, y_train, y_val, y_test = split_classification(table)
    assert [len(y_train), len(y_val), len(y_test)] == [840, 180, 180]
    assert [y_train.sum(), y_val.sum(), y_test.sum()] == [195, 42, 41]
    checks = {"train_rows": len(y_train), "val_rows": len(y_val), "test_rows": len(y_test),
              "train_delayed": y_train.sum(), "val_delayed": y_val.sum(), "test_delayed": y_test.sum()}
    rows = [{"model": "blind_all_on_time", **classification_metrics(y_val, np.zeros(len(y_val), dtype=int))}]
    tree_trials = []
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(x_train, y_train)
        tree_trials.append((f1_score(y_val, tree.predict(x_val)), depth, tree))
    _, depth, tree = max(tree_trials, key=lambda trial: (trial[0], -trial[1]))
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
        f"decision_tree_depth_{depth}": tree,
        "random_forest_default": forest,
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        rows.append({"model": name, **classification_metrics(y_val, model.predict(x_val))})
    write_csv(pd.DataFrame(rows), "outputs/part_c_models.csv")
    selector_all = SelectKBest(f_classif, k="all").fit(x_train, y_train)
    feature_scores = pd.DataFrame({
        "feature": FEATURES, "f_score": selector_all.scores_,
        "f_rank": pd.Series(selector_all.scores_).rank(ascending=False, method="first").astype(int),
        "rf_importance": forest.feature_importances_, "is_noise": [name in NOISE_COLUMNS for name in FEATURES],
    })
    write_csv(feature_scores, "outputs/part_c_feature_scores.csv")
    checks["noise_importance_share"] = feature_scores.loc[feature_scores["is_noise"], "rf_importance"].sum()
    write_checks(checks, "outputs/part_c_checks.csv")
    selectk = []
    for k in [5, 7, 10, 12, 17]:
        model = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                          ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        model.fit(x_train, y_train)
        selectk.append({"k": k, "val_f1": f1_score(y_val, model.predict(x_val))})
    write_csv(pd.DataFrame(selectk), "outputs/part_c_selectk.csv")
    selector = SelectKBest(f_classif, k=7).fit(x_train, y_train)
    train_selected, val_selected = selector.transform(x_train), selector.transform(x_val)
    grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
    searches = {
        "grid_search": GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5),
        "randomized_search": RandomizedSearchCV(
            RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42,
        ),
    }
    tuning, configurations = [], {}
    for name, search in searches.items():
        search.fit(train_selected, y_train)
        candidates = len(search.cv_results_["params"])
        configurations[name] = search.best_params_
        tuning.append({"search": name, "candidates": candidates, "cv_folds": 5, "total_fits": candidates * 5,
                       "best_params": json.dumps(search.best_params_, sort_keys=True),
                       "best_cv_accuracy": search.best_score_,
                       "val_f1": f1_score(y_val, search.predict(val_selected))})
    default = RandomForestClassifier(n_estimators=100, random_state=42).fit(train_selected, y_train)
    configurations["default_forest_same_7_features"] = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    tuning.append({"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
                   "total_fits": 1, "best_params": "defaults", "best_cv_accuracy": np.nan,
                   "val_f1": f1_score(y_val, default.predict(val_selected))})
    write_csv(pd.DataFrame(tuning), "outputs/part_c_tuning.csv")
    priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    chosen = min(tuning, key=lambda row: (-row["val_f1"], priority[row["search"]]))["search"]
    params = configurations[chosen]
    model = make_classifier(params).fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]
    thresholds = [{"threshold": t, **classification_metrics(y_val, probabilities >= t)}
                  for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    write_csv(pd.DataFrame(thresholds), "outputs/part_c_thresholds.csv")
    threshold = max((row["threshold"] for row in thresholds if row["recall"] >= 0.8), default=0.1)
    test = {"threshold": threshold, **classification_metrics(y_test, model.predict_proba(x_test)[:, 1] >= threshold)}
    write_csv(pd.DataFrame([test]), "outputs/part_c_final_test.csv")
    write_csv(pd.DataFrame([{
        "selector_k": 7, "n_estimators": params["n_estimators"],
        "max_depth": "unlimited" if params.get("max_depth") is None else params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold, "chosen_from": chosen,
    }]), "outputs/part_c_final_config.csv")
    print(f"Part C: {chosen}; threshold {threshold:.1f}; test F1 {test['f1']:.4f}")


if __name__ == "__main__":
    main()
