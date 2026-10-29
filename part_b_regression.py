"""Compare delivery-hour regressors using validation selection and one final test."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.reporting import regression_metrics, write_checks, write_csv
from src.shipment_prep import FEATURES


def main():
    table = pd.read_csv("data/modelling_table.csv")
    x_train, x_temp, y_train, y_temp = train_test_split(
        table[FEATURES], table["delivery_hours"], test_size=0.3, random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42,
    )
    scaler = StandardScaler().fit(x_train)
    write_checks({
        "train_rows": len(x_train), "val_rows": len(x_val), "test_rows": len(x_test),
        "train_scaled_max_abs_mean": np.abs(scaler.transform(x_train).mean(axis=0)).max(),
        "test_scaled_max_abs_mean": np.abs(scaler.transform(x_test).mean(axis=0)).max(),
    }, "outputs/part_b_checks.csv")
    metrics = [{"model": "mean_baseline", "split": "val", **regression_metrics(
        y_val, np.full(len(y_val), y_train.mean()),
    )}]
    depths = []
    trees = {}
    for depth in [1, 2, 3, 4, 6, 8, 12, None]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(x_train, y_train)
        trees[depth] = tree
        depths.append({"max_depth": "unlimited" if depth is None else depth,
                       "train_rmse": regression_metrics(y_train, tree.predict(x_train))["RMSE"],
                       "val_rmse": regression_metrics(y_val, tree.predict(x_val))["RMSE"]})
    best_depth = min(range(len(depths)), key=lambda i: depths[i]["val_rmse"])
    best_tree = trees[[1, 2, 3, 4, 6, 8, 12, None][best_depth]]
    write_csv(pd.DataFrame(depths), "outputs/part_b_tree_depth.csv")
    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": best_tree,
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics.append({"model": name, "split": "val", **regression_metrics(y_val, model.predict(x_val))})
    rates = []
    for rate in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42)
        model.fit(x_train, y_train)
        scores = regression_metrics(y_val, model.predict(x_val))
        rates.append({"learning_rate": rate, "val_rmse": scores["RMSE"], "val_r2": scores["R2"]})
    write_csv(pd.DataFrame(rates), "outputs/part_b_gb_learning_rate.csv")
    ranked = sorted(metrics[1:], key=lambda row: row["RMSE"])
    x_train_val, y_train_val = pd.concat([x_train, x_val]), pd.concat([y_train, y_val])
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = []
    for row in ranked[:2]:
        scores = cross_val_score(models[row["model"]], x_train_val, y_train_val, cv=folds)
        cv.append({"model": row["model"], "mean_r2": scores.mean(), "std_r2": scores.std()})
    write_csv(pd.DataFrame(cv), "outputs/part_b_cv.csv")
    winner = ranked[0]["model"]
    metrics.append({"model": winner, "split": "test", **regression_metrics(y_test, models[winner].predict(x_test))})
    write_csv(pd.DataFrame(metrics), "outputs/part_b_metrics.csv")
    print(f"Part B: validation winner {winner}; test R2 {metrics[-1]['R2']:.4f}")


if __name__ == "__main__":
    main()
