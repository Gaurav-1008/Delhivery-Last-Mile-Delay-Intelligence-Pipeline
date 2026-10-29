# Delhivery Last-Mile Delay Intelligence Pipeline

A connected, offline capstone pipeline that estimates delivery hours, identifies parcels likely to miss their promise, segments hubs into operating playbooks, and produces dispatch actions. All shipment data is synthetic; this project uses no real Delhivery records.

## How to run

Run the following commands **from the repository root**, in this order. The project was executed with Python 3.11. The four direct dependencies are pinned to the versions used in [requirements.txt](requirements.txt).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
python -m unittest discover -s tests -v
```

The generator prints `24 1224 48`. The data files are already included, and the commands regenerate the data, every result CSV, and the packaged classifier. Nothing needs an account or network access after dependencies are installed. The provided generator was transcribed from the supplied screenshots with its constants and logic preserved; byte-for-byte identity with the original source cannot be established from screenshots alone. Its output passes the supplied numerical references.

`python package_model.py` regenerates `models/delay_model.pkl` from the chosen configuration in [part_c_final_config.csv](outputs/part_c_final_config.csv). It rebuilds the same stratified split and fits **only the training rows**. The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; when installed versions differ, the grader should rerun `package_model.py` instead of loading the committed pickle.

For another batch with the same raw schema, replace `data/new_shipments.csv` and run `python predict_delay.py`. The result is [new_shipment_actions.csv](outputs/new_shipment_actions.csv), in input file order. Exact duplicate input rows are removed by the shared cleaning function; conflicting repeated shipment IDs, unknown hubs or services, invalid amounts, and missing model features fail explicitly.

All CSVs in `outputs/` and `models/` omit the DataFrame index and serialize decimals to four places. The historical modelling table keeps its computed precision so writing it does not change model inputs. Correlation row labels are stored explicitly as the first column.

The main files and their connections are:

```text
generate_data.py
  -> data/hubs.csv, raw_shipments.csv, new_shipments.csv
src/shipment_prep.py: clean_shipments + build_features + pinned constants
  -> part_a_clean_eda.py -> data/modelling_table.csv + Part A evidence
       -> part_b_regression.py -> regression comparisons
       -> part_c_classification.py -> chosen classifier configuration + threshold
       -> part_d_segments.py -> hub profiles + segments
       -> package_model.py -> models/ + reload check
models/ + hub_segments.csv + new_shipments.csv
  -> predict_delay.py -> new_shipment_actions.csv
tests/test_pipeline.py -> input validation, acceptance checks, reproducible scoring
```

The model receives the pinned feature list in its original order, including the bookkeeping columns. The classifier's saved `SelectKBest` step selects the seven training-derived signal features internally. The scorer imports the same preprocessing functions used by Part A and uses stored historical medians; it never fits preprocessing on the new batch.

For submission, publish this repository on the single branch `main` and make `main` the default branch. Submit that repository's public root URL or its `/tree/main` URL. No screenshots, charts, PDFs, or slides are needed.

## Data findings

All figures below were computed from repository data rather than copied from reference answers.

1. **Historical data quality and overall delay.** The raw file has 1,224 rows; exact deduplication removes 24, leaving 1,200 unique shipments. There are 36 missing distances, 24 originally missing weights, and 5 additional weight sentinels. The sentinels are replaced with missing values before imputation. The hub merge retains 1,200 rows, and the feature and target columns contain zero nulls. There are 278 delayed parcels, a delay share of 0.2317. See [part_a_checks.csv](outputs/part_a_checks.csv).

2. **Express is delayed most often.** Its delay share is 0.4897, compared with 0.2056 for Standard and 0.0857 for Economy. Express accounts for 119 delayed parcels among 243 shipments; the tight promise matters more than absolute delivery duration. These are associations in synthetic data, not measured causal effects. See [part_a_delay_by_group.csv](outputs/part_a_delay_by_group.csv).

3. **Tier and distance show large differences.** Tier 1, Tier 2, and Tier 3 delay shares are 0.0400, 0.2150, and 0.4400. Delay shares across the right-inclusive distance bands are:

   | Distance band (km) | Shipments | Delayed | Delay share |
   |---|---:|---:|---:|
   | 0–200 | 274 | 6 | 0.0219 |
   | 201–400 | 563 | 77 | 0.1368 |
   | 401–600 | 275 | 120 | 0.4364 |
   | 601+ | 88 | 75 | 0.8523 |

   The actual bin edges are `0, 200, 400, 600, 2000`; the labels follow the brief, so a fractional distance above 200 belongs to the band labelled 201–400. See [part_a_delay_by_group.csv](outputs/part_a_delay_by_group.csv).

4. **The strongest correlations with delivery hours**, ranked by absolute value and excluding delivery hours itself, are `promised_hours` **+0.8372**, `distance_km` **+0.4013**, and `n_stops` **+0.3247**. The promise is a dispatch-time service attribute. A high correlation with delivery hours does not mean it is an observed delivery outcome. See [part_a_correlations.csv](outputs/part_a_correlations.csv).

5. **Log transformation reduces declared-value skew.** After median filling, declared value has mean **INR 1,533.0542** and median **INR 1,121.5000**; its natural log has mean **7.0277** and median **7.0224**. The closer log-scale mean and median indicate less right skew, while retaining relative value differences. See [part_a_checks.csv](outputs/part_a_checks.csv).

The delivery-hours quartiles are 43.2550 and 72.5625. The unrounded quartiles produce IQR bounds of -0.7063 and 116.5238 hours, with one outlier. That observed outcome is retained: only the documented weight scanner faults are treated as invalid measurements. Service normalization reduces distinct spellings from 12 to 3. Imputation medians are 314.9500 km, 1.8300 kg, and INR 1,121.5000, stored in [impute_values.csv](models/impute_values.csv). Per the brief, these target-independent medians use the full deduplicated historical table; scalers and selectors are fitted on training data only.

## Model results (regression)

Part B uses an unstratified two-stage split with `random_state=42`: 840 training, 180 validation, and 180 test rows. A scaler fitted only on training produces maximum absolute training-column mean 0.0000 after serialization; the corresponding test value is 0.2187. The linear model keeps the scaler inside its pipeline, including during cross-validation. See [part_b_checks.csv](outputs/part_b_checks.csv).

| Model | Split | MAE (hours) | MSE | RMSE (hours) | R² |
|---|---|---:|---:|---:|---:|
| Mean baseline | Validation | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| Linear regression | Validation | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| Decision tree | Validation | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| Random forest | Validation | 4.2794 | 30.3153 | 5.5059 | 0.9129 |
| Gradient boosting | Validation | 3.9166 | 24.1722 | 4.9165 | 0.9306 |
| Linear regression | Test | 3.5671 | 20.4624 | 4.5235 | 0.9493 |

Source: [part_b_metrics.csv](outputs/part_b_metrics.csv). RMSE is computed as the square root of MSE before CSV rounding. The baseline's negative R² means that predicting the training mean performs slightly worse on validation than a constant equal to the validation mean. Linear regression has the lowest validation RMSE, so it is the required final model. It is evaluated once on the test split without refitting on validation.

The depth diagnostic shows **high bias at depth 1**: training RMSE is 11.6334 and validation RMSE is 12.1324; both are high and close. Depth 6 minimizes validation RMSE at 6.2796, with training RMSE 4.7114. At depths 8 and 12, training error continues falling while validation error rises. The unlimited tree has training RMSE 0.0000 and validation RMSE 7.8368, a clear high-variance pattern. Source: [part_b_tree_depth.csv](outputs/part_b_tree_depth.csv).

The learning-rate experiment gives:

| Learning rate | Validation RMSE | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

Source: [part_b_gb_learning_rate.csv](outputs/part_b_gb_learning_rate.csv). With the fixed estimator budget, 0.01 learns too slowly; 1.0 makes large residual corrections and loses validation accuracy through overshooting or overfitting. Rate 0.1 gives the best measured balance.

The two strongest validation candidates use the same shuffled five-fold `KFold` object on the concatenated training-plus-validation rows:

| Model | Mean CV R² | Standard deviation |
|---|---:|---:|
| Linear regression | 0.9261 | 0.0048 |
| Gradient boosting | 0.9353 | 0.0045 |

Source: [part_b_cv.csv](outputs/part_b_cv.csv). Cross-validation favors boosting while the held-out validation split narrowly favors linear regression. I would ship the prescribed linear-regression winner for this release: it follows the declared selection rule and offers a simpler model with slightly lower validation RMSE. The CV advantage makes boosting worth a later comparison on new, time-separated data; the inspected test split must not be reused to switch models.

## Model results (classification)

Part C stratifies both split calls. The split sizes remain 840/180/180, with 195/42/41 delayed parcels respectively. See [part_c_checks.csv](outputs/part_c_checks.csv).

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Blind: all on time | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision tree, depth 6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Default forest, all features | 132 | 6 | 10 | 32 | 0.9111 | 0.8421 | 0.7619 | 0.8000 |

Source: [part_c_models.csv](outputs/part_c_models.csv). The blind model's apparently respectable accuracy hides zero delayed parcels caught. Precision, recall, and F1 reveal that it is useless for proactive intervention. The decision-tree depth is selected by highest validation F1, with smaller depth breaking a tie.

The five bookkeeping columns collectively receive **0.2252** of the all-feature forest's importance, despite all being outside the top seven by F-score and having F-scores below 10. This share is summed before rounding individual importances. Impurity importance can allocate substantial weight to random, high-cardinality columns; it is not evidence of useful out-of-sample signal. The selected signal columns, ranked by F-score, are distance, hub tier, capacity, stops, promise, rain, and sale flag. Sources: [part_c_checks.csv](outputs/part_c_checks.csv) and [part_c_feature_scores.csv](outputs/part_c_feature_scores.csv).

| Selected k | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.8140 |
| 12 | 0.8049 |
| 17 | 0.8000 |

Source: [part_c_selectk.csv](outputs/part_c_selectk.csv). Five features discard useful signal; seven perform best on this split; adding weaker features reduces F1. The final selector is fixed to seven, as required, and fitted on training rows.

| Search | Candidates | CV folds | CV fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Grid | 27 | 5 | 135 | 0.8952 | 0.7750 |
| Randomized | 9 | 5 | 45 | 0.8905 | 0.7901 |
| Untuned, same seven features | 1 | 0 | 1 | — | 0.8333 |

Source: [part_c_tuning.csv](outputs/part_c_tuning.csv). Search fit counts exclude the searches' final refits. The grid selects depth 6, leaf minimum 1, and 100 trees; randomized search selects depth 6, leaf minimum 1, and 50 trees. The untuned forest wins validation F1, so the final pipeline uses 100 trees, unlimited depth, and leaf minimum 1. Tuning improves its own cross-validation accuracy objective but does not beat the untuned candidate on validation F1. With only 180 validation parcels, these rankings are estimates rather than a guarantee that tuning is unhelpful generally.

**The alert threshold is 0.5**, the highest tested threshold with validation recall at least 0.80. At 0.5, precision is 0.7955 and recall is 0.8333; at 0.6, precision rises to 0.8889 but recall falls to 0.7619 and fails the rule. A lower threshold, 0.4, catches more delays with recall 0.8810 but reduces precision to 0.7400. The explicit threshold grid uses exact decimal literals and flags probabilities **greater than or equal to** the threshold. Source: [part_c_thresholds.csv](outputs/part_c_thresholds.csv).

The selection tables use the estimator's default `predict`, which assigns an exact probability tie to class 0. The alert rule explicitly includes ties at the threshold. Therefore the seven-feature candidate's selection F1 of 0.8333 and its alert-rule F1 of 0.8140 at 0.5 differ; the threshold table is the source for operational alert metrics.

The single final test evaluation at the selected threshold produces **precision 0.6889, recall 0.7561, and F1 0.7209**, with TN 125, FP 14, FN 10, and TP 31. Recall is below validation's 0.8333: the validation constraint is not a guarantee on future data, and this test contains only 41 delayed parcels. Neither model nor threshold is retuned after inspecting the test. Sources: [part_c_final_test.csv](outputs/part_c_final_test.csv) and [part_c_final_config.csv](outputs/part_c_final_config.csv).

## Operations recommendation

Use segment behavior to choose the hub playbook and the parcel classifier to decide which individual parcels need intervention. A hub in the Escalate segment does not automatically cause every parcel to be flagged.

| Segment | Hubs | Mean delivery hours | Delay share | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.1550 | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.4600 | 470.4757 | 84.3444 |

Source: [segment_profile.csv](outputs/segment_profile.csv). These are means of hub-level measurements. Every hub has 50 historical parcels. Hub IDs enter scaling and clustering in ascending order, and the four profile measurements are standardized before K-Means. Segment names follow increasing mean delay share rather than arbitrary cluster numbers.

Prioritize capacity review and manager follow-up at these **Escalate** hubs:

| Hub | City | Delay share |
|---|---|---:|
| HUB13 | Nagpur | 0.4600 |
| HUB15 | Kochi | 0.3600 |
| HUB17 | Patna | 0.4600 |
| HUB18 | Ranchi | 0.3800 |
| HUB19 | Guwahati | 0.6200 |
| HUB20 | Raipur | 0.4800 |
| HUB21 | Bhubaneswar | 0.4000 |
| HUB23 | Siliguri | 0.4400 |
| HUB24 | Agartala | 0.5400 |

Sources: [hub_segments.csv](outputs/hub_segments.csv) and [hub_profile.csv](outputs/hub_profile.csv). Use routine monitoring for Standard hubs; review queue pressure and route capacity for Watch hubs; use manager escalation for flagged parcels at Escalate hubs.

All tier-versus-behavior mismatches are:

| Hub | City | Labelled tier | Behavioral segment | Operational implication |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Apply the Watch review process despite its Tier 1 label. |
| HUB13 | Nagpur | 2 | Escalate | Use manager escalation for its flagged parcels. |
| HUB15 | Kochi | 2 | Escalate | Use manager escalation for its flagged parcels. |
| HUB22 | Dehradun | 3 | Watch | Use the Watch playbook instead of escalating solely on tier. |

Source: [tier_vs_segment_mismatch.csv](outputs/tier_vs_segment_mismatch.csv). Tier remains an administrative label and a model input; observed segment behavior determines the playbook. A tier mismatch does not establish the cause of a hub's delays.

At threshold 0.5, the test precision implies about **6.9 of every 10 flagged parcels** are actually delayed (31 true positives among 45 flags). Test recall implies about **7.6 of every 10 delayed parcels** are caught (31 among 41 delays). These are test estimates, not outcomes observed for the new batch. Source: [part_c_final_test.csv](outputs/part_c_final_test.csv).

The 48 new parcels produce **8 flags: 6 hub-manager escalations and 2 proactive customer calls**; 40 receive no action. Escalate only when both the parcel is flagged and its hub segment is Escalate. Other flags receive a customer call. Sources: [new_shipment_action_summary.csv](outputs/new_shipment_action_summary.csv) and [new_shipment_actions.csv](outputs/new_shipment_actions.csv).

| K | Across/within distance ratio | Descending cluster sizes |
|---:|---:|---|
| 2 | 2.8940 | 15–9 |
| 3 | 5.3928 | 9–8–7 |
| 4 | 6.3047 | 8–7–5–4 |
| 5 | 6.8866 | 8–7–4–3–2 |

Source: [part_d_separation.csv](outputs/part_d_separation.csv). Ratios use mean Euclidean distances over all unordered hub pairs on the same scaled profile matrix, separating cross-cluster and within-cluster pairs. Keep **K = 3** because operations runs three playbooks and each segment has at least six hubs. Although the ratio rises at K = 4 and K = 5, both create segments smaller than six hubs, too small for separate playbooks. The mandated ascending hub order, `random_state=42`, and `n_init=10` reproduce the reference solutions; finer partitions can settle into different local optima under other orders or seeds.

## Production plan

These are **proposed deployment controls**, not claims that production systems or monitoring already exist.

| Readiness characteristic | Concrete repository support | Deployment step |
|---|---|---|
| Accuracy | Validation selection, threshold table, and isolated final-test metrics | Review precision and recall on matured parcel outcomes before promotion. |
| Reliability | Input checks, finite feature validation, and unique-key merge validation | Reject an invalid batch before writing a new action file; record the failure and retry after correction. |
| Reproducibility | Pinned packages, fixed seeds, data generator, and rebuild scripts | Keep the environment and model/configuration artifacts together under a release commit. |
| Performance | `predict_delay.py` loads one model and predicts the batch together | Measure scoring duration on representative batch sizes before setting a service target. |
| Maintainability | Shared preprocessing in `src/shipment_prep.py` and acceptance tests | Require the tests and a clean rebuild for changes to features or model settings. |
| Consistency | Stored feature order, historical imputation medians, and serialized selector | Use the same release's model, medians, threshold, and segment files for every batch. |
| Monitorability | Action-summary counts and recorded confusion-matrix metrics | Persist release ID, score, action, and eventual outcome for cohort reports. |

Proposed weekly monitoring thresholds:

| Quantity | Numeric trigger | Response |
|---|---|---|
| Recall on matured outcomes | Below 0.80 for two consecutive comparable weekly cohorts, each with at least 50 delayed parcels | Review missed cases, threshold, and feature drift; widen the evaluation window when there are too few outcomes. |
| Delayed parcel share | More than 0.05 absolute change from the matching service/segment/event reference cohort for two weeks | Investigate operations changes and assess model recalibration. |
| Flagged share | More than 0.05 absolute change from a comparable production reference cohort for two weeks | Check capacity for calls, score distribution, missingness, and input consistency. |
| Missingness by imputed feature | Above 0.10 of a weekly cohort | Investigate export or scanner quality before trusting imputed scores. |
| Distance distribution | Median shifts by more than 0.20 of the matching reference median for two weeks | Review lane mix and whether the learned segments and classifier remain representative. |

The operational recall target remains 0.80; the observed test recall of 0.7561 would call for investigation before a real rollout. The alert threshold was not changed to conceal that result. Baselines for the proposed monitoring system must come from a production shadow period, since a small synthetic new batch cannot establish normal live traffic.

Proposed schedule: monitor weekly, review threshold calibration monthly on newly matured outcomes, and retrain quarterly or earlier after a sustained trigger. Use a forward-time holdout for release approval and fit replacement preprocessing only within the training window. Recompute hub profiles when sufficient recent outcomes are available, then review playbook changes with operations.

The **15–28 September sale window** must be compared with sale-event cohorts for the same service and segment, not ordinary weeks. Keep `sale_event_flag` available at dispatch, evaluate sale and non-sale recall separately, and review sale performance after outcomes mature. Here, drift would mean changed lane distances, stop counts, service or hub mix, missingness, or a changed relationship between the sale flag and delays. A predictable sale-related increase is not automatically drift; a deviation from comparable sale behavior is the signal to investigate.

## Limitations

- The data and behavioral differences are synthetic. Matching the brief's reference numbers does not demonstrate performance on real logistics records.
- The classification test contains only 41 delayed parcels, so its recall estimate is uncertain. The observed 0.7561 recall is below the validation constraint.
- Feature count, candidate choice, and alert threshold are assessed on one validation split. Repeated use can overfit that split; later releases need new validation data.
- Random splits follow the assignment but do not simulate forward-time deployment or performance on unseen hubs. A real rollout needs temporal evaluation and, when relevant, hub-held-out evaluation.
- The target-independent medians intentionally use the whole historical table as prescribed. A production evaluation should fit medians inside its training window as well.
- The feature selector precedes the tuning searches as prescribed, so its labels include the searches' internal validation folds. Reported search CV accuracy is therefore conditional on that selector, rather than a fully nested estimate; the outer validation and test splits remain excluded from selector fitting.
- Probability scores are uncalibrated forest probabilities. The recall rule does not incorporate a measured intervention cost or call-capacity budget, and its validation guarantee does not extend to future batches.
- Hub clustering uses historical delivery outcomes and is appropriate for the existing hubs only. Profiles can change with service mix; the scorer rejects hubs without a known profile and segment.
- Synthetic timestamps are local, timezone-naive dispatch times, and rain and sale indicators are assumed available at dispatch. A live system needs consistent timestamp semantics and reliable advance signals.
- The generator's source was supplied as screenshots; original-file byte identity remains unverified even though the generated data passes the numerical acceptance checks.

The acceptance tests verify preprocessing reuse, absence of outcome dependence in features, invalid-input failures, the brief's numeric checks and pass bands, confusion-matrix consistency, training-only model rebuilding, model reload equality, and byte-identical consecutive scoring outputs.
