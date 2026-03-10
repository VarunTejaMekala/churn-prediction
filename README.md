# Changelog — Churn Prediction Project

All bugs fixed and improvements made vs. the original submission.

---

## `training_pipeline.py` (constants / `__init__.py`)
| # | Issue | Fix |
|---|---|---|
| 1 | `PIPELINE_NAME = "NetwrokSecurity"` — wrong domain name, typo | → `"ChurnPrediction"` |
| 2 | `DATA_INGESTION_COLLECTION_NAME = "NetworkData"` — wrong domain | → `"ChurnData"` |
| 3 | `DATA_INGESTION_DATABASE_NAME = "varuntejamekala"` — personal name hardcoded | → `"churn_db"` (overridable via env) |
| 4 | `DATA_INGESTION_TRAIN_TEST_SPLIT_RATION` — typo | → `DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO` |
| 5 | `MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD` — typo | → `MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD` |
| 6 | `DATA_TRANSFORMATION_IMPUTER_PARAMS` referenced KNNImputer but SimpleImputer was used | Removed unused constant |
| 7 | `CLASSIFICATION_THRESHOLD` not centralised | Added `CLASSIFICATION_THRESHOLD = 0.35` as constant |

---

## `config_entity.py`
| # | Issue | Fix |
|---|---|---|
| 8 | `print(training_pipeline.PIPELINE_NAME)` at module level | Removed — side-effect on import |
| 9 | Referenced `DATA_INGESTION_TRAIN_TEST_SPLIT_RATION` (typo) | Updated to corrected constant name |
| 10 | Referenced `MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD` (typo) | Updated to corrected constant name |

---

## `artifact_entity.py`
| # | Issue | Fix |
|---|---|---|
| 11 | `ClassificationMetricArtifact` missing `roc_auc_score` field | Added `roc_auc_score: float` |

---

## `utils.py`
| # | Issue | Fix |
|---|---|---|
| 12 | `read_yaml_file` opened file in `"rb"` (binary) mode — `yaml.safe_load` needs text | Changed to `"r"` |
| 13 | `evaluate_models` used `r2_score` — this is a **regression** metric. On binary classification it returns negative / meaningless values | Replaced with `f1_score` |
| 14 | `GridSearchCV` used default `cv=3` integer — does not preserve class balance | Replaced with `StratifiedKFold(n_splits=3, shuffle=True)` |
| 15 | `evaluate_models` returned `{name: float}` but `model_trainer.py` needed the fitted model | Return format changed to `{name: {"score", "model", "params"}}` |
| 16 | `evaluate_models` scored with `.predict()` — ignores probability threshold | Now uses `predict_proba` + configurable `threshold` |

---

## `data_ingestion.py`
| # | Issue | Fix |
|---|---|---|
| 17 | `train_test_split` had no `stratify` argument — churn minority class can be imbalanced across splits | Added `stratify=target` |

---

## `data_validation.py`
| # | Issue | Fix |
|---|---|---|
| 18 | `check_missing_values` raised `Exception` and crashed pipeline when missing values found — but missing values are handled downstream by `SimpleImputer` | Changed to `logging.warning` + return `False` |
| 19 | `check_duplicates` raised `Exception` on duplicates | Changed to `logging.warning` + return `False` |
| 20 | `detect_dataset_drift` iterated all columns including objects and silently skipped them without explanation | Now explicitly selects `select_dtypes(include="number")` |
| 21 | Drift failure stopped the pipeline entirely | Now logs a `WARNING` and continues (drift is informational) |

---

## `data_transformation.py`
| # | Issue | Fix |
|---|---|---|
| 22 | `tenure_group = pd.cut(..., labels=[...])` returns `pd.Categorical` — `OneHotEncoder` fails on this dtype silently | Added `.astype(str)` after `pd.cut` |
| 23 | `service_count` feature was not included in `numerical_columns` list in the preprocessor | Added to numeric pipeline columns |
| 24 | `service_count` feature was not engineered at all in original | Added to `add_engineered_features()` |

---

## `classification_report.py`
| # | Issue | Fix |
|---|---|---|
| 25 | `get_classification_score` had no `roc_auc` computation | Added `y_prob` parameter and `roc_auc_score` computation |
| 26 | No `zero_division=0` guard on f1/precision/recall | Added to avoid `UndefinedMetricWarning` |

---

## `estimator.py`
| # | Issue | Fix |
|---|---|---|
| 27 | Class named `NetworkModel` — wrong domain | Renamed to `ChurnModel` |
| 28 | `predict()` used `model.predict()` directly — ignores probability threshold | Now uses `predict_proba` + threshold |
| 29 | No `predict_proba()` method exposed | Added for downstream scoring |
| 30 | No input type guard — crashed if `np.ndarray` passed instead of `DataFrame` | Added `isinstance` check |

---

## `model_trainer.py`
| # | Issue | Fix |
|---|---|---|
| 31 | `track_mlflow()` called twice in sequence — second call starts a new run while the first may still be active | Each call now uses a distinct `run_name` in its own `with mlflow.start_run()` block |
| 32 | `evaluate_models` result treated as `{name: float}` but updated utils returns `{name: dict}` | Unpacking updated to use `result["score"]`, `result["model"]`, `result["params"]` |
| 33 | Overfitting threshold defined in constants but **never checked** | Added `_check_overfitting()` method that logs a warning when train/test F1 diff exceeds threshold |
| 34 | Expected score threshold defined but **never enforced** | Added explicit check — raises if best model F1 < `MODEL_TRAINER_EXPECTED_SCORE` |
| 35 | MLflow didn't log `roc_auc_score` | Added |
| 36 | `NetworkModel` import used — replaced with `ChurnModel` | Updated import |

---

## `push_data.py`
| # | Issue | Fix |
|---|---|---|
| 37 | Hardcoded absolute Windows path `E:\codes\...` | Replaced with `os.getenv("DATA_FILE_PATH", "data/Telco-Customer-Churn.csv")` |
| 38 | `NetworkDataExtract` — wrong domain name | Renamed to `ChurnDataExtract` |
| 39 | `insert__data_mongodb` — double underscore typo | Fixed to `insert_data_mongodb` |
| 40 | No check if file exists before reading | Added `os.path.exists` guard with helpful error message |
| 41 | `NaN` values not handled before MongoDB insert (PyMongo rejects `float('nan')`) | Added `df.where(pd.notnull(df), None)` |

---

## `main.py`
| # | Issue | Fix |
|---|---|---|
| 42 | `"Model Training sstared"` — typo in log message | Fixed to `"Model Training"` |
| 43 | No step separators in logs — hard to read pipeline progress | Added `=`-delimited step headers |

---

## New files added
| File | Purpose |
|---|---|
| `data_schema/schema.yaml` | Column schema for data validation (was referenced but missing) |
| `networksecurity/exception/exception.py` | Custom exception (was imported everywhere, now included) |
| `networksecurity/logging/logger.py` | Logger (was imported everywhere, now included) |
| `requirements.txt` | Full dependency list |
| `CHANGES.md` | This file |
