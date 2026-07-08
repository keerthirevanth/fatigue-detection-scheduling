"""
Stage 4 & 5 — training, tuning, and HONEST evaluation.

Key properties (the things the original prototype lacked):
  * Speaker-independent CV via StratifiedGroupKFold — no speaker appears in both
    train and test, so metrics reflect generalisation to *new* workers.
  * Scaler is fit inside each fold (in a Pipeline) — never on the test fold.
  * Metrics reported as cross-val mean ± std, not a single lucky split.
  * Optional randomized hyper-parameter search, also group-aware.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, classification_report

from .config import RANDOM_STATE, LABEL_TO_INT

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FATIGUED = LABEL_TO_INT["fatigued"]


# --------------------------------------------------------------------- models
def get_model_specs():
    """
    name -> (estimator, param_distributions_for_randomized_search)
    Estimators are the classifier step only; a StandardScaler is prepended
    in a Pipeline at fit time. Param keys are prefixed 'clf__' to match.
    """
    specs = {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                   random_state=RANDOM_STATE),
            {"clf__n_estimators": [200, 300, 500, 800],
             "clf__max_depth": [None, 6, 10, 20],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", "log2", 0.5]},
        ),
        "SVM": (
            SVC(kernel="rbf", class_weight="balanced", probability=True,
                random_state=RANDOM_STATE),
            {"clf__C": [0.5, 1, 2, 5, 10],
             "clf__gamma": ["scale", "auto", 0.01, 0.1]},
        ),
        "MLP": (
            MLPClassifier(max_iter=800, early_stopping=True,
                          random_state=RANDOM_STATE),
            {"clf__hidden_layer_sizes": [(128, 64), (256, 128), (128,), (64, 32)],
             "clf__alpha": [1e-4, 1e-3, 1e-2],
             "clf__learning_rate_init": [1e-3, 5e-4]},
        ),
    }
    if HAS_XGB:
        specs["XGBoost"] = (
            XGBClassifier(objective="multi:softprob", num_class=3,
                          eval_metric="mlogloss", tree_method="hist",
                          n_jobs=-1, random_state=RANDOM_STATE),  # no use_label_encoder
            {"clf__n_estimators": [300, 400, 600],
             "clf__max_depth": [3, 4, 6, 8],
             "clf__learning_rate": [0.03, 0.05, 0.1],
             "clf__subsample": [0.8, 1.0],
             "clf__colsample_bytree": [0.8, 1.0]},
        )
    return specs


def _pipe(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


# --------------------------------------------------------------------- CV
def cross_validate_models(X, y, groups, n_splits=5, verbose=True):
    """
    Run speaker-independent StratifiedGroupKFold CV for every model.

    Returns a dict:
        name -> {
            "f1_mean", "f1_std", "recall_fat_mean", "recall_fat_std",
            "y_true", "y_pred"   (out-of-fold, aggregated over all folds)
        }
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    results = {}

    for name, (estimator, _) in get_model_specs().items():
        fold_f1, fold_rec = [], []
        oof_true, oof_pred = [], []
        for tr, te in cv.split(X, y, groups):
            model = _pipe(estimator)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            fold_f1.append(f1_score(y[te], pred, average="macro"))
            fold_rec.append(recall_score(y[te], pred, labels=[FATIGUED],
                                         average="macro", zero_division=0))
            oof_true.extend(y[te]); oof_pred.extend(pred)

        results[name] = {
            "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
            "recall_fat_mean": float(np.mean(fold_rec)),
            "recall_fat_std": float(np.std(fold_rec)),
            "y_true": np.array(oof_true), "y_pred": np.array(oof_pred),
        }
        if verbose:
            r = results[name]
            print(f"  {name:13s}  F1={r['f1_mean']:.3f}±{r['f1_std']:.3f}   "
                  f"Recall(fatigued)={r['recall_fat_mean']:.3f}±{r['recall_fat_std']:.3f}")
    return results


def select_best(results):
    """Prioritise recall on 'fatigued' (missing a tired worker is the costly error),
    break ties with macro-F1."""
    return max(results, key=lambda k: (results[k]["recall_fat_mean"],
                                       results[k]["f1_mean"]))


# --------------------------------------------------------------------- tuning
def tune_model(name, X, y, groups, n_iter=25, n_splits=5):
    """Group-aware randomized search for one model. Returns the fitted best Pipeline."""
    estimator, param_dist = get_model_specs()[name]
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        _pipe(estimator), param_distributions=param_dist, n_iter=n_iter,
        scoring="f1_macro", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    print(f"  Best {name} params: {search.best_params_}")
    print(f"  Best CV f1_macro : {search.best_score_:.3f}")
    return search.best_estimator_


def fit_final(estimator_or_pipe, X, y):
    """Fit the chosen model on ALL delta samples for saving/deployment.
    Wraps a bare estimator in a scaler Pipeline if needed."""
    model = estimator_or_pipe
    if not isinstance(model, Pipeline):
        model = _pipe(model)
    model.fit(X, y)
    return model


def oof_report(results, name):
    """Pretty classification report from a model's out-of-fold predictions."""
    r = results[name]
    return classification_report(r["y_true"], r["y_pred"],
                                 target_names=list(LABEL_TO_INT.keys()),
                                 zero_division=0)
