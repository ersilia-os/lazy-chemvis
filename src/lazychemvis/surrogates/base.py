"""
Shared training pipeline for the XGBoost surrogates.

The t-SNE and UMAP projections are non-parametric, so new molecules are placed by
a multi-output XGBoost regressor that maps ECFP fingerprints to the reference
coordinates. Both surrogates are trained the same way — Optuna hyperparameter
search → k-fold cross-validation → production model on all data — and differ only
in which projection they learn and, for UMAP, how the fingerprint rows are aligned
with the coordinates. That shared procedure lives here, once, so the two cannot
drift apart.

Subclasses set :attr:`surrogate_name`, :attr:`projection_name` and :attr:`label`,
implement :meth:`_load_targets`, and may override :meth:`_align_features`.
"""

import gc
import json
import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from ..featurizers.ecfp import ECFPFeaturizer
from ..helpers.live import LiveProgressBar
from ..helpers.logger import get_logger
from ..report import plots as report_plots
from ..report.perf import log_cv_results

logger = get_logger(__name__)

# Libraries above this size tune hyperparameters on a random subsample.
OPTUNA_SUBSAMPLE_ABOVE = 100_000
OPTUNA_SUBSAMPLE_SIZE = 400_000
OPTUNA_TRIALS = 30
OPTUNA_INNER_FOLDS = 3


class XGBSurrogate(object):
    """
    XGBoost surrogate mapping ECFP fingerprints to a projection's reference coordinates.

    Training follows Optimisation (Optuna) → Validation (CV) → Production (100% data):

    1. Validation is performed BEFORE training the final model.
    2. Cross-validation reports mean ± std across folds.
    3. Final production model is trained on 100% of data.

    The cross-validation is not nested: Optuna tunes the hyperparameters on the same
    molecules (or a subsample of them) before the folds are drawn, so the reported
    metrics are cross-validated scores after tuning and can be slightly optimistic.

    Parameters
    ----------
    dir_path : str
        Directory where artifacts are stored.
    evaluate : bool, default=True
        If True, perform cross-validation BEFORE training the final model.
    cv_folds : int, default=5
        Number of cross-validation folds.
        Set to 0 to skip validation (not recommended for production).
    optimize : bool, default=True
        If True, run Optuna hyperparameter search before training.
    random_state : int, default=42
        Seed used for the Optuna sampler, the Optuna subsampling draw, the
        cross-validation splits and XGBoost itself, so that reported metrics
        are reproducible across runs.
    """

    #: Output subdirectory, e.g. ``"tsne_surrogate"``.
    surrogate_name = None
    #: Projection whose coordinates are learned, e.g. ``"tsne"``.
    projection_name = None
    #: Human-readable projection name for progress output, e.g. ``"t-SNE"``.
    label = None

    def __init__(
        self,
        dir_path: str,
        evaluate: bool = True,
        cv_folds: int = 5,
        optimize: bool = True,
        random_state: int = 42,
    ):
        self.dir_path = os.path.abspath(dir_path)
        self.evaluate = evaluate
        self.cv_folds = cv_folds
        self.metrics = None
        self.optimize = optimize
        self.random_state = random_state

        self.best_params = {
            "n_estimators": 300,
            "max_depth": 9,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
            "random_state": random_state,
        }

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _load_targets(self):
        """
        Return the reference coordinates and the axis scaler of the projection.

        Returns
        -------
        y_coords : numpy.ndarray of shape (n_molecules, 2)
            The projection's reference coordinates.
        axis_scaler : sklearn.preprocessing.MinMaxScaler
            The scaler that mapped the projection onto [-1, 1].
        """
        raise NotImplementedError

    def _align_features(self, X):
        """
        Return the fingerprint rows that correspond to the target coordinates.

        The default is the identity; override it when the projection was fitted on a
        subset or reordering of the reference molecules.

        Parameters
        ----------
        X : numpy.ndarray
            The reference ECFP matrix, one row per validated reference molecule.

        Returns
        -------
        numpy.ndarray
            The rows of ``X`` aligned with the target coordinates.
        """
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train the surrogate XGBoost following the flow:
        Optimisation (Optuna) → Validation (CV) → Production (100% data).
        """
        # 1. DATA LOADING & ALIGNMENT
        logger.info(f"Loading ECFP features and {self.label} coordinates...")
        ecfp_feat = ECFPFeaturizer.load(dir_path=self.dir_path)
        y_coords, self.axis_scaler = self._load_targets()
        X = self._align_features(ecfp_feat.X)

        if X.shape[0] != y_coords.shape[0]:
            logger.error(f"Dimension mismatch: X={X.shape[0]}, Y={y_coords.shape[0]}")
            raise ValueError(
                f"Dimension mismatch: X={X.shape[0]}, Y={y_coords.shape[0]}"
            )

        # Release the featurizer wrapper — the arrays live on via X and y_coords
        del ecfp_feat
        gc.collect()

        # PHASE 0: HYPERPARAMETER OPTIMISATION
        if self.optimize:
            logger.info("Phase 0 — hyperparameter optimisation (Optuna)")
            if X.shape[0] > OPTUNA_SUBSAMPLE_ABOVE:
                n_optuna = min(OPTUNA_SUBSAMPLE_SIZE, X.shape[0])
                logger.info(
                    f"Large dataset detected — subsampling {n_optuna:,} molecules for Optuna."
                )
                rng = np.random.default_rng(self.random_state)
                indices = rng.choice(X.shape[0], n_optuna, replace=False)
                self.best_params = self._run_optuna_study(X[indices], y_coords[indices])
                del indices
            else:
                self.best_params = self._run_optuna_study(X, y_coords)
        else:
            logger.info(
                "Skipping hyperparameter optimisation. Using default parameters."
            )

        # PHASE 1: CROSS-VALIDATION
        if self.evaluate and self.cv_folds > 0:
            logger.info(f"Phase 1 — {self.cv_folds}-fold cross-validation")
            self._run_cross_validation(X, y_coords)

        # PHASE 2: PRODUCTION MODEL
        logger.info(
            f"Phase 2 — production model on 100% of the data ({X.shape[0]:,} molecules)"
        )
        base_xgb = XGBRegressor(**self.best_params)
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X, y_coords)
        logger.success("Final production model trained successfully.")

        del X, y_coords
        gc.collect()

        self.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_optuna_study(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna hyperparameter search and return the best params dict."""
        n_trials = OPTUNA_TRIALS
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            """Mean RMSE over an inner k-fold split for one hyperparameter draw."""
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.15, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            cv = KFold(
                n_splits=OPTUNA_INNER_FOLDS,
                shuffle=True,
                random_state=self.random_state,
            )
            rmses = []
            for fold_idx, (t_idx, v_idx) in enumerate(cv.split(X)):
                model = MultiOutputRegressor(XGBRegressor(**param))
                model.fit(X[t_idx], y[t_idx])
                preds = model.predict(X[v_idx])
                rmse = np.sqrt(mean_squared_error(y[v_idx], preds))
                rmses.append(rmse)
                del model, preds
                trial.report(rmse, fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return np.mean(rmses)

        # The sampler must be seeded explicitly: an unseeded TPESampler makes the
        # selected hyperparameters — and therefore every reported metric — differ
        # from run to run.
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        progress = LiveProgressBar(f"{self.label} Optuna trials", total=n_trials)
        with progress.live() as bar:

            def _callback(study, trial):
                bar.advance()
                bar.set_note(f"best RMSE {study.best_value:.4f}")

            study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[_callback],
                show_progress_bar=False,
            )

        logger.success(f"Optimisation complete — best RMSE: {study.best_value:.4f}")

        final_params = {
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
            "random_state": self.random_state,
        }
        final_params.update(study.best_params)
        logger.info(
            "Best hyperparameters: "
            + ", ".join(f"{k}={v}" for k, v in study.best_params.items())
        )
        return final_params

    def _run_cross_validation(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """K-Fold CV loop: computes metrics, saves CSVs, and delegates plotting."""
        val_dir = os.path.join(
            self.dir_path, self.surrogate_name, "validation_artifacts"
        )
        os.makedirs(val_dir, exist_ok=True)
        logger.info(f"Saving validation artifacts to: {val_dir}")

        # Shade the full reference set once and reuse it as the backdrop for every
        # fold figure. Scattering all N points per fold with matplotlib was the
        # slowest part of reporting on a large library.
        background = report_plots.landscape_image(self.dir_path, self.projection_name)

        r2_list, rmse_list, mae_list, euc_list = [], [], [], []
        kfold = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        progress = LiveProgressBar(
            f"{self.label} cross-validation folds", total=self.cv_folds
        )
        with progress.live() as bar:
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
                fold_num = fold_idx + 1
                bar.set_note(f"fold {fold_num}/{self.cv_folds}")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y_true[train_idx], y_true[test_idx]

                model = MultiOutputRegressor(XGBRegressor(**self.best_params))
                model.fit(X_train, y_train_fold)
                y_pred_fold = model.predict(X_test)

                # Metrics
                r_squared = r2_score(y_test_fold, y_pred_fold)
                r2_list.append(r_squared)
                rmse_list.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
                mae_list.append(mean_absolute_error(y_test_fold, y_pred_fold))
                dists = np.sqrt(np.sum((y_test_fold - y_pred_fold) ** 2, axis=1))
                euc_mean = np.mean(dists)
                euc_list.append(euc_mean)

                logger.debug(
                    f"Fold {fold_num} — R²: {r_squared:.4f} | "
                    f"RMSE: {rmse_list[-1]:.4f} | Euclidean: {euc_mean:.4f}"
                )

                # Persist predictions
                df_fold = pd.DataFrame(
                    {
                        "true_x": y_test_fold[:, 0],
                        "true_y": y_test_fold[:, 1],
                        "pred_x": y_pred_fold[:, 0],
                        "pred_y": y_pred_fold[:, 1],
                        "euclidean_error": dists,
                    }
                )
                df_fold.to_csv(
                    os.path.join(val_dir, f"fold_{fold_num}_predictions.csv"),
                    index=False,
                )

                # Figures go straight into the report directory as PNG + PDF.
                report_plots.FoldComparisonPlot(
                    self.projection_name,
                    fold_num,
                    y_test_fold,
                    y_pred_fold,
                    (r_squared, euc_mean),
                    self.dir_path,
                    background_image=background,
                ).save()
                report_plots.FoldDistributionsPlot(
                    self.projection_name,
                    fold_num,
                    y_test_fold,
                    y_pred_fold,
                    self.dir_path,
                ).save()
                report_plots.FoldZonesPlot(
                    self.projection_name,
                    fold_num,
                    y_test_fold,
                    y_pred_fold,
                    self.dir_path,
                    random_state=self.random_state,
                ).save()

                # Free all fold-local allocations before next iteration
                del (
                    model,
                    X_train,
                    X_test,
                    y_train_fold,
                    y_test_fold,
                    y_pred_fold,
                    dists,
                    df_fold,
                )
                gc.collect()

                bar.advance()

        self.metrics = {
            "eval_type": "Cross-Validation",
            "cv_folds": self.cv_folds,
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
            "r2_per_fold": [float(v) for v in r2_list],
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
            "euclidean_mean": float(np.mean(euc_list)),
            "euclidean_std": float(np.std(euc_list)),
            "euclidean_per_fold": [float(v) for v in euc_list],
        }

        log_cv_results(self.metrics)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save the surrogate regressor, scaler, and validation results."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)
        logger.info(f"Saving surrogate artifacts to: {proj_path}")

        joblib.dump(self.model, os.path.join(proj_path, "xgb_model.joblib"))
        logger.debug("Saved: xgb_model.joblib")

        joblib.dump(self.axis_scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        logger.debug("Saved: axis_scaler.pkl")

        if self.metrics is not None:
            joblib.dump(self.metrics, os.path.join(proj_path, "metrics.joblib"))
            logger.debug("Saved: metrics.joblib (summary statistics)")
            # Also as JSON: the report reads it without needing joblib, and it is
            # the only human-readable record of the validation numbers.
            with open(os.path.join(proj_path, "metrics.json"), "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.debug("Saved: metrics.json")

        logger.success("All surrogate components saved successfully.")

    def load(self):
        """Load the surrogate components from disk."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        logger.info(f"Loading surrogate from: {proj_path}")

        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        self.axis_scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))

        metrics_path = os.path.join(proj_path, "metrics.joblib")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
            if self.metrics:
                logger.info(
                    f"CV results — R²: {self.metrics['r2_mean']:.4f} ± {self.metrics['r2_std']:.4f}"
                )

        logger.success("Surrogate loaded successfully.")
        return self
