import os
import gc
import joblib

import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from ..featurizers.ecfp import ECFPFeaturizer
from ..projectors.tsne_projector import TSNEProjector
from ..plots.validation_plots import ValidationPlots
from ..helpers.logger import get_logger, console

logger = get_logger(__name__)


class TSNESurrogate(object):
    """
    Shortcut Surrogate for t-SNE:
    Maps local ECFP bits directly to CheMeleon-t-SNE coordinates.
    Bypasses the need for the CheMeleon API during inference.

    Scientific best practices:
    1. Validation is performed BEFORE training the final model.
    2. Cross-validation reports mean ± std across folds.
    3. Final production model is trained on 100% of data.
    """

    def __init__(
        self,
        dir_path: str,
        evaluate: bool = True,
        cv_folds: int = 5,
        optimize: bool = True,
        random_state: int = 42,
    ):
        """
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
        self.surrogate_name = "tsne_surrogate"
        self.dir_path = os.path.abspath(dir_path)
        self.evaluate = evaluate
        self.cv_folds = cv_folds
        self.cv_results = None
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
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train the surrogate XGBoost following the flow:
        Optimisation (Optuna) → Validation (CV) → Production (100% data).
        """
        console.print(Panel.fit("🚀  t-SNE Surrogate — Training Pipeline", style="bold cyan"))

        # 1. DATA LOADING
        logger.info("Loading ECFP features and t-SNE projector...")
        ecfp_feat = ECFPFeaturizer.load(dir_path=self.dir_path)
        X = ecfp_feat.X

        projector = TSNEProjector.load(dir_path=self.dir_path)
        y_coords = projector.X
        self.axis_scaler = projector.scaler

        if X.shape[0] != y_coords.shape[0]:
            logger.error(f"Dimension mismatch: X={X.shape[0]}, Y={y_coords.shape[0]}")
            raise ValueError(f"Dimension mismatch: X={X.shape[0]}, Y={y_coords.shape[0]}")

        # Release featurizer and projector wrappers — arrays live on via X and y_coords
        del ecfp_feat, projector
        gc.collect()
        logger.debug("Memory freed: featurizer and projector wrappers released.")

        # PHASE 0: HYPERPARAMETER OPTIMISATION
        if self.optimize:
            console.print(Panel.fit("Phase 0 — Hyperparameter Optimisation (Optuna)", style="bold yellow"))
            if X.shape[0] > 100_000:
                n_optuna = min(400_000, X.shape[0])
                logger.info(f"Large dataset detected — subsampling {n_optuna:,} molecules for Optuna.")
                rng = np.random.default_rng(self.random_state)
                indices = rng.choice(X.shape[0], n_optuna, replace=False)
                self.best_params = self._run_optuna_study(X[indices], y_coords[indices])
                del indices
            else:
                self.best_params = self._run_optuna_study(X, y_coords)
        else:
            logger.info("Skipping hyperparameter optimisation. Using default parameters.")

        # PHASE 1: CROSS-VALIDATION
        if self.evaluate and self.cv_folds > 0:
            console.print(Panel.fit(f"Phase 1 — Validation ({self.cv_folds}-Fold Cross-Validation)", style="bold blue"))
            logger.info("Evaluating optimised configuration on held-out data...")
            self._run_cross_validation(X, y_coords)

        # PHASE 2: PRODUCTION MODEL
        console.print(Panel.fit("Phase 2 — Training Final Production Model", style="bold green"))
        logger.info(f"Training on 100% of the data ({X.shape[0]:,} molecules)...")

        base_xgb = XGBRegressor(**self.best_params)
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X, y_coords)
        logger.success("Final production model trained successfully.")

        del X, y_coords
        gc.collect()
        logger.debug("Memory freed: training arrays released after production fit.")

        self.save()
        console.print(Panel.fit("✅  Pipeline completed successfully", style="bold green"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_optuna_study(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna hyperparameter search and return the best params dict."""
        n_trials = 30
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[bold yellow]{task.completed}/{task.total} trials"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Optimising hyperparameters...", total=n_trials)

            def _callback(study, trial):
                best = study.best_value
                progress.update(
                    task,
                    advance=1,
                    description=f"[yellow]Trial {trial.number + 1}/{n_trials} — best RMSE: {best:.4f}",
                )

            study.optimize(objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)

        logger.success(f"Optimisation complete — best RMSE: {study.best_value:.4f}")

        final_params = {
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
            "random_state": self.random_state,
        }
        final_params.update(study.best_params)

        table = Table(title="Best Hyperparameters", box=box.SIMPLE_HEAVY, style="cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        for k, v in study.best_params.items():
            table.add_row(k, str(v))
        console.print(table)

        return final_params

    def _run_cross_validation(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """K-Fold CV loop: computes metrics, saves CSVs, and delegates plotting."""
        val_dir = os.path.join(self.dir_path, self.surrogate_name, "validation_artifacts")
        os.makedirs(val_dir, exist_ok=True)
        logger.info(f"Saving validation artifacts to: {val_dir}")

        r2_list, rmse_list, mae_list, euc_list = [], [], [], []
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[bold cyan]{task.completed}/{task.total} folds"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Cross-validation...", total=self.cv_folds)

            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
                fold_num = fold_idx + 1
                progress.update(task, description=f"[cyan]Fold {fold_num}/{self.cv_folds}")

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
                df_fold = pd.DataFrame({
                    "true_x": y_test_fold[:, 0],
                    "true_y": y_test_fold[:, 1],
                    "pred_x": y_pred_fold[:, 0],
                    "pred_y": y_pred_fold[:, 1],
                    "euclidean_error": dists,
                })
                df_fold.to_csv(
                    os.path.join(val_dir, f"fold_{fold_num}_predictions.csv"), index=False
                )

                # Plotting
                ValidationPlots.plot_fold_comparison(
                    y_all_true=y_true,
                    y_test_true=y_test_fold,
                    y_test_pred=y_pred_fold,
                    fold_num=fold_num,
                    save_dir=val_dir,
                    metrics=(r_squared, euc_mean),
                )
                ValidationPlots.plot_distributions(
                    y_test_true=y_test_fold,
                    y_test_pred=y_pred_fold,
                    fold_num=fold_num,
                    save_dir=val_dir,
                )
                ValidationPlots.plot_kmeans_zones(
                    y_test_true=y_test_fold,
                    y_test_pred=y_pred_fold,
                    fold_num=fold_num,
                    save_dir=val_dir,
                )

                # Free all fold-local allocations before next iteration
                del model, X_train, X_test, y_train_fold, y_test_fold, y_pred_fold, dists, df_fold
                gc.collect()

                progress.advance(task)

        self.metrics = {
            "eval_type": "Cross-Validation",
            "cv_folds": self.cv_folds,
            "r2_mean": np.mean(r2_list),
            "r2_std": np.std(r2_list),
            "r2_per_fold": r2_list,
            "rmse_mean": np.mean(rmse_list),
            "rmse_std": np.std(rmse_list),
            "mae_mean": np.mean(mae_list),
            "mae_std": np.std(mae_list),
            "euclidean_mean": np.mean(euc_list),
            "euclidean_std": np.std(euc_list),
            "euclidean_per_fold": euc_list,
        }

        ValidationPlots.print_cv_results(self.metrics)

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

        if self.cv_results is not None:
            joblib.dump(self.cv_results, os.path.join(proj_path, "cv_results.pkl"))
            logger.debug("Saved: cv_results.pkl (raw fold scores)")

        if self.metrics is not None:
            joblib.dump(self.metrics, os.path.join(proj_path, "metrics.joblib"))
            logger.debug("Saved: metrics.joblib (summary statistics)")

        logger.success("All surrogate components saved successfully.")

    def load(self):
        """Load the surrogate components from disk."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        logger.info(f"Loading surrogate from: {proj_path}")

        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        self.axis_scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))

        cv_path = os.path.join(proj_path, "cv_results.pkl")
        if os.path.exists(cv_path):
            self.cv_results = joblib.load(cv_path)
            logger.debug("Loaded: cv_results.pkl")

        metrics_path = os.path.join(proj_path, "metrics.joblib")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
            if self.metrics:
                logger.info(
                    f"CV results — R²: {self.metrics['r2_mean']:.4f} ± {self.metrics['r2_std']:.4f}"
                )

        logger.success("Surrogate loaded successfully.")
        return self
