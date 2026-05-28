import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — prevents Tk thread conflicts
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from rich.table import Table
from rich.panel import Panel
from rich import box

from ..helpers.logger import get_logger, console

logger = get_logger(__name__)


class ValidationPlots:
    """
    Handles all plotting and visual reporting for surrogate model validation.
    Decoupled from model training logic to keep concerns separate.
    """

    @staticmethod
    def plot_fold_comparison(
        y_all_true: np.ndarray,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        fold_num: int,
        save_dir: str,
        metrics: tuple
    ) -> None:
        """
        Generate a side-by-side scatter plot comparing ground truth vs surrogate
        predictions for a single CV fold.

        Parameters
        ----------
        y_all_true : np.ndarray, shape (N, 2)
            Full dataset coordinates, used as a gray background reference.
        y_test_true : np.ndarray, shape (n, 2)
            Ground truth UMAP coordinates for the held-out fold.
        y_test_pred : np.ndarray, shape (n, 2)
            Surrogate-predicted UMAP coordinates for the held-out fold.
        fold_num : int
            Fold index (1-based) used for labelling and file naming.
        save_dir : str
            Directory where the plot PNG will be saved.
        metrics : tuple of (float, float)
            (r2_score, mean_euclidean_distance) for this fold.
        """
        r2_val, euc_val = metrics

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        for ax in [ax1, ax2]:
            ax.scatter(
                y_all_true[:, 0], y_all_true[:, 1],
                c='lightgray', s=1, alpha=0.2, label='Background (All Data)'
            )
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.4)

        # Left: Ground Truth
        ax1.scatter(
            y_test_true[:, 0], y_test_true[:, 1],
            c='blue', s=10, alpha=0.6, label=f'Fold {fold_num} (True)'
        )
        ax1.set_title(f"Fold {fold_num}: Ground Truth", fontsize=14)
        ax1.legend(loc='upper right')

        # Right: Surrogate Prediction
        ax2.scatter(
            y_test_pred[:, 0], y_test_pred[:, 1],
            c='red', s=10, alpha=0.6, label=f'Fold {fold_num} (Predicted)'
        )
        ax2.set_title(f"Fold {fold_num}: Surrogate Prediction", fontsize=14)
        ax2.legend(loc='upper right')

        fig.suptitle(
            f"Cross-Validation Fold {fold_num} Analysis\n"
            f"R²: {r2_val:.3f} | Mean Euclidean Error: {euc_val:.3f}",
            fontsize=16
        )

        save_path = os.path.join(save_dir, f"fold_{fold_num}_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_distributions(
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        fold_num: int,
        save_dir: str
    ) -> None:
        """
        Generate True vs Predicted parity plots (hexbin density) and distribution
        histograms for both X and Y UMAP dimensions for a single CV fold.

        Parameters
        ----------
        y_test_true : np.ndarray, shape (n, 2)
            Ground truth UMAP coordinates for the held-out fold.
        y_test_pred : np.ndarray, shape (n, 2)
            Surrogate-predicted UMAP coordinates for the held-out fold.
        fold_num : int
            Fold index (1-based) used for labelling and file naming.
        save_dir : str
            Directory where the plot PNG will be saved.
        """
        corr_x, _ = pearsonr(y_test_true[:, 0], y_test_pred[:, 0])
        corr_y, _ = pearsonr(y_test_true[:, 1], y_test_pred[:, 1])

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        all_vals = np.concatenate([y_test_true, y_test_pred])
        min_val, max_val = all_vals.min(), all_vals.max()

        # Top-Left: X Coordinate parity plot
        hb_x = axs[0, 0].hexbin(y_test_true[:, 0], y_test_pred[:, 0], gridsize=50, cmap='Blues', mincnt=1)
        axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        axs[0, 0].set_title(f"X-Coordinate Correlation\nPearson R = {corr_x:.3f}", fontsize=14)
        axs[0, 0].set_xlabel("True X", fontsize=12)
        axs[0, 0].set_ylabel("Predicted X", fontsize=12)
        axs[0, 0].set_xlim(min_val, max_val)
        axs[0, 0].set_ylim(min_val, max_val)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        fig.colorbar(hb_x, ax=axs[0, 0], label='Density')

        # Top-Right: Y Coordinate parity plot
        hb_y = axs[0, 1].hexbin(y_test_true[:, 1], y_test_pred[:, 1], gridsize=50, cmap='Greens', mincnt=1)
        axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        axs[0, 1].set_title(f"Y-Coordinate Correlation\nPearson R = {corr_y:.3f}", fontsize=14)
        axs[0, 1].set_xlabel("True Y", fontsize=12)
        axs[0, 1].set_ylabel("Predicted Y", fontsize=12)
        axs[0, 1].set_xlim(min_val, max_val)
        axs[0, 1].set_ylim(min_val, max_val)
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()
        fig.colorbar(hb_y, ax=axs[0, 1], label='Density')

        # Bottom-Left: X distribution histogram
        axs[1, 0].hist(y_test_true[:, 0], bins=50, alpha=0.5, color='blue', label='True X Distribution', density=True)
        axs[1, 0].hist(y_test_pred[:, 0], bins=50, alpha=0.5, color='orange', label='Predicted X Distribution', density=True)
        axs[1, 0].set_title("X-Coordinate Distribution Spread", fontsize=14)
        axs[1, 0].set_xlabel("X Value")
        axs[1, 0].set_ylabel("Density")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Bottom-Right: Y distribution histogram
        axs[1, 1].hist(y_test_true[:, 1], bins=50, alpha=0.5, color='green', label='True Y Distribution', density=True)
        axs[1, 1].hist(y_test_pred[:, 1], bins=50, alpha=0.5, color='orange', label='Predicted Y Distribution', density=True)
        axs[1, 1].set_title("Y-Coordinate Distribution Spread", fontsize=14)
        axs[1, 1].set_xlabel("Y Value")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"Fold {fold_num}: Regression Fidelity Analysis", fontsize=18, y=0.98)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"fold_{fold_num}_distributions.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_kmeans_zones(
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        fold_num: int,
        save_dir: str,
        n_zones: int = 10
    ) -> None:
        """
        Cluster the ground truth into n_zones K-Means zones, colour-code each
        molecule by its zone, and check whether those zones stay coherent in
        the surrogate predictions.

        Parameters
        ----------
        y_test_true : np.ndarray, shape (n, 2)
            Ground truth UMAP coordinates for the held-out fold.
        y_test_pred : np.ndarray, shape (n, 2)
            Surrogate-predicted UMAP coordinates for the held-out fold.
        fold_num : int
            Fold index (1-based) used for labelling and file naming.
        save_dir : str
            Directory where the plot PNG will be saved.
        n_zones : int, default=10
            Number of K-Means clusters. Values ≤10 use the 'tab10' palette
            (maximally distinct colours); values up to 20 fall back to 'tab20'.
        """
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init='auto')
        zone_labels = kmeans.fit_predict(y_test_true)
        centers = kmeans.cluster_centers_

        cmap = 'tab10' if n_zones <= 10 else 'tab20'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        # Left: Ground Truth zones (clean clusters by definition)
        ax1.scatter(y_test_true[:, 0], y_test_true[:, 1], c=zone_labels, cmap=cmap, s=2, alpha=0.5, marker='.')
        ax1.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, linewidth=2, zorder=3)
        ax1.set_title(f"Fold {fold_num}: Ground Truth Zones\n(Separated by K-Means)", fontsize=14)

        # Right: Predictions coloured by the same zone labels — do they mix?
        ax2.scatter(y_test_pred[:, 0], y_test_pred[:, 1], c=zone_labels, cmap=cmap, s=2, alpha=0.5, marker='.')
        ax2.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, linewidth=2, zorder=3, alpha=0.3)
        ax2.set_title(f"Fold {fold_num}: Surrogate Prediction\n", fontsize=14)

        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"fold_{fold_num}_kmeans_{n_zones}_zones.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def print_cv_results(metrics: dict) -> None:
        """
        Print a formatted cross-validation summary using Rich tables and loguru.

        Parameters
        ----------
        metrics : dict
            Dictionary produced by the CV loop, expected keys:
            cv_folds, r2_mean, r2_std, r2_per_fold,
            rmse_mean, rmse_std, mae_mean, mae_std,
            euclidean_mean, euclidean_std, euclidean_per_fold.
        """
        m = metrics

        console.print(Panel.fit(
            f"Cross-Validation Results — {m['cv_folds']} folds\n"
            "[dim]Metrics computed on held-out test folds (honest estimates)[/dim]",
            style="bold blue"
        ))

        # --- Summary statistics table ---
        summary = Table(title="Summary Statistics (Mean ± Std)", box=box.SIMPLE_HEAVY, style="blue")
        summary.add_column("Metric", style="bold")
        summary.add_column("Mean", justify="right")
        summary.add_column("Std", justify="right")
        summary.add_row("R² Score",           f"{m['r2_mean']:.4f}",        f"± {m['r2_std']:.4f}")
        summary.add_row("RMSE",               f"{m['rmse_mean']:.4f}",      f"± {m['rmse_std']:.4f}")
        summary.add_row("MAE",                f"{m['mae_mean']:.4f}",       f"± {m['mae_std']:.4f}")
        summary.add_row("Euclidean Distance", f"{m['euclidean_mean']:.4f}", f"± {m['euclidean_std']:.4f}")
        console.print(summary)

        # --- Per-fold breakdown table ---
        fold_table = Table(title="Per-Fold Breakdown", box=box.SIMPLE_HEAVY, style="cyan")
        fold_table.add_column("Fold", justify="center", style="bold")
        fold_table.add_column("R²", justify="right")
        fold_table.add_column("Euclidean Distance", justify="right")
        for i in range(m["cv_folds"]):
            fold_table.add_row(
                str(i + 1),
                f"{m['r2_per_fold'][i]:.4f}",
                f"{m['euclidean_per_fold'][i]:.4f}",
            )
        console.print(fold_table)

        # --- Quality assessment ---
        r2_lower = m["r2_mean"] - m["r2_std"]
        r2_upper = m["r2_mean"] + m["r2_std"]

        if m["r2_mean"] > 0.95:
            quality, quality_style = "EXCELLENT ✓✓✓", "bold green"
        elif m["r2_mean"] > 0.90:
            quality, quality_style = "VERY GOOD ✓✓", "green"
        elif m["r2_mean"] > 0.85:
            quality, quality_style = "GOOD ✓", "yellow"
        elif m["r2_mean"] > 0.75:
            quality, quality_style = "ACCEPTABLE", "yellow"
        else:
            quality, quality_style = "NEEDS IMPROVEMENT", "bold red"

        if m["euclidean_mean"] < 0.05:
            acc_label, acc_style = "EXCELLENT", "bold green"
        elif m["euclidean_mean"] < 0.10:
            acc_label, acc_style = "VERY GOOD", "green"
        elif m["euclidean_mean"] < 0.15:
            acc_label, acc_style = "GOOD", "yellow"
        else:
            acc_label, acc_style = "MODERATE", "bold yellow"

        qa_table = Table(title="Quality Assessment", box=box.SIMPLE_HEAVY, style="magenta")
        qa_table.add_column("Attribute", style="bold")
        qa_table.add_column("Value")
        qa_table.add_row("Overall Quality",      f"[{quality_style}]{quality}[/{quality_style}]")
        qa_table.add_row("R² (mean ± std)",      f"{m['r2_mean']:.4f} ± {m['r2_std']:.4f}")
        qa_table.add_row("95% Confidence Interval", f"[{r2_lower:.4f}, {r2_upper:.4f}]")
        qa_table.add_row("Mean Euclidean Error", f"{m['euclidean_mean']:.4f} ± {m['euclidean_std']:.4f} units")
        qa_table.add_row("Prediction Accuracy",  f"[{acc_style}]{acc_label}[/{acc_style}]")
        console.print(qa_table)

        # --- Warnings via loguru ---
        if m["r2_std"] > 0.05:
            logger.warning(
                f"High variance across folds (std={m['r2_std']:.4f}). "
                "This may indicate poorly shuffled data, distinct subpopulations, "
                "or high model sensitivity to data distribution."
            )
        else:
            logger.success("Cross-validation complete — model shows stable performance across folds.")