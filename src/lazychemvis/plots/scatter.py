import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — prevents Tk thread conflicts
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt

from ..helpers.logger import get_logger

logger = get_logger(__name__)

class ScatterPlot(object):
    def __init__(self, projection_name: str, dir_path: str, output_path:str=''):
        self.projection_name = projection_name
        self.dir_path = os.path.abspath(dir_path)
        self.output_path = os.path.abspath(output_path)
        
        # Your specific aesthetic settings
        self.background_cmap = ["#e7e2e2", "#cac8c8", "#B1B0B0", "#989797"]
        self.overlay_color = "#1E13AC"
        
        # Use a consistent extent for the -1 to 1 artifact space 
        self.extent = [-1.05, 1.05, -1.05, 1.05]

    def _generate_background_image(self):
        """Internal helper to render the Datashader reference map."""
        data_path = os.path.join(self.dir_path, self.projection_name, "reduced.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Reference coordinates not found at {data_path}")
            
        X = np.load(data_path).astype(np.float32)
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})

        # Matches your 1000x1000 preference
        cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        agg = cvs.points(df, "x", "y")
        
        # Shade using your custom grey palette
        img = tf.shade(agg, cmap=self.background_cmap, how='eq_hist')
        img = tf.spread(img, px=1, shape='circle')
        img = tf.set_background(img, "white")
        
        return img.to_pil()

    def plot_reference(self, save_name: str = "reference_space.png"):
        """Plots only the underlying chemical landscape."""
        pil_img = self._generate_background_image()
        
        plt.figure(figsize=(12, 12), facecolor='white')
        ax = plt.gca()
        
        ax.imshow(pil_img, extent=self.extent, interpolation='lanczos')
        
        self._apply_styling(ax, f"Chemical Space: {self.projection_name.upper()}")

        out_path = os.path.join(self.dir_path, self.projection_name, save_name)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Reference plot saved: {out_path}")

    def plot_overlay(self, new_coords: np.ndarray, label: str = "New Compounds"):
        """
        Plots new molecules on top of the grey landscape using your coral palette.
        """
        pil_img = self._generate_background_image()
        
        plt.figure(figsize=(12, 12), facecolor='white')
        ax = plt.gca()
        
        # 1. Background
        ax.imshow(pil_img, extent=self.extent, interpolation='lanczos', alpha=1.0)
        
        # 2. Overlay
        ax.scatter(
            new_coords[:, 0], 
            new_coords[:, 1], 
            c=self.overlay_color, 
            s=10, 
            alpha=1.0, 
            edgecolors='white', 
            linewidths=0.3, 
            label=label,
            zorder=10
        )
        
        # 3. Styling & Legend
        self._apply_styling(ax, f"Overlay Analysis: {self.projection_name.upper()}")
        ax.legend(loc='upper right', frameon=True, fontsize=12, edgecolor='#dddddd')

        out_path = os.path.join(self.output_path, f"{self.projection_name}_plot.png")

        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Overlay plot saved: {out_path}")
        
    def _apply_styling(self, ax, title):
        """Standardizes the look of the plots."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20, loc='center')