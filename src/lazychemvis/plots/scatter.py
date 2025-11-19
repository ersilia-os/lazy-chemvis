import os
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf


import matplotlib.pyplot as plt


class ScatterPlot(object):
    def __init__(self, projection_name, dir_path: str):
        self.plot_name = "scatter"
        self.dir_path = os.path.abspath(dir_path)
        self.projection_name = projection_name
        self.x_min = -1.0
        self.x_max = 1.0
        self.y_min = -1.0
        self.y_max = 1.0

    def plot_reference(self):
        data_path = os.path.join(self.dir_path, self.projection_name, "reduced.npy")
        X = np.load(data_path).astype(np.float32)
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        canvas = ds.Canvas(
            plot_width=1000,
            plot_height=1000,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )
        agg = canvas.points(df, "x", "y")
        img = tf.shade(agg, how="eq_hist")
        img = tf.set_background(img, "white")

        out_path = os.path.join(
            self.dir_path, self.projection_name, f"{self.plot_name}_shaded.png"
        )
        img.to_pil().save(out_path)

        plt.figure(figsize=(10, 10))
        plt.scatter(df["x"], df["y"], s=0.1, alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(
            os.path.join(
                self.dir_path, self.projection_name, f"{self.plot_name}_dots.png"
            ),
            dpi=300,
        )

    def plot_overlay(self):
        pass
