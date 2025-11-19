import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "test/pca_projector/pca_reduced.npy"
data = np.load(file_name)

print("PCA reduced data shape:", data.shape)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.7)

plt.savefig("pca_scatter_plot.png", dpi=300)

df = pd.read_csv("test.csv")
print(df.head())
plt.figure(figsize=(8, 6))
plt.scatter(df["pca_x"], df["pca_y"], alpha=0.7)

plt.savefig("pca_scatter_plot_from_csv.png", dpi=300)