import gc
import os
import pandas as pd

from .helpers.libraries import load_lib_input
from .artifacts.pca import PCAArtifact
from .artifacts.tmap import TMAPArtifact
from .artifacts.tsne import TSNEArtifact
from .artifacts.umap import UMAPArtifact
from .featurizers.ecfp import ECFPFeaturizer

from .plots.scatter import ScatterPlot


class Pipeline(object):
    def __init__(self, lib_input: str, dir_path: str, output_path: str, no_plots: bool = False):
        self.lib_input = lib_input
        self.dir_path = os.path.abspath(dir_path)
        self.output_path = output_path
        self.no_plots = no_plots

    def _pca_step(self, smiles_list):
        pca_artifact = PCAArtifact(dir_name=self.dir_path)
        X_reduced = pca_artifact.transform(smiles_list)
        del pca_artifact
        return pd.DataFrame(X_reduced, columns=["pca_x", "pca_y"])

    def _tmap_step(self, smiles_list):
        tmap_artifact = TMAPArtifact(dir_name=self.dir_path)
        X_reduced = tmap_artifact.transform(smiles_list)
        del tmap_artifact
        return pd.DataFrame(X_reduced, columns=["tmap_x", "tmap_y"])

    def _tsne_step(self, smiles_list, X_ecfp=None):
        tsne_artifact = TSNEArtifact(dir_name=self.dir_path)
        X_reduced = tsne_artifact.transform(smiles_list, X_ecfp=X_ecfp)
        del tsne_artifact
        return pd.DataFrame(X_reduced, columns=["tsne_x", "tsne_y"])

    def _umap_step(self, smiles_list, X_ecfp=None):
        umap_artifact = UMAPArtifact(dir_name=self.dir_path)
        X_reduced = umap_artifact.transform(smiles_list, X_ecfp=X_ecfp)
        del umap_artifact
        return pd.DataFrame(X_reduced, columns=["umap_x", "umap_y"])

    def run(self):
        smiles_list = load_lib_input(self.lib_input)

        # Initialize with SMILES column
        df_combined = pd.DataFrame({'smiles': smiles_list})

        # 1. PCA step
        df_pca = self._pca_step(smiles_list)
        df_combined = pd.concat([df_combined, df_pca], axis=1)
        if not self.no_plots:
            scatter_plot = ScatterPlot(projection_name="pca", dir_path=self.dir_path, output_path=self.output_path)
            scatter_plot.plot_overlay(new_coords=df_pca[["pca_x", "pca_y"]].to_numpy(), label="Input Molecules")
        del df_pca
        gc.collect()

        # 2. TMAP step
        df_tmap = self._tmap_step(smiles_list)
        df_combined = pd.concat([df_combined, df_tmap], axis=1)
        if not self.no_plots:
            scatter_plot = ScatterPlot(projection_name="tmap", dir_path=self.dir_path, output_path=self.output_path)
            scatter_plot.plot_overlay(new_coords=df_tmap[["tmap_x", "tmap_y"]].to_numpy(), label="Input Molecules")
        del df_tmap
        gc.collect()

        # 3. TSNE step — compute ECFP once and reuse for UMAP
        ecfp_featurizer = ECFPFeaturizer.load(dir_path=self.dir_path, load_X=False)
        X_ecfp = ecfp_featurizer.transform(smiles_list)
        del ecfp_featurizer

        df_tsne = self._tsne_step(smiles_list, X_ecfp=X_ecfp)
        df_combined = pd.concat([df_combined, df_tsne], axis=1)
        if not self.no_plots:
            scatter_plot = ScatterPlot(projection_name="tsne", dir_path=self.dir_path, output_path=self.output_path)
            scatter_plot.plot_overlay(new_coords=df_tsne[["tsne_x", "tsne_y"]].to_numpy(), label="Input Molecules")
        del df_tsne
        gc.collect()

        # 4. UMAP step — reuse X_ecfp computed above
        df_umap = self._umap_step(smiles_list, X_ecfp=X_ecfp)
        df_combined = pd.concat([df_combined, df_umap], axis=1)
        if not self.no_plots:
            scatter_plot = ScatterPlot(projection_name="umap", dir_path=self.dir_path, output_path=self.output_path)
            scatter_plot.plot_overlay(new_coords=df_umap[["umap_x", "umap_y"]].to_numpy(), label="Input Molecules")
        del df_umap, X_ecfp
        gc.collect()

        # Output - Save the combined dataframe
        output_df = os.path.join(self.output_path, "coordinates.csv")
        df_combined.to_csv(output_df, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib_input",
        type=str,
        help="Path to input library (SMILES format) or name of built-in dataset",
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        help="Directory where trained featurizers and projectors are saved",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the transformed PCA coordinates (CSV format)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip overlay plots and only output the CSV.",
    )
    args = parser.parse_args()
    pipe = Pipeline(args.lib_input, args.dir_path, args.output_path, args.no_plots)
    pipe.run()


if __name__ == "__main__":
    main()
