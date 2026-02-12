import os
import pandas as pd

from .helpers.libraries import load_lib_input
from .artifacts.pca import PCAArtifact
from .artifacts.tmap import TMAPArtifact
from .artifacts.tsne import TSNEArtifact
from .artifacts.umap import UMAPArtifact

from .plots.scatter import ScatterPlot


class Pipeline(object):
    def __init__(self, lib_input: str, dir_path: str, output_path: str):
        self.lib_input = lib_input
        self.dir_path = os.path.abspath(dir_path)
        self.output_path = output_path

    def _pca_step(self, smiles_list):
        pca_artifact = PCAArtifact(dir_name=self.dir_path)
        X_reduced = pca_artifact.transform(smiles_list)
        df = pd.DataFrame(X_reduced, columns=["pca_x", "pca_y"])
        return df
    
    # def _tmap_step(self, smiles_list):
    #     # This calls the Artifact we just wrote
    #     tmap_artifact = TMAPArtifact(dir_name=self.dir_path)
    #     X_reduced = tmap_artifact.transform(smiles_list)
    #     return pd.DataFrame(X_reduced, columns=["tmap_x", "tmap_y"])

    def run(self):
        smiles_list = load_lib_input(self.lib_input)

        # 1. PCA step
        df = self._pca_step(smiles_list)

        #plotting PCA
        scatter_plot = ScatterPlot(projection_name="pca", dir_path=self.dir_path, output_path=self.output_path)
        scatter_plot.plot_overlay(new_coords=df[["pca_x", "pca_y"]].to_numpy(), label="Input Molecules")

        # 2. TMAP step
        # df = self._tmap_step(smiles_list)

        #3. TSNE step
        # df = self._tsne_step(smiles_list)   
          
        #4. UMAP step
        # df = self._umap_step(smiles_list)
        output_df =os.path.join(self.output_path, "coordinates.csv")
    #    df.to_csv(output_df, index=False)


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
    args = parser.parse_args()
    pipe = Pipeline(args.lib_input, args.dir_path, args.output_path)
    pipe.run()


if __name__ == "__main__":
    main()
