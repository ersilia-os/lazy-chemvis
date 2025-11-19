import os
import pandas as pd

from .helpers import load_lib_input
from .artifacts import PCAArtifact


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

    def run(self):
        smiles_list = load_lib_input(self.lib_input)
        df = self._pca_step(smiles_list)
        df.to_csv(self.output_path, index=False)


if __name__ == "__main__":
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
