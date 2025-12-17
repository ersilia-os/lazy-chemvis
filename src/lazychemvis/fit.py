"""
End-to-end PCA pipeline.

This script loads a SMILES library, computes RDKit descriptors, fits a 2D PCA
projection, creates a PyTorch surrogate model for the PCA transform, and
generates a reference scatter plot. All trained components and outputs are
stored in the specified directory.
"""

from .helpers.libraries import load_lib_input
from .featurizers.ecfp import ECFPFeaturizer
from .featurizers.rdkit_descriptor import RDKitDescriptor

from .projectors.tmap_projector import TMAPProjector
from .projectors.pca import PCAProjector

from .surrogates.pca import PCASurrogate
from .plots.scatter import ScatterPlot


class Pipeline(object):
    """
    Complete processing pipeline for computing a PCA chemical space.

    This pipeline:
      1. Loads input SMILES.
      2. Fits RDKitDescriptor on the molecules and saves it.
      3. Fits a PCA model on the descriptor matrix and saves it.
      4. Converts the PCA model into a PyTorch surrogate and saves it.
      5. Produces a 2D scatter plot of the reference projection.

    The results stored in `dir_path` can later be used to project new molecules
    into the same PCA space via the PCAArtifact class.
    """

    def __init__(self, lib_input: str, dir_path: str, tmap_env:str):
        """
        Initialize the pipeline.

        Parameters
        ----------
        lib_input : str
            Path to a SMILES file or name of a built-in dataset.
        dir_path : str
            Directory in which all trained models and outputs will be saved.
        """
        self.lib_input = lib_input
        self.dir_path = dir_path
        self.tmap_env=tmap_env

    def _pca_step(self, smiles_list):
        """
        Execute the descriptor → PCA → surrogate → plot sequence.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings representing the reference chemical space.

        Steps
        -----
        1. Fit RDKit descriptor featurizer.
        2. Fit PCA on the descriptor matrix.
        3. Create and save a fixed PyTorch PCA surrogate.
        4. Generate and save a scatter plot of the PCA projection.
        """
        # 1. Fit and save descriptor featurizer
        featurizer = RDKitDescriptor(dir_path=self.dir_path)
        featurizer.fit(smiles_list)
        featurizer.save()

        # 2. Fit and save PCA projection
        pca_proj = PCAProjector(dir_path=self.dir_path)
        pca_proj.fit()
        pca_proj.save()

        # 3. Fit and save PCA surrogate model
        pca_surrogate = PCASurrogate(dir_path=self.dir_path)
        pca_surrogate.fit()
        pca_surrogate.save()

        # 4. Plot reference chemical space
        scatter = ScatterPlot(projection_name="pca", dir_path=self.dir_path)
        scatter.plot_reference()

    def _tmap_step(self, smiles_list):
        """
        Execute the ECFP → TMAP → plot sequence.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings representing the reference chemical space.

        Steps
        -----
        1. Fit ECFP fingerprint featurizer.
        2. Fit TMAP layout on the fingerprint matrix.
        3. Save all artifacts (LSH forest, scaler, coordinates).
        4. Generate and save a scatter plot of the TMAP projection.
        """
        # 1. Fit and save ECFP featurizer
        featurizer = ECFPFeaturizer(dir_path=self.dir_path)
        featurizer.fit(smiles_list)
        featurizer.save()

        # 2. Fit and save TMAP projection
        tmap_proj = TMAPProjector(dir_path=self.dir_path)
        tmap_proj.fit(self.tmap_env)

        # 3. Plot reference chemical space
        scatter = ScatterPlot(projection_name="tmap", dir_path=self.dir_path)
        scatter.plot_reference()

    def run(self):
        """
        Run the full PCA pipeline.

        This loads the SMILES library using load_lib_input() and executes the
        PCA step on the resulting molecules.

        Returns
        -------
        None
        """
        smiles_list = load_lib_input(self.lib_input)
        self._pca_step(smiles_list)
        self._tmap_step(smiles_list)


def main():
    """
    Command-line entry point for running the pipeline.

    Expected command:
        python fit.py --lib_input <file_or_dataset> --dir_path <output_dir>
    """
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
        help="Directory to save trained featurizers and projectors",
    )

    parser.add_argument("--tmap_env",
        type=str,
        default="tmap-env",
        help="Path to the TMAP conda environment",
    )
    args = parser.parse_args()

    pipe = Pipeline(args.lib_input, args.dir_path, args.tmap_env)
    pipe.run()


if __name__ == "__main__":
    main()
