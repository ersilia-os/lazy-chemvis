from .helpers.libraries import load_lib_input

from .featurizers.rdkit_descriptor import RDKitDescriptor
from .projectors.pca import PCAProjector
from .surrogates.pca import PCASurrogate
from .plots.scatter import ScatterPlot


class Pipeline(object):
    def __init__(self, lib_input: str, dir_path: str):
        self.lib_input = lib_input
        self.dir_path = dir_path

    def _pca_step(self, smiles_list):
        featurizer = RDKitDescriptor(dir_path=self.dir_path)
        featurizer.fit(smiles_list)
        featurizer.save()
        pca_proj = PCAProjector(dir_path=self.dir_path)
        pca_proj.fit()
        pca_proj.save()
        pca_surrogate = PCASurrogate(dir_path=self.dir_path)
        pca_surrogate.fit()
        pca_surrogate.save()
        scatter = ScatterPlot(projection_name="pca", dir_path=self.dir_path)
        scatter.plot_reference()

    def run(self):
        smiles_list = load_lib_input(self.lib_input)
        self._pca_step(smiles_list)


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
        help="Directory to save trained featurizers and projectors",
    )
    args = parser.parse_args()
    pipe = Pipeline(args.lib_input, args.dir_path)
    pipe.run()


if __name__ == "__main__":
    main()
