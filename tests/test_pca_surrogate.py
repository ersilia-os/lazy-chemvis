import os
import subprocess
import sys

import numpy as np
import pytest

from lazychemvis.artifacts.pca import PCAArtifact
from lazychemvis.featurizers.rdkit_descriptor import RDKitDescriptor
from lazychemvis.projectors.pca import PCAProjector
from lazychemvis.surrogates.pca import SURROGATE_FILE, PCASurrogate


@pytest.fixture
def fitted_pca(tmp_path, library_a):
    """A reference directory with descriptors, PCA projector and PCA surrogate."""
    featurizer = RDKitDescriptor(dir_path=str(tmp_path))
    featurizer.fit(library_a, reuse=False)
    featurizer.save()
    projector = PCAProjector(dir_path=str(tmp_path))
    projector.fit()
    projector.save()
    surrogate = PCASurrogate(dir_path=str(tmp_path))
    surrogate.fit()
    surrogate.save()
    return tmp_path, projector.X


def test_surrogate_reproduces_reference_coordinates(fitted_pca, library_a):
    path, reference = fitted_pca
    coords = PCAArtifact(str(path)).transform(library_a)
    np.testing.assert_allclose(coords, reference, atol=1e-10)


def test_falls_back_to_sklearn_pca_without_npz(fitted_pca, library_a):
    path, reference = fitted_pca
    os.remove(path / "pca" / SURROGATE_FILE)
    coords = PCAArtifact(str(path)).transform(library_a)
    np.testing.assert_allclose(coords, reference, atol=1e-10)


def test_missing_pca_raises(fitted_pca):
    path, _ = fitted_pca
    os.remove(path / "pca" / SURROGATE_FILE)
    os.remove(path / "pca" / "orig.pkl")
    with pytest.raises(FileNotFoundError, match="refit"):
        PCAArtifact(str(path))


def test_transform_does_not_import_torch():
    # An import hook that refuses torch. (Setting sys.modules['torch'] = None
    # instead would break scipy, which probes sys.modules for torch arrays.)
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'torch':\n"
        "            raise ImportError('torch is blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import lazychemvis.transform, lazychemvis.artifacts.pca\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
