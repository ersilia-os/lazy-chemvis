"""
TMAP artifact.

TMAP has no parametric transform: its layout is a spanning tree over the
reference set, so there is no function to apply to an unseen molecule. New
molecules are placed by *coordinate inheritance* — each one takes the
coordinates of its most similar reference molecule, by Tanimoto similarity over
Morgan fingerprints, looked up in an FPSim2 database.

Note that this means a projected molecule always lands exactly on top of an
existing reference point, and two molecules sharing a nearest neighbour receive
identical coordinates. Unlike the PCA, t-SNE and UMAP panels, the TMAP panel
cannot place a molecule between known regions.
"""

import os
import numpy as np
from typing import List

from rdkit import Chem
from rdkit import RDLogger
from FPSim2 import FPSim2Engine

from ..helpers.logger import get_logger, console

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)

# Number of offending entries quoted in the warning message
_N_EXAMPLES = 5

# Kept at 0.0 deliberately. The threshold is not a quality filter here: any
# higher value can return no neighbour at all for a dissimilar molecule, leaving
# it with nowhere to be placed. Selectivity comes from top_k, not the threshold.
_THRESHOLD = 0.0


class TMAPArtifact:
    def __init__(self, dir_name: str, n_workers: int = 1):
        """
        Parameters
        ----------
        dir_name : str
            Base directory containing the fitted 'tmap' folder.
        n_workers : int, default=1
            Worker threads used for each similarity search.
        """
        self.dir_name = os.path.abspath(dir_name)
        self.n_workers = n_workers
        proj_path = os.path.join(self.dir_name, "tmap")

        fps_path = os.path.join(proj_path, "fps.h5")
        coords_path = os.path.join(proj_path, "ref_coords.npy")
        for path in (fps_path, coords_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TMAP artifact not found: {path}\n"
                    f"Fit a reference space with lazychemvis_fit before transforming."
                )

        self.engine = FPSim2Engine(fps_path, in_memory_fps=True)
        self.ref_coords = np.load(coords_path)

    def transform(self, smiles_list: List[str]):
        """
        Place molecules on the TMAP landscape by nearest-neighbour inheritance.

        Parameters
        ----------
        smiles_list : List[str]
            Molecules to project.

        Returns
        -------
        numpy.ndarray of shape (n_molecules, 2)
            Coordinates of each molecule's nearest reference neighbour. Rows for
            molecules that could not be parsed, or for which no neighbour was
            found, are NaN — so the output always has one row per input, aligned
            with the other projections.
        """
        n_dim = self.ref_coords.shape[1]
        # Preserve the stored precision, promoted to at least float32 so that
        # unplaceable molecules can be represented as NaN.
        dtype = np.promote_types(self.ref_coords.dtype, np.float32)
        coords = np.full((len(smiles_list), n_dim), np.nan, dtype=dtype)
        failures = []

        for i, smi in enumerate(smiles_list):
            # Parse here rather than letting FPSim2 hand a None straight to C++,
            # which raises an ArgumentError naming neither the molecule nor its row.
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                failures.append((i, smi))
                continue

            # top_k(k=1) rather than similarity(): the latter returns and sorts
            # every molecule in the reference set just to read the first row.
            results = self.engine.top_k(
                mol, 1, _THRESHOLD, n_workers=self.n_workers
            )
            if len(results) == 0:
                failures.append((i, smi))
                continue

            coords[i] = self.ref_coords[int(results[0]["mol_id"])]

        if failures:
            n_failed = len(failures)
            logger.warning(
                f"TMAP: {n_failed:,} of {len(smiles_list):,} molecules could not "
                f"be placed; their coordinates are NaN."
            )
            examples = ", ".join(f"row {i + 2}: {smi!r}" for i, smi in failures[:_N_EXAMPLES])
            if n_failed > _N_EXAMPLES:
                examples += f", … (+{n_failed - _N_EXAMPLES:,} more)"
            console.print(
                f"  [bold yellow]![/bold yellow] TMAP could not place [bold]{n_failed:,}[/bold] "
                f"of {len(smiles_list):,} molecules; their coordinates are NaN.\n"
                f"    {examples}",
                style="yellow",
            )

        return coords
