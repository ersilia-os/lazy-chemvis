"""
Shared SMILES validation.

Every featurizer in the fit pipeline produces a matrix whose rows must line up
one-to-one with the reference SMILES list, because projectors and surrogates
index across those matrices (e.g. the t-SNE surrogate maps ``ecfp/X.npy`` rows
onto ``tsne/reduced.npy`` rows). Validating once, before any featurization,
guarantees that alignment instead of leaving each featurizer to apply its own
policy for unparseable input.
"""

from rdkit import Chem
from rdkit import RDLogger

from .logger import get_logger, console

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)

# Number of offending entries quoted in the warning message
_N_EXAMPLES = 5


def validate_smiles(smiles_list, quiet: bool = False):
    """
    Filter a SMILES list down to the entries RDKit can parse.

    Parameters
    ----------
    smiles_list : list of str
        Candidate SMILES strings.
    quiet : bool, default=False
        If True, do not print a warning when entries are dropped.

    Returns
    -------
    valid_smiles : list of str
        The entries RDKit parsed successfully, in their original order.
    valid_indices : list of int
        Position of each retained entry in the input list, so callers can map
        results back onto their original rows.

    Raises
    ------
    ValueError
        If no entry could be parsed.
    """
    valid_smiles = []
    valid_indices = []
    invalid = []

    for i, smi in enumerate(smiles_list):
        if smi is None or not str(smi).strip():
            invalid.append((i, smi))
            continue
        if Chem.MolFromSmiles(smi) is None:
            invalid.append((i, smi))
            continue
        valid_smiles.append(smi)
        valid_indices.append(i)

    n_total = len(smiles_list)
    n_invalid = len(invalid)

    if not valid_smiles:
        raise ValueError(
            f"None of the {n_total:,} input molecules could be parsed by RDKit. "
            "Check that the first column of the input CSV contains valid SMILES."
        )

    if n_invalid:
        logger.warning(f"Dropped {n_invalid:,} of {n_total:,} unparseable molecules.")
        if not quiet:
            # Routed through the Rich console because loguru output is suppressed
            # package-wide, and silently discarding molecules is not acceptable.
            examples = ", ".join(
                f"row {i + 2}: {smi!r}" for i, smi in invalid[:_N_EXAMPLES]
            )
            if n_invalid > _N_EXAMPLES:
                examples += f", … (+{n_invalid - _N_EXAMPLES:,} more)"
            console.print(
                f"  [bold yellow]![/bold yellow] Dropped [bold]{n_invalid:,}[/bold] of "
                f"{n_total:,} molecules that RDKit could not parse "
                f"([bold]{len(valid_smiles):,}[/bold] retained).\n"
                f"    {examples}",
                style="yellow",
            )

    return valid_smiles, valid_indices
