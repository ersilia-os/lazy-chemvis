"""
Input library loading.

A library is a CSV file with a header row and SMILES in the first column. Any
further columns are ignored.
"""

import os
import csv


def load_any_library(smiles_path: str):
    """
    Read SMILES from the first column of a CSV file, skipping the header row.

    Parameters
    ----------
    smiles_path : str
        Path to the CSV file.

    Returns
    -------
    list of str
        The SMILES strings found in the first column.

    Raises
    ------
    ValueError
        If the file is empty or contains no data rows after the header.
    """
    smiles_path = os.path.abspath(smiles_path)

    with open(smiles_path, "r") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # discard header row
        except StopIteration:
            raise ValueError(
                f"Input library is empty: {smiles_path}\n"
                "Expected a CSV file with a header row and SMILES in the first column."
            )
        smiles_list = [row[0] for row in reader if row]

    if not smiles_list:
        raise ValueError(
            f"No SMILES found in {smiles_path}\n"
            "The file contains a header row but no data rows. Expected a CSV file "
            "with a header row and SMILES in the first column."
        )

    return smiles_list


def load_lib_input(input_value: str):
    """
    Load an input library from a path.

    Parameters
    ----------
    input_value : str
        Path to a CSV file with a header row and SMILES in the first column.

    Returns
    -------
    list of str
        The SMILES strings read from the file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    ValueError
        If the file contains no SMILES.
    """
    if not input_value:
        raise ValueError(
            "No input library given. Pass a path to a CSV file with a header row "
            "and SMILES in the first column via --lib_input."
        )

    if not os.path.exists(input_value):
        raise FileNotFoundError(
            f"Input library not found: {input_value}\n"
            "--lib_input must be a path to an existing CSV file with a header row "
            "and SMILES in the first column."
        )

    if not os.path.isfile(input_value):
        raise FileNotFoundError(
            f"Input library is a directory, not a file: {input_value}\n"
            "--lib_input must be a path to a CSV file with a header row and SMILES "
            "in the first column."
        )

    return load_any_library(input_value)
