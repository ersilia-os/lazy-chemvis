import os
import csv


def load_any_library(smiles_path: str):
    smiles_path = os.path.abspath(smiles_path)
    with open(smiles_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        smiles_list = [row[0] for row in reader]
    return smiles_list


def load_reference_library(lib_name: str):
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "data",
        lib_name + ".csv"
    )
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        smiles_list = [row[0] for row in reader]
    return smiles_list


def load_lib_input(input_value: str):
    if os.path.isfile(input_value):
        smiles_list = load_any_library(input_value)
    else:
        smiles_list = load_reference_library(input_value)
    return smiles_list