"""
One-off migration: build fps.h5 from original SMILES for models that still
have knn_index.joblib. Run once per model; fps.h5 lands next to the existing
ref_coords.npy in the checkpoint tmap dir.

Usage:
    python scripts/migrate_tmap_to_fpsim2.py \
        --smiles  /home/marina/models/training/merged/Enamine_Hit_Locator_460K.csv \
        --out-dir /home/marina/models/models_refactor/eos1klk/model/checkpoints/tmap
"""
import argparse
import os

import pandas as pd
from FPSim2.io import create_db_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles",  required=True, help="CSV with 'smiles' column")
    p.add_argument("--out-dir", required=True, help="Destination tmap checkpoint dir (ref_coords.npy already there)")
    args = p.parse_args()

    df = pd.read_csv(args.smiles)
    col = next(c for c in df.columns if c.lower() == "smiles")
    smiles = df[col].tolist()
    print(f"Loaded {len(smiles):,} SMILES")

    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = os.path.join(args.out_dir, "fps.h5")

    print("Building FPSim2 database (radius=2, nBits=2048)...")
    mols = [(smi, i) for i, smi in enumerate(smiles)]
    create_db_file(mols, out_h5, "smiles", "Morgan", {"radius": 2, "fpSize": 2048})
    print(f"Saved: {out_h5}")


if __name__ == "__main__":
    main()
