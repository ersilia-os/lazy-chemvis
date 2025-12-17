import os
import argparse
import tmap as tm
import numpy as np


def generate_tmap_coords(input_path: str, n_permutations: int = 128):
    # 1. Load data
    X_raw = np.load(input_path)
    n_nodes, d = X_raw.shape

    # 2. Scale and convert to Uint32 for TMAP LSH
    X_min, X_max = X_raw.min(), X_raw.max()
    X_uint = ((X_raw - X_min) / (X_max - X_min) * 100).astype(np.uint32)

    # 3. Initialize and Index Forest
    lf = tm.LSHForest(d, n_permutations)
    for row in X_uint:
        lf.add(tm.VectorUint(row))
    lf.index()
    
    # 4. Calculate Layout
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 70
    cfg.mmm_repeats = 2
    cfg.sl_repeats = 2

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    
    # 5. Normalize coordinates to [-1, 1] for ScatterPlot compatibility
    x = np.array(x)
    y = np.array(y)
    
    def normalize_to_range(arr, target_min=-1.0, target_max=1.0):
        return (arr - arr.min()) / (arr.max() - arr.min()) * (target_max - target_min) + target_min

    x_norm = normalize_to_range(x)
    y_norm = normalize_to_range(y)
    
    # Return 2D array for reduced.npy and edge arrays
    return np.column_stack((x_norm, y_norm)), np.array(s), np.array(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TMAP Data Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to input X.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        coords, s, t = generate_tmap_coords(args.input)
        
        # Save as reduced.npy (Shape: [n_samples, 2]) for ScatterPlot class
        reduced_path = os.path.join(args.output_dir, "reduced.npy")
        np.save(reduced_path, coords.astype(np.float32))
        
        # Save edges separately in case a future tree-plotter needs them
        edges_path = os.path.join(args.output_dir, "edges.npz")
        np.savez(edges_path, s=s, t=t)
        
        print(f"SUCCESS: Data saved to {args.output_dir}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
