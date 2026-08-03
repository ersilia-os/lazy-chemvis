import os
import gc
import argparse
import tmap as tm
import numpy as np


DEFAULT_K = 100
DEFAULT_KC = 1000

LOW_MEMORY_DEFAULT_K = 40
LOW_MEMORY_DEFAULT_KC = 10


def generate_tmap_coords(input_path: str, n_permutations: int = 128, batch_size: int = 10000,
                         k: int = DEFAULT_K, kc: int = DEFAULT_KC):
    """
    Generate TMAP coordinates with memory optimization for large datasets.

    Parameters
    ----------
    input_path : str
        Path to input X.npy file
    n_permutations : int
        Number of LSH permutations (default: 128)
    batch_size : int
        Number of molecules to process at once (default: 10000)
    k : int
        Number of nearest neighbours used to build the k-NN graph (default: 100).
        Higher values create more edges between distant clusters, pulling
        otherwise-floating branches towards the main body of the map.
    kc : int
        Node-connectivity factor for the layout (default: 1000).

    Returns
    -------
    coords : np.ndarray
        2D coordinates normalized to [-1, 1]
    s : np.ndarray
        Source indices for edges
    t : np.ndarray
        Target indices for edges
    """
    print(f"[TMAP] Loading data from {input_path}")
    
    # 1. Load data using memory mapping to avoid loading entire array
    X_raw = np.load(input_path, mmap_mode='r')  # Memory-mapped, not fully loaded
    n_nodes, d = X_raw.shape
    print(f"[TMAP] Dataset: {n_nodes} molecules, {d} features")
    
    # 2. Compute global min/max for scaling (with batching to save memory)
    print(f"[TMAP] Computing scaling parameters...")
    X_min = np.inf
    X_max = -np.inf
    
    for i in range(0, n_nodes, batch_size):
        batch = X_raw[i:i+batch_size]
        X_min = min(X_min, batch.min())
        X_max = max(X_max, batch.max())
        del batch
        gc.collect()
    
    print(f"[TMAP] Scale range: [{X_min}, {X_max}]")
    
    # 3. Initialize LSH Forest
    print(f"[TMAP] Building LSH Forest with {n_permutations} permutations...")
    lf = tm.LSHForest(d, n_permutations)
    
    # 4. Add vectors in batches
    for i in range(0, n_nodes, batch_size):
        end_idx = min(i + batch_size, n_nodes)
        batch = X_raw[i:end_idx]
        
        # Scale and convert batch
        X_batch_scaled = ((batch - X_min) / (X_max - X_min) * 100).astype(np.uint32)
        
        # Add to forest
        for row in X_batch_scaled:
            lf.add(tm.VectorUint(row))
        
        del batch, X_batch_scaled
        gc.collect()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"[TMAP] Processed {end_idx}/{n_nodes} molecules ({100*end_idx/n_nodes:.1f}%)")
    
    # 5. Index the forest
    print(f"[TMAP] Indexing LSH Forest...")
    lf.index()
    
    # Clear the memory-mapped array
    del X_raw
    gc.collect()
    
    # 6. Calculate Layout (this is the memory-intensive part)
    print(f"[TMAP] Computing layout (this may take a while)...")
    cfg = tm.LayoutConfiguration()
# --- KEY PARAMETERS FOR UNIFIED MAP ---

    # 1. k (Neighbors): The most important setting.
    # TMAP's own default is 10. Raising it forces more edges between
    # distant "islands" and the mainland.
    print(f"[TMAP] Layout connectivity: k={k}, kc={kc}")
    cfg.k = k
    cfg.kc = kc

    # 2. Increase Repeats: Allows the layout engine more time to pull
    # floating branches into the center.
    cfg.mmm_repeats = 1 
    cfg.sl_repeats = 1
    
    # 4. Node Size: Smaller nodes relative to the map can help clustering.
    cfg.node_size = 1 / 100 
    
    # For very large datasets, you might want to reduce these:
    if n_nodes > 500000:
        print(f"[TMAP] Large dataset detected, using faster settings...")
        cfg.mmm_repeats = 1
        cfg.sl_repeats = 1
    
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    
    print(f"[TMAP] Layout complete. {len(s)} edges generated.")
    
    # 7. Normalize coordinates to [-1, 1]
    print(f"[TMAP] Normalizing coordinates...")
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    def normalize_to_range(arr, target_min=-1.0, target_max=1.0):
        arr_min, arr_max = arr.min(), arr.max()
        return (arr - arr_min) / (arr_max - arr_min) * (target_max - target_min) + target_min
    
    x_norm = normalize_to_range(x)
    y_norm = normalize_to_range(y)
    
    coords = np.column_stack((x_norm, y_norm))
    
    # Convert edges to arrays
    s = np.array(s, dtype=np.uint32)
    t = np.array(t, dtype=np.uint32)
    
    print(f"[TMAP] Coordinate generation complete.")
    return coords, s, t


def generate_tmap_coords_low_memory(input_path: str, n_permutations: int = 64,
                                    k: int = LOW_MEMORY_DEFAULT_K,
                                    kc: int = LOW_MEMORY_DEFAULT_KC):
    """
    Ultra-low memory version for datasets > 1M molecules.

    Sacrifices some quality for memory efficiency:
    - Fewer permutations (64 instead of 128)
    - Reduced layout quality settings (lower k, larger nodes)
    - Aggressive garbage collection

    Use this if the standard version runs out of memory.

    Parameters
    ----------
    input_path : str
        Path to input X.npy file
    n_permutations : int
        Number of LSH permutations (default: 64)
    k : int
        Number of nearest neighbours for the k-NN graph (default: 40)
    kc : int
        Node-connectivity factor for the layout (default: 10)
    """
    print(f"[TMAP LOW-MEM] Processing {input_path}")
    
    X_raw = np.load(input_path, mmap_mode='r')
    n_nodes, d = X_raw.shape
    print(f"[TMAP LOW-MEM] Dataset: {n_nodes} molecules, {d} features")
    
    # Compute scaling parameters
    print(f"[TMAP LOW-MEM] Computing scale...")
    X_min = X_raw.min()
    X_max = X_raw.max()
    
    # Build forest with reduced permutations
    print(f"[TMAP LOW-MEM] Building LSH Forest (n_permutations={n_permutations})...")
    lf = tm.LSHForest(d, n_permutations)
    
    batch_size = 5000  # Smaller batches
    for i in range(0, n_nodes, batch_size):
        batch = X_raw[i:min(i+batch_size, n_nodes)]
        X_batch = ((batch - X_min) / (X_max - X_min) * 100).astype(np.uint32)
        for row in X_batch:
            lf.add(tm.VectorUint(row))
        del batch, X_batch
        if i % 50000 == 0:
            gc.collect()
    
    print(f"[TMAP LOW-MEM] Indexing...")
    lf.index()
    del X_raw
    gc.collect()
    
    # Minimal layout settings
    print(f"[TMAP LOW-MEM] Computing layout (minimal settings)...")
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 50  # Larger nodes = fewer calculations
    cfg.mmm_repeats = 1
    cfg.sl_repeats = 1
    print(f"[TMAP LOW-MEM] Layout connectivity: k={k}, kc={kc}")
    cfg.k = k  # Reduce number of nearest neighbors
    cfg.kc = kc
    
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    
    print(f"[TMAP LOW-MEM] Normalizing...")
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    y = (y - y.min()) / (y.max() - y.min()) * 2 - 1
    
    coords = np.column_stack((x, y))
    s = np.array(s, dtype=np.uint32)
    t = np.array(t, dtype=np.uint32)
    
    print(f"[TMAP LOW-MEM] Complete.")
    return coords, s, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized TMAP Data Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to input X.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--low_memory", action="store_true", 
                        help="Use ultra-low memory mode (faster but lower quality)")
    parser.add_argument("--n_permutations", type=int, default=128,
                        help="Number of LSH permutations (default: 128, low-mem: 64)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for processing (default: 10000)")
    parser.add_argument("--k", type=int, default=None,
                        help=f"Nearest neighbours for the k-NN graph "
                             f"(default: {DEFAULT_K}, low-mem: {LOW_MEMORY_DEFAULT_K})")
    parser.add_argument("--kc", type=int, default=None,
                        help=f"Node-connectivity factor for the layout "
                             f"(default: {DEFAULT_KC}, low-mem: {LOW_MEMORY_DEFAULT_KC})")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Choose memory mode. Unspecified k / kc fall back to the default for
        # the selected mode, so behaviour is unchanged unless explicitly overridden.
        if args.low_memory:
            print("=" * 60)
            print("RUNNING IN LOW-MEMORY MODE")
            print("=" * 60)
            coords, s, t = generate_tmap_coords_low_memory(
                args.input,
                n_permutations=min(args.n_permutations, 64),
                k=LOW_MEMORY_DEFAULT_K if args.k is None else args.k,
                kc=LOW_MEMORY_DEFAULT_KC if args.kc is None else args.kc,
            )
        else:
            coords, s, t = generate_tmap_coords(
                args.input,
                n_permutations=args.n_permutations,
                batch_size=args.batch_size,
                k=DEFAULT_K if args.k is None else args.k,
                kc=DEFAULT_KC if args.kc is None else args.kc,
            )
        
        # Save results
        reduced_path = os.path.join(args.output_dir, "reduced.npy")
        np.save(reduced_path, coords.astype(np.float32))
        print(f"[SAVE] Saved coordinates to {reduced_path}")
        
        # Save edges (compressed to save disk space)
        edges_path = os.path.join(args.output_dir, "edges.npz")
        np.savez_compressed(edges_path, s=s, t=t)
        print(f"[SAVE] Saved edges to {edges_path}")
        
        print("=" * 60)
        print(f"SUCCESS: Data saved to {args.output_dir}")
        print(f"Molecules: {len(coords)}, Edges: {len(s)}")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print(f"ERROR: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)