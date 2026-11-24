"""
TMAP layout driver, run as a subprocess inside the TMAP conda environment.

Fingerprints are MinHashed before they are indexed: ``tm.LSHForest`` expects
MinHash signatures, and its distance is the Jaccard distance estimated from
them. The k-NN graph behind the layout is therefore built on Jaccard (Tanimoto)
similarity between ECFP fingerprints, as in the original TMAP method.

Not imported by the main package (``tmap`` only exists in the TMAP environment);
:mod:`lazychemvis.projectors.tmap_projector` duplicates the defaults it needs.
"""

import argparse
import gc
import os

import numpy as np
import tmap as tm

DEFAULT_K = 100
DEFAULT_KC = 1000

LOW_MEMORY_DEFAULT_K = 40
LOW_MEMORY_DEFAULT_KC = 10

# Number of prefix trees in the LSH forest (tmap's own default). It must divide
# the number of MinHash permutations: each tree indexes an equal slice of them.
DEFAULT_N_TREES = 8

# Seed of the MinHash permutations, so the signatures are reproducible.
MINHASH_SEED = 42


def build_lsh_forest(
    input_path: str,
    n_permutations: int,
    n_trees: int = DEFAULT_N_TREES,
    batch_size: int = 10000,
    tag: str = "[TMAP]",
):
    """
    MinHash a binary fingerprint matrix and index it in an LSH forest.

    Parameters
    ----------
    input_path : str
        Path to an ``X.npy`` of binary fingerprints, one row per molecule.
    n_permutations : int
        Number of MinHash permutations, i.e. the length of each signature.
    n_trees : int, default=8
        Number of prefix trees in the forest. Must divide ``n_permutations``.
    batch_size : int, default=10000
        Rows read from the memory-mapped input at a time.
    tag : str, default="[TMAP]"
        Prefix of the progress lines.

    Returns
    -------
    lf : tmap.LSHForest
        The indexed forest.
    n_nodes : int
        Number of molecules added.

    Raises
    ------
    ValueError
        If ``n_trees`` does not divide ``n_permutations``, or the input is not
        a binary matrix.
    """
    if n_permutations % n_trees != 0:
        raise ValueError(
            f"n_trees ({n_trees}) must divide n_permutations ({n_permutations})."
        )

    print(f"{tag} Loading data from {input_path}")
    X_raw = np.load(input_path, mmap_mode="r")  # memory-mapped, not fully loaded
    n_nodes, d = X_raw.shape
    print(f"{tag} Dataset: {n_nodes} molecules, {d} bits")

    print(
        f"{tag} MinHash: {n_permutations} permutations; "
        f"LSH forest: {n_trees} prefix trees"
    )
    enc = tm.Minhash(n_permutations, seed=MINHASH_SEED)
    lf = tm.LSHForest(n_permutations, n_trees)

    for i in range(0, n_nodes, batch_size):
        end_idx = min(i + batch_size, n_nodes)
        batch = np.asarray(X_raw[i:end_idx])

        # The signature must be computed from the set of on-bits. Values other
        # than 0/1 would be silently treated as "on", so refuse them instead.
        if batch.size and (batch.min() < 0 or batch.max() > 1):
            raise ValueError(
                f"Expected binary fingerprints, found values in "
                f"[{batch.min()}, {batch.max()}] in rows {i}-{end_idx}."
            )

        # The sparse encoder takes on-bit indices and gives the same signature
        # as the dense one, at a fraction of the cost for sparse fingerprints.
        on_bits = [
            tm.VectorUint(np.flatnonzero(row).astype(np.uint32)) for row in batch
        ]
        lf.batch_add(enc.batch_from_sparse_binary_array(on_bits))

        del batch, on_bits
        if (i // batch_size + 1) % 10 == 0:
            gc.collect()
            print(
                f"{tag} Processed {end_idx}/{n_nodes} molecules "
                f"({100 * end_idx / n_nodes:.1f}%)"
            )

    del X_raw
    gc.collect()

    print(f"{tag} Indexing LSH Forest...")
    lf.index()
    return lf, n_nodes


def _normalize_to_range(arr, target_min=-1.0, target_max=1.0):
    """Min-max scale ``arr`` onto ``[target_min, target_max]``."""
    arr_min, arr_max = arr.min(), arr.max()
    return (arr - arr_min) / (arr_max - arr_min) * (
        target_max - target_min
    ) + target_min


def _layout(lf, cfg, tag):
    """Lay out the forest's k-NN graph and return [-1, 1] coordinates and edges."""
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    print(f"{tag} Layout complete. {len(s)} edges generated.")

    print(f"{tag} Normalizing coordinates...")
    x = _normalize_to_range(np.array(x, dtype=np.float32))
    y = _normalize_to_range(np.array(y, dtype=np.float32))
    coords = np.column_stack((x, y))

    s = np.array(s, dtype=np.uint32)
    t = np.array(t, dtype=np.uint32)
    return coords, s, t


def generate_tmap_coords(
    input_path: str,
    n_permutations: int = 128,
    n_trees: int = DEFAULT_N_TREES,
    batch_size: int = 10000,
    k: int = DEFAULT_K,
    kc: int = DEFAULT_KC,
):
    """
    Generate TMAP coordinates for a binary fingerprint matrix.

    Parameters
    ----------
    input_path : str
        Path to input X.npy file
    n_permutations : int
        Number of MinHash permutations (default: 128)
    n_trees : int
        Number of LSH forest prefix trees (default: 8)
    batch_size : int
        Number of molecules MinHashed at once (default: 10000)
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
    tag = "[TMAP]"
    lf, n_nodes = build_lsh_forest(
        input_path, n_permutations, n_trees=n_trees, batch_size=batch_size, tag=tag
    )

    print(f"{tag} Computing layout (this may take a while)...")
    cfg = tm.LayoutConfiguration()
    # k is the most important setting: TMAP's own default is 10, and raising it
    # adds edges between distant "islands" and the mainland.
    print(f"{tag} Layout connectivity: k={k}, kc={kc}")
    cfg.k = k
    cfg.kc = kc
    cfg.mmm_repeats = 1
    cfg.sl_repeats = 1
    # Smaller nodes relative to the map help clustering.
    cfg.node_size = 1 / 100

    coords, s, t = _layout(lf, cfg, tag)
    print(f"{tag} Coordinate generation complete.")
    return coords, s, t


def generate_tmap_coords_low_memory(
    input_path: str,
    n_permutations: int = 64,
    n_trees: int = DEFAULT_N_TREES,
    k: int = LOW_MEMORY_DEFAULT_K,
    kc: int = LOW_MEMORY_DEFAULT_KC,
):
    """
    Lower-memory version for datasets > 1M molecules.

    Sacrifices some quality for memory efficiency:
    - Fewer MinHash permutations (64 instead of 128)
    - Reduced layout quality settings (lower k and kc, larger nodes)
    - Smaller batches

    Use this if the standard version runs out of memory.

    Parameters
    ----------
    input_path : str
        Path to input X.npy file
    n_permutations : int
        Number of MinHash permutations (default: 64)
    n_trees : int
        Number of LSH forest prefix trees (default: 8)
    k : int
        Number of nearest neighbours for the k-NN graph (default: 40)
    kc : int
        Node-connectivity factor for the layout (default: 10)
    """
    tag = "[TMAP LOW-MEM]"
    lf, n_nodes = build_lsh_forest(
        input_path, n_permutations, n_trees=n_trees, batch_size=5000, tag=tag
    )

    print(f"{tag} Computing layout (minimal settings)...")
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 50  # Larger nodes = fewer calculations
    cfg.mmm_repeats = 1
    cfg.sl_repeats = 1
    print(f"{tag} Layout connectivity: k={k}, kc={kc}")
    cfg.k = k
    cfg.kc = kc

    coords, s, t = _layout(lf, cfg, tag)
    print(f"{tag} Complete.")
    return coords, s, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized TMAP Data Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to input X.npy")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Use ultra-low memory mode (faster but lower quality)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128, low-mem: 64)",
    )
    parser.add_argument(
        "--n_trees",
        type=int,
        default=DEFAULT_N_TREES,
        help=f"Number of LSH forest prefix trees (default: {DEFAULT_N_TREES})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Nearest neighbours for the k-NN graph "
        f"(default: {DEFAULT_K}, low-mem: {LOW_MEMORY_DEFAULT_K})",
    )
    parser.add_argument(
        "--kc",
        type=int,
        default=None,
        help=f"Node-connectivity factor for the layout "
        f"(default: {DEFAULT_KC}, low-mem: {LOW_MEMORY_DEFAULT_KC})",
    )
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
                n_trees=args.n_trees,
                k=LOW_MEMORY_DEFAULT_K if args.k is None else args.k,
                kc=LOW_MEMORY_DEFAULT_KC if args.kc is None else args.kc,
            )
        else:
            coords, s, t = generate_tmap_coords(
                args.input,
                n_permutations=args.n_permutations,
                n_trees=args.n_trees,
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
