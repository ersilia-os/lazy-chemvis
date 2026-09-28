import rich_click as click


@click.command(short_help="Fit a reference chemical space.")
@click.option_panel("Required", options=["input_path", "reference_path", "tmap_env"])
@click.option_panel("Options", options=["no_cache", "low_memory", "no_report", "verbose", "help"])
@click.option("-i", "--input", "input_path", required=True, metavar="CSV",
              help="CSV of reference SMILES: a header row and SMILES in the first column.")
@click.option("-r", "--reference", "reference_path", required=True, metavar="DIR",
              help="Directory in which the fitted reference space is written.")
@click.option("--tmap-env", required=True, metavar="DIR",
              help="Path to the TMAP conda environment directory (not its name); "
                   "see 'conda env list'.")
@click.option("--no-cache", is_flag=True,
              help="Recompute every featurizer output instead of reusing the ones "
                   "already in the reference directory.")
@click.option("--low-memory", is_flag=True,
              help="Lighter TMAP settings for very large libraries (above ~1M molecules).")
@click.option("--no-report", is_flag=True, help="Skip building the HTML report.")
@click.option("-v", "--verbose", is_flag=True, help="Show the full log on the console.")
def fit(input_path, reference_path, tmap_env, no_cache, low_memory, no_report, verbose):
    """
    Fit a reference chemical space from a library of molecules.

    Computes the PCA, TMAP, t-SNE and UMAP reference maps, trains the surrogates
    that "lazychemvis transform" uses to place new molecules, and writes an HTML
    report.
    """
    # Imported here so that `lazychemvis --help` stays fast.
    from ...fit import Pipeline
    from ...helpers import logger as log_manager

    log_manager.set_verbosity(verbose)
    Pipeline(
        lib_input=input_path,
        dir_path=reference_path,
        tmap_env=tmap_env,
        no_cache=no_cache,
        low_memory=low_memory,
        verbose=verbose,
        no_report=no_report,
    ).run()
