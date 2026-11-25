import rich_click as click


@click.command(short_help="Project molecules onto a fitted reference space.")
@click.option_panel("Required", options=["input_path", "reference_path", "output_path"])
@click.option_panel("Options", options=["no_plots", "no_report", "verbose", "help"])
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    metavar="CSV",
    help="CSV of molecules to project: a header row and SMILES in the first column.",
)
@click.option(
    "-r",
    "--reference",
    "reference_path",
    required=True,
    metavar="DIR",
    help="Directory of a previously fitted reference space.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    metavar="DIR",
    help="Directory for the output coordinates, figures and report.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Write only coordinates.csv, skipping the figures and the report.",
)
@click.option(
    "--no-report", is_flag=True, help="Write the figures but skip the HTML report."
)
@click.option("-v", "--verbose", is_flag=True, help="Show the full log on the console.")
def transform(input_path, reference_path, output_path, no_plots, no_report, verbose):
    """
    Project new molecules onto a fitted reference space.

    Writes coordinates.csv with one row per input molecule for each of the four
    projections, plus the figures and an HTML report.
    """
    # Imported here so that `lazychemvis --help` stays fast.
    from ...helpers import logger as log_manager
    from ...transform import Pipeline

    log_manager.set_verbosity(verbose)
    Pipeline(
        lib_input=input_path,
        dir_path=reference_path,
        output_path=output_path,
        no_plots=no_plots,
        no_report=no_report,
    ).run()
