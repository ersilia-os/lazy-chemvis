"""
The ``lazychemvis`` command-line interface.

One Click group with a subcommand per file in :mod:`lazychemvis.cli.commands`,
following the layout of ``ersilia``'s own CLI. Help screens are rendered with
rich-click; their styling lives in :mod:`lazychemvis.cli.style`.
"""

import rich_click as click

from .commands.fit import fit
from .commands.transform import transform
from .style import HELP_CONFIG


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.rich_config(help_config=HELP_CONFIG)
@click.version_option(package_name="lazychemvis", prog_name="lazychemvis")
def cli():
    """Fit a reference chemical space once, then project any new molecules onto it."""


cli.add_command(fit)
cli.add_command(transform)


def main():
    """Entry point for the ``lazychemvis`` executable."""
    cli()


if __name__ == "__main__":
    main()
