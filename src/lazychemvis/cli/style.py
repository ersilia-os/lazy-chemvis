"""
Look of the ``lazychemvis`` help screens.

Rendered by rich-click, in the same palette as the pipeline's console output
(:data:`lazychemvis.helpers.tracker.STEP_COLORS`), so ``--help`` and a running fit
look like one tool.
"""

import rich_click as click

HELP_CONFIG = click.RichHelpConfiguration(
    header_text=(
        "[bold cyan]LazyChemVis[/] [dim]·[/] automated 2D visualizations of chemical spaces"
    ),
    footer_text="[dim]Docs and issues:[/] [link]https://github.com/ersilia-os/lazy-chemvis[/]",
    style_usage="bold yellow",
    style_usage_command="bold magenta",
    style_command="bold magenta",
    style_option="bold cyan",
    style_switch="bold green",
    style_metavar="yellow",
    style_required_short="red",
    style_required_long="dim red",
    # The "Required" panel and its red asterisks already say it; no "[required]" suffix.
    required_long_string="",
    style_options_panel_border="cyan",
    style_commands_panel_border="magenta",
    style_errors_panel_border="red",
    text_markup="rich",
)
