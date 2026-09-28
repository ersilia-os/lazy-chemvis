"""
Colour system for the report.

Semantic keys resolved through stylia, so the figures and the HTML share one
palette and match the rest of Ersilia's output. Nothing here hardcodes a hex
value that stylia already owns.
"""

import stylia

_named = stylia.NamedColors()

# Semantic key → stylia colour name.
_SEMANTIC = {
    "reference": "silver",     # the grey background landscape
    "overlay": "periwinkle",   # newly projected molecules
    "truth": "cobalt",         # ground-truth coordinates in validation figures
    "prediction": "crimson",   # surrogate-predicted coordinates
    "good": "lime",
    "warn": "amber",
    "bad": "crimson",
    "neutral": "black",
}

# Accent colour per projection, mirroring the console step colours.
_PROJECTION = {
    "pca": "cobalt",
    "tmap": "orchid",
    "tsne": "turquoise",
    "umap": "amber",
}


def rgb(key: str, alpha: float = None, lighten: float = None):
    """
    Return the RGB(A) tuple for a semantic key or a stylia colour name.

    Parameters
    ----------
    key : str
        A semantic key (e.g. 'overlay') or a stylia name (e.g. 'crimson').
    alpha, lighten : float, optional
        Passed through to stylia.
    """
    name = _SEMANTIC.get(key, key)
    if alpha is None and lighten is None:
        return getattr(_named, name)
    return _named.get(name, alpha=alpha, lighten=lighten)


def hexcol(key: str) -> str:
    """Return the hex string for a semantic key or stylia colour name."""
    name = _SEMANTIC.get(key, key)
    return _named.hex[name]


def projection_rgb(projection: str):
    """RGB tuple for a projection ('pca', 'tmap', 'tsne', 'umap')."""
    return getattr(_named, _PROJECTION.get(projection, "cobalt"))


def projection_hex(projection: str) -> str:
    """Hex string for a projection, used for the HTML badges."""
    return _named.hex[_PROJECTION.get(projection, "cobalt")]


#: Hex colour per projection, consumed by the HTML report.
PROJECTION_COLORS = {k: _named.hex[v] for k, v in _PROJECTION.items()}


def categorical_rgb(n: int):
    """Return ``n`` visually distinct RGB tuples from stylia's categorical palette."""
    return stylia.CategoricalPalette().get(n)


def grey_ramp():
    """
    Return the light→dark grey ramp used to shade the reference density.

    Kept explicit rather than derived from stylia: this is a monotonic
    lightness ramp for datashader's ``eq_hist`` shading, not a categorical
    palette, and the original tuning of these four steps is worth preserving.
    """
    return ["#e7e2e2", "#cac8c8", "#B1B0B0", "#989797"]
