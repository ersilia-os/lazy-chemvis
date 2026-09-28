"""
HTML report assembly.

Plain f-strings with one inlined stylesheet — no templating engine, matching
zairachem. Figures are referenced as relative ``png/<stem>.png`` links rather than
base64-embedded, which keeps the document small; the consequence is that
``report.html`` must travel with its ``png/`` folder.

The layout is a sticky sidebar plus a responsive card grid, printable via an
``@media print`` block so the report doubles as a PDF-able run record.
"""

import html as _html
import os

from ..helpers.logger import get_logger
from . import load_figure_cells
from .colors import PROJECTION_COLORS, hexcol
from .fetcher import PROJECTION_DESCRIPTIONS, PROJECTION_LABELS, PROJECTIONS
from .perf import summarise

logger = get_logger(__name__)

REPORT_FILENAME = "report.html"

_CSS = """
:root{
  --fg:#1d2430; --muted:#66707d; --line:#e4e7ec; --bg:#ffffff;
  --sidebar:#fafbfc; --card:#ffffff; --accent:#457B9D;
  --good:#6BBF59; --warn:#FCBF49; --bad:#E63946;
  --card-h:320px;
}
*{box-sizing:border-box;}
body{
  margin:0; color:var(--fg); background:var(--bg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.55;
}
a{color:var(--accent);}
.layout{display:flex; align-items:flex-start; max-width:1500px; margin:0 auto;}
nav{
  position:sticky; top:0; flex:0 0 220px; height:100vh; overflow-y:auto;
  padding:28px 18px; background:var(--sidebar); border-right:1px solid var(--line);
}
nav h1{font-size:15px; margin:0 0 4px; letter-spacing:.02em;}
nav .sub{font-size:12px; color:var(--muted); margin-bottom:20px;}
nav ol{list-style:none; margin:0; padding:0; font-size:13px;}
nav li{margin:0 0 2px;}
nav a{display:block; padding:5px 8px; border-radius:5px; text-decoration:none; color:var(--fg);}
nav a:hover{background:#eef1f4;}
main{flex:1 1 auto; padding:28px 34px; min-width:0;}
header.run{border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:26px;}
header.run h2{margin:0 0 6px; font-size:22px;}
header.run .meta{color:var(--muted); font-size:13px;}
section{margin:0 0 38px; scroll-margin-top:18px;}
section > h3{
  font-size:16px; margin:0 0 6px; padding-bottom:6px; border-bottom:1px solid var(--line);
}
section > p.lede{color:var(--muted); font-size:13.5px; margin:0 0 16px;}
.grid{display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));}
.card{
  background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:12px 12px 10px; display:flex; flex-direction:column;
}
.card h4{margin:0 0 8px; font-size:13.5px; font-weight:600;}
.card .figwrap{
  height:var(--card-h); display:flex; align-items:center; justify-content:center;
  overflow:hidden; background:#fff;
}
.card img{max-width:100%; max-height:100%; object-fit:contain;}
.card .foot{margin-top:8px; font-size:11.5px; color:var(--muted); display:flex; gap:10px;}
.card.wide{grid-column:1 / -1;}
.card.wide .figwrap{height:auto;}
.badge{
  display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px;
  font-weight:600; color:#fff;
}
.pill{
  display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px;
  border:1px solid var(--line); color:var(--muted);
}
.v-good{color:#3f8f33; font-weight:600;}
.v-warn{color:#9a7213; font-weight:600;}
.v-bad{color:#c1121f; font-weight:600;}
table.data{border-collapse:collapse; width:100%; font-size:13px;}
table.data th,table.data td{
  text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
}
table.data th{font-weight:600; color:var(--muted); font-size:11.5px;
  text-transform:uppercase; letter-spacing:.04em;}
table.data td.num,table.data th.num{text-align:right; font-variant-numeric:tabular-nums;}
.tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:8px;}
.kv{display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px 22px;}
.kv div{border-bottom:1px dotted var(--line); padding-bottom:5px; font-size:13px;}
.kv .k{color:var(--muted); font-size:11.5px; display:block; text-transform:uppercase;
  letter-spacing:.04em;}
.note{
  border-left:3px solid var(--accent); background:#f7f9fb; padding:10px 14px;
  font-size:13px; border-radius:0 6px 6px 0; margin:0 0 14px;
}
.note.caveat{border-left-color:var(--warn); background:#fffaf0;}
.carousel .figwrap{position:relative;}
.carousel .slide{display:none;}
.carousel .slide.on{display:flex; align-items:center; justify-content:center;
  width:100%; height:100%;}
.carousel .ctrl{display:flex; align-items:center; gap:8px; margin-top:8px; font-size:12px;}
.carousel button{
  border:1px solid var(--line); background:#fff; border-radius:5px; cursor:pointer;
  padding:2px 9px; font-size:13px; color:var(--fg);
}
.carousel button:hover{background:#eef1f4;}
footer{
  border-top:1px solid var(--line); margin-top:30px; padding-top:14px;
  font-size:12px; color:var(--muted);
}
@media print{
  nav{display:none;}
  .layout{max-width:none;}
  main{padding:0;}
  .card{break-inside:avoid; page-break-inside:avoid;}
  .card .figwrap{height:auto;}
  .carousel .slide{display:flex !important;}
}
"""

_JS = """
document.querySelectorAll('.carousel').forEach(function(car){
  var slides = car.querySelectorAll('.slide');
  if (!slides.length) return;
  var label = car.querySelector('.count');
  var i = 0;
  function show(n){
    i = (n + slides.length) % slides.length;
    slides.forEach(function(s, k){ s.classList.toggle('on', k === i); });
    if (label) label.textContent = (i + 1) + ' / ' + slides.length;
  }
  var prev = car.querySelector('.prev'), next = car.querySelector('.next');
  if (prev) prev.addEventListener('click', function(){ show(i - 1); });
  if (next) next.addEventListener('click', function(){ show(i + 1); });
  car.addEventListener('keydown', function(e){
    if (e.key === 'ArrowLeft') show(i - 1);
    if (e.key === 'ArrowRight') show(i + 1);
  });
  show(0);
});
"""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _esc(value):
    """HTML-escape a value for safe interpolation."""
    return _html.escape(str(value), quote=True)


def _humanize(stem):
    """Turn a figure stem into a human title ('tsne_fold_1_zones' → 't-SNE fold 1 zones')."""
    parts = stem.split("_")
    out = []
    for i, p in enumerate(parts):
        if p in PROJECTION_LABELS:
            # Never re-case a projection label: capitalising the first character
            # would turn "t-SNE" into "T-SNE".
            out.append(PROJECTION_LABELS[p])
        elif p == "cv":
            out.append("Cross-validation" if i == 0 else "cross-validation")
        elif i == 0:
            out.append(p[:1].upper() + p[1:])
        else:
            out.append(p)
    return " ".join(out)


def _img_src(stem):
    return f"png/{stem}.png"


def _exists(report_dir, stem):
    return os.path.exists(os.path.join(report_dir, "png", f"{stem}.png"))


def _dim_badge(cells):
    """Render a figure's cell footprint, e.g. '3x6 cells'."""
    if not cells:
        return ""
    return f"{cells[0]}×{cells[1]} cells"


def _card(report_dir, stem, cells_map, title=None, wide=False):
    """One figure card. Returns '' if the figure is not on disk."""
    if not _exists(report_dir, stem):
        return ""
    title = title or _humanize(stem)
    cells = cells_map.get(stem)
    pdf_rel = f"pdf/{stem}.pdf"
    pdf_link = (
        f"<a href='{_esc(pdf_rel)}'>pdf</a>"
        if os.path.exists(os.path.join(report_dir, "pdf", f"{stem}.pdf"))
        else ""
    )
    return f"""
      <div class="card{" wide" if wide else ""}">
        <h4>{_esc(title)}</h4>
        <div class="figwrap"><img src="{_esc(_img_src(stem))}" alt="{_esc(title)}"></div>
        <div class="foot"><span>{_esc(_dim_badge(cells))}</span>{pdf_link}</div>
      </div>"""


def _carousel(report_dir, title, stems, cells_map):
    """
    A multi-figure card with prev/next controls.

    Used for the per-fold validation figures: five folds x three diagnostics is
    fifteen figures per surrogate, which as flat cards would bury everything else.
    """
    stems = [s for s in stems if _exists(report_dir, s)]
    if not stems:
        return ""
    slides = "".join(
        f"<div class='slide'><img src='{_esc(_img_src(s))}' alt='{_esc(_humanize(s))}'></div>"
        for s in stems
    )
    return f"""
      <div class="card wide carousel" tabindex="0">
        <h4>{_esc(title)}</h4>
        <div class="figwrap">{slides}</div>
        <div class="ctrl">
          <button class="prev" type="button">‹</button>
          <button class="next" type="button">›</button>
          <span class="count"></span>
          <span class="pill">{len(stems)} figures</span>
        </div>
      </div>"""


def _kv(rows):
    """A responsive key/value grid."""
    cells = "".join(
        f"<div><span class='k'>{_esc(k)}</span>{v}</div>"
        for k, v in rows
        if v is not None
    )
    return f"<div class='kv'>{cells}</div>"


def _projection_badge(projection):
    color = PROJECTION_COLORS.get(projection, hexcol("neutral"))
    label = PROJECTION_LABELS.get(projection, projection)
    return f"<span class='badge' style='background:{color}'>{_esc(label)}</span>"


def _verdict(label, css_class):
    return f"<span class='v-{css_class}'>{_esc(label)}</span>"


def _table(headers, rows, numeric_from=1):
    """Render a data table; columns from ``numeric_from`` onwards are right-aligned."""
    head = "".join(
        f"<th{' class=num' if i >= numeric_from else ''}>{_esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = ""
    for row in rows:
        tds = "".join(
            f"<td{' class=num' if i >= numeric_from else ''}>{c}</td>"
            for i, c in enumerate(row)
        )
        body += f"<tr>{tds}</tr>"
    return (
        f"<div class='tablewrap'><table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def _config_section(manifest, fetcher):
    """Run configuration: what was fitted, from what, with which parameters."""
    n_valid = manifest.get("n_valid") or fetcher.n_reference_molecules()
    n_invalid = manifest.get("n_invalid")
    versions = manifest.get("versions", {})

    rows = [
        ("molecules", f"{n_valid:,}" if n_valid else "—"),
        ("invalid dropped", f"{n_invalid:,}" if n_invalid is not None else "—"),
        (
            "input",
            f"<code>{_esc(os.path.basename(manifest.get('lib_input', '—')))}</code>",
        ),
        ("lazychemvis", _esc(versions.get("lazychemvis", "—"))),
        ("rdkit", _esc(versions.get("rdkit", "—"))),
        ("low memory", "yes" if manifest.get("low_memory") else "no"),
    ]

    per_projection = ""
    param_rows = []
    for p in PROJECTIONS:
        cfg = manifest.get(p, {})
        if not cfg:
            continue
        params = ", ".join(f"{k} = {v}" for k, v in cfg.items())
        param_rows.append(
            (
                f"{_projection_badge(p)} "
                f"<span class='pill'>{_esc(PROJECTION_DESCRIPTIONS.get(p, ''))}</span>",
                params,
            )
        )
    if param_rows:
        per_projection = _table(
            ["Projection", "Parameters"],
            param_rows,
            numeric_from=99,  # both columns left-aligned
        )

    return f"""
    <section id="configuration">
      <h3>Run configuration</h3>
      <p class="lede">What was fitted, and with which parameters.</p>
      {_kv(rows)}
      <div style="height:14px"></div>
      {per_projection}
    </section>"""


def _landscape_section(report_dir, cells_map, fetcher):
    """The four reference landscapes."""
    # The figure carries its own "Chemical space — PCA" title so the PDF stands alone;
    # the card therefore shows only the pipeline description, to avoid repeating the
    # projection name twice in the same card.
    cards = "".join(
        _card(
            report_dir,
            f"{p}_reference_space",
            cells_map,
            title=PROJECTION_DESCRIPTIONS.get(p, PROJECTION_LABELS.get(p, p)),
        )
        for p in PROJECTIONS
    )
    if not cards.strip():
        return ""
    return f"""
    <section id="chemical-space">
      <h3>Chemical space</h3>
      <p class="lede">Reference landscapes, shaded by molecule density.</p>
      <div class="grid">{cards}</div>
    </section>"""


def _surrogate_section(report_dir, cells_map, fetcher):
    """Cross-validated surrogate quality, per projection, plus the caveats."""
    blocks = []
    summary_rows = []

    for projection in ("tsne", "umap"):
        m = summarise(fetcher.metrics(projection))
        if m is None:
            continue
        label = PROJECTION_LABELS.get(projection, projection)
        summary_rows.append(
            [
                _projection_badge(projection),
                f"{m['r2_mean']:.4f} ± {m['r2_std']:.4f}",
                f"{m['rmse_mean']:.4f} ± {m['rmse_std']:.4f}",
                f"{m['mae_mean']:.4f} ± {m['mae_std']:.4f}",
                f"{m['euclidean_mean']:.4f} ± {m['euclidean_std']:.4f}",
                _verdict(m["quality"], m["quality_class"]),
            ]
        )

        fold_rows = [
            [
                str(i + 1),
                f"{m['r2_per_fold'][i]:.4f}",
                f"{m['euclidean_per_fold'][i]:.4f}",
            ]
            for i in range(m["cv_folds"])
        ]
        stability = (
            ""
            if m["stable"]
            else f"<div class='note caveat'>High variance across folds "
            f"(R² std = {m['r2_std']:.4f}).</div>"
        )

        # Ground truth vs prediction maps go last, after the per-axis diagnostics.
        stems = [
            f"{projection}_fold_{i + 1}_{kind}"
            for kind in ("distributions", "zones", "comparison")
            for i in range(m["cv_folds"])
        ]

        blocks.append(f"""
      <h4 style="margin:22px 0 8px">{_projection_badge(projection)} {
            _esc(label)
        } surrogate</h4>
      {
            _kv(
                [
                    ("R² (mean ± std)", f"{m['r2_mean']:.4f} ± {m['r2_std']:.4f}"),
                    (
                        "R² spread (± 1 std)",
                        f"[{m['spread_low']:.4f}, {m['spread_high']:.4f}]",
                    ),
                    (
                        "mean euclidean error",
                        f"{m['euclidean_mean']:.4f} ± {m['euclidean_std']:.4f}",
                    ),
                    ("folds", str(m["cv_folds"])),
                    ("overall quality", _verdict(m["quality"], m["quality_class"])),
                    (
                        "placement accuracy",
                        _verdict(m["accuracy"], m["accuracy_class"]),
                    ),
                ]
            )
        }
      {stability}
      <div style="height:12px"></div>
      {_table(["Fold", "R²", "Euclidean error"], fold_rows)}
      <div style="height:14px"></div>
      <div class="grid">{
            _carousel(report_dir, f"{label} per-fold validation", stems, cells_map)
        }</div>""")

    if not blocks:
        return ""

    overall = _table(
        ["Surrogate", "R²", "RMSE", "MAE", "Euclidean", "Verdict"], summary_rows
    )
    figures = "".join(
        _card(report_dir, stem, cells_map)
        for stem in ("cv_metric_bars", "euclidean_error_histogram")
    )

    return f"""
    <section id="surrogate-quality">
      <h3>Surrogate quality</h3>
      <p class="lede">Cross-validation of the surrogates that place new molecules on the
      t-SNE and UMAP maps. PCA and TMAP have no metrics: their surrogates reuse the
      reference coordinates directly.</p>
      {overall}
      <div style="height:16px"></div>
      <div class="grid">{figures}</div>
      {"".join(blocks)}
    </section>"""


def _performance_section(report_dir, cells_map, fetcher):
    """Wall-clock timings."""
    timings = fetcher.timings()
    if not timings:
        return ""
    rows = [[PROJECTION_LABELS.get(k, k), f"{v:.1f}"] for k, v in timings.items()]
    total = sum(timings.values())
    rows.append(["<strong>total</strong>", f"<strong>{total:.1f}</strong>"])
    return f"""
    <section id="performance">
      <h3>Computational performance</h3>
      <p class="lede">Wall-clock time per pipeline step.</p>
      {_table(["Step", "Seconds"], rows)}
      <div style="height:16px"></div>
      <div class="grid">{_card(report_dir, "step_timing", cells_map)}</div>
    </section>"""


def _transform_input_section(manifest, fetcher):
    """Input summary, including how many SMILES could not be parsed."""
    df = fetcher.coordinates()
    n = 0 if df is None else len(df)
    # TMAP is the one projection that marks an unparseable molecule with NaN, so its
    # NaN count is the number of unparseable SMILES.
    n_bad = fetcher.unplaceable_counts().get("tmap")

    note = ""
    if n_bad:
        note = """
      <div class="note caveat">Unparseable SMILES have NaN TMAP coordinates in
      <code>coordinates.csv</code>; exclude those rows from every projection.</div>"""

    return f"""
    <section id="input">
      <h3>Input</h3>
      <p class="lede">Molecules projected onto the pretrained reference space.</p>
      {
        _kv(
            [
                ("molecules", f"{n:,}"),
                (
                    "input",
                    f"<code>{_esc(os.path.basename(manifest.get('lib_input', '—')))}</code>",
                ),
                (
                    "reference space",
                    f"<code>{_esc(os.path.basename(manifest.get('reference_path', '—')))}</code>",
                ),
                ("unparseable SMILES", f"{n_bad:,}" if n_bad is not None else None),
            ]
        )
    }
      <div style="height:14px"></div>
      {note}
    </section>"""


def _overlay_section(report_dir, cells_map):
    """Overlay figures plus coordinate coverage."""
    overlays = "".join(
        _card(
            report_dir,
            f"{p}_overlay",
            cells_map,
            title=f"{PROJECTION_LABELS.get(p, p)} overlay",
        )
        for p in PROJECTIONS
    )
    density = "".join(
        _card(report_dir, f"{p}_coordinate_density", cells_map, wide=True)
        for p in PROJECTIONS
    )
    if not (overlays.strip() or density.strip()):
        return ""
    return f"""
    <section id="overlays">
      <h3>Where the molecules landed</h3>
      <p class="lede">Input molecules (coloured) over the reference landscape (grey), and
      their coordinate distributions against the reference.</p>
      <div class="grid">{overlays}{density}</div>
    </section>"""


def _coordinates_section(fetcher, top_n=25):
    """A preview of coordinates.csv with a link to the full file."""
    df = fetcher.coordinates()
    if df is None or df.empty:
        return ""
    head = df.head(top_n)
    headers = list(head.columns)

    def fmt(v):
        if isinstance(v, float):
            return "—" if v != v else f"{v:.4f}"  # NaN check without importing numpy
        return _esc(v)

    rows = [[fmt(v) for v in rec] for rec in head.itertuples(index=False)]
    return f"""
    <section id="coordinates">
      <h3>Coordinates</h3>
      <p class="lede">First {len(head)} of {len(df):,} rows.
      Full table: <a href="../coordinates.csv">coordinates.csv</a>.</p>
      {_table(headers, rows)}
    </section>"""


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def write_html_report(output_dir, mode="fit", fetcher=None):
    """
    Assemble ``<output_dir>/report/report.html``.

    Parameters
    ----------
    output_dir : str
        The run directory (its ``report/`` subfolder already holds the figures).
    mode : {'fit', 'transform'}
        Selects which sections are rendered.
    fetcher : ResultsFetcher, optional
        Reused if given, so the reference coordinates are not re-read.

    Returns
    -------
    str
        Path to the written HTML file.
    """
    from .fetcher import ResultsFetcher

    output_dir = os.path.abspath(output_dir)
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    fetcher = fetcher or ResultsFetcher(output_dir)
    cells_map = load_figure_cells(report_dir)
    manifest = fetcher.manifest()

    if mode == "fit":
        title = "Chemical space reference report"
        subtitle = "Fitted reference space and surrogate validation"
        nav = [
            ("configuration", "Configuration"),
            ("chemical-space", "Chemical space"),
            ("surrogate-quality", "Surrogate quality"),
            ("performance", "Performance"),
        ]
        body = "".join(
            [
                _config_section(manifest, fetcher),
                _landscape_section(report_dir, cells_map, fetcher),
                _surrogate_section(report_dir, cells_map, fetcher),
                _performance_section(report_dir, cells_map, fetcher),
            ]
        )
    else:
        title = "Chemical space projection report"
        subtitle = "New molecules projected onto a pretrained reference space"
        nav = [
            ("input", "Input"),
            ("overlays", "Where they landed"),
            ("coordinates", "Coordinates"),
        ]
        body = "".join(
            [
                _transform_input_section(manifest, fetcher),
                _overlay_section(report_dir, cells_map),
                _coordinates_section(fetcher),
            ]
        )

    nav_html = "".join(f"<li><a href='#{a}'>{_esc(t)}</a></li>" for a, t in nav)
    # A transform report counts the molecules that were *projected*; falling back to the
    # reference size here would report the reference space's molecule count instead.
    if mode == "fit":
        n_mols = manifest.get("n_valid") or fetcher.n_reference_molecules()
        meta_bits = [f"{n_mols:,} molecules"] if n_mols else []
    else:
        df = fetcher.coordinates()
        n_mols = manifest.get("n_input") or (len(df) if df is not None else None)
        n_ref = fetcher.n_reference_molecules()
        meta_bits = [f"{n_mols:,} molecules projected"] if n_mols else []
        if n_ref:
            meta_bits.append(f"reference space of {n_ref:,}")
    if manifest.get("versions", {}).get("lazychemvis"):
        meta_bits.append(f"LazyChemVis {manifest['versions']['lazychemvis']}")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LazyChemVis — {_esc(subtitle)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="layout">
  <nav>
    <h1>LazyChemVis</h1>
    <div class="sub">{_esc(subtitle)}</div>
    <ol>{nav_html}</ol>
  </nav>
  <main>
    <header class="run">
      <h2>{_esc(title)}</h2>
      <div class="meta">{_esc(" · ".join(meta_bits))}</div>
    </header>
    {body}
    <footer>
      Produced with <a href="https://github.com/ersilia-os/lazy-chemvis">LazyChemVis</a>
      by the <a href="https://github.com/ersilia-os/ersilia">Ersilia Open Source
      Initiative</a>. Figures are also available as PDFs in <code>report/pdf/</code>.
    </footer>
  </main>
</div>
<script>{_JS}</script>
</body>
</html>
"""

    path = os.path.join(report_dir, REPORT_FILENAME)
    with open(path, "w") as f:
        f.write(doc)
    logger.debug(f"Wrote HTML report: {path}")
    return path
