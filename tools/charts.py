"""Inline-SVG charts for the posts. Values are the paper's test-set numbers (reproduced by
github.com/DZhang3210/address-matching-by-pair-type, results/paper_tables.ipynb).

Each chart: CSS-variable colours (light/dark in style.css), visible text labels on every row
(two series colours sit below 3:1 contrast, so colour is never the only identity), a <title>
tooltip on every mark, and a table view underneath (<details>).
"""
from html import escape

W = 680                         # viewBox width; the SVG scales to the column
MODEL_SLOT = {"Random forest": 1, "Cross-encoder": 2, "Bi-encoder": 3, "ESIM": 4}


def _fmt(v):
    lo, hi = v
    return f"{lo:.3f}" if lo == hi else f"{lo:.3f}–{hi:.3f}"


def _x(v, lo, hi, x0, x1):
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


def _axis(lo, hi, ticks, x0, x1, y_top, y_bot, chance=None):
    out = []
    for t in ticks:
        x = _x(t, lo, hi, x0, x1)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{y_bot + 16}" text-anchor="middle">{t:g}</text>')
    if chance is not None:
        x = _x(chance, lo, hi, x0, x1)
        out.append(f'<line class="chance" x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}"/>')
        out.append(f'<text class="note" x="{x + 4:.1f}" y="{y_top + 10}">chance</text>')
    return out


def _range_mark(v, lo, hi, x0, x1, y, slot, tip):
    a, b = _x(v[0], lo, hi, x0, x1), _x(v[1], lo, hi, x0, x1)
    out = [f'<g class="mark"><title>{escape(tip)}</title>']
    if b - a > 1:
        out.append(f'<line class="range s{slot}" x1="{a:.1f}" y1="{y}" x2="{b:.1f}" y2="{y}"/>')
    out += [f'<circle class="dot s{slot}" cx="{a:.1f}" cy="{y}" r="5"/>',
            f'<circle class="dot s{slot}" cx="{b:.1f}" cy="{y}" r="5"/>',
            f'<rect class="hit" x="{a - 8:.1f}" y="{y - 10}" width="{b - a + 16:.1f}" height="20"/>', "</g>"]
    return out


def _legend(items):
    """items: [(label, slot)] -> HTML legend row."""
    return ('<div class="legend">' + "".join(
        f'<span><i class="sw s{slot}"></i>{escape(label)}</span>' for label, slot in items) + "</div>")


def _table(head, rows):
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details class="table-view"><summary>Show as a table</summary><table><tr>{h}</tr>{b}</table></details>'


def _figure(title, sub, svg, legend, table):
    return (f'<figure class="chart"><figcaption><strong>{escape(title)}</strong>'
            f'<span>{escape(sub)}</span></figcaption>{legend}<div class="svg-scroll">{svg}</div>{table}</figure>')


# ---------------------------------------------------------------- AUC by house-number group (Table 13)
GROUPS = [   # (group label, n, {model: (min, max)})
    ("Same house number on both sides", 638,
     {"Random forest": (0.923, 0.923), "Cross-encoder": (0.922, 0.927), "Bi-encoder": (0.925, 0.934), "ESIM": (0.880, 0.906)}),
    ("House number missing on one side", 164,
     {"Random forest": (0.845, 0.845), "Cross-encoder": (0.689, 0.729), "Bi-encoder": (0.448, 0.497), "ESIM": (0.547, 0.667)}),
    ("Two different house numbers", 41,
     {"Random forest": (0.798, 0.798), "Cross-encoder": (0.928, 0.940), "Bi-encoder": (0.949, 0.963), "ESIM": (0.808, 0.853)}),
]


def auc_groups(groups=GROUPS):
    lo, hi, x0, x1 = 0.4, 1.0, 190, W - 20
    ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    row, title_h, gap, top = 22, 30, 18, 4
    height = top + len(groups) * (title_h + 4 * row + gap) + 34
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="AUC of true matches against look-alike streets, by house-number group">']
    y = top
    for label, n, vals in groups:
        out.append(f'<text class="group" x="0" y="{y + 16}">{escape(label)} <tspan class="muted">({n} matches)</tspan></text>')
        y += title_h
        band_top, band_bot = y - 10, y + 3 * row + 10
        for t in ticks:   # gridlines only inside this group's rows, so they never cross the titles
            x = _x(t, lo, hi, x0, x1)
            out.append(f'<line class="{"chance" if t == 0.5 else "grid"}" x1="{x:.1f}" y1="{band_top}" x2="{x:.1f}" y2="{band_bot}"/>')
        for model, v in vals.items():
            out.append(f'<text class="label" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{escape(model)}</text>')
            seeds = "1 run" if v[0] == v[1] else "3 seeds, min–max"
            out += _range_mark(v, lo, hi, x0, x1, y, MODEL_SLOT[model], f"{model}, {label.lower()}: AUC {_fmt(v)} ({seeds})")
            y += row
        y += gap
    for t in ticks:
        x = _x(t, lo, hi, x0, x1)
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 20}" text-anchor="middle">{"0.5 (chance)" if t == 0.5 else f"{t:g}"}</text>')
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">AUC: true matches vs look-alike streets (0.5 = coin flip)</text>')
    out.append("</svg>")
    legend = _legend([(m, s) for m, s in MODEL_SLOT.items()])
    table = _table(["Group", "n", *MODEL_SLOT], [[g, n, *(_fmt(v[m]) for m in MODEL_SLOT)] for g, n, v in groups])
    return _figure("Where the models differ: the house number",
                   "How well each model ranks true non-identical matches above look-alike-street fakes, split by what happened to the house number. Test set; bars span three seeds.",
                   "\n".join(out), legend, table)


# ---------------------------------------------------------------- overall AUC (Table 14)
OVERALL = [("Random forest / XGBoost", (0.902, 0.902)), ("DistilBERT cross-encoder", (0.877, 0.888)),
           ("Siamese BiLSTM", (0.854, 0.854)), ("DistilBERT bi-encoder", (0.836, 0.845)),
           ("ESIM", (0.815, 0.842))]


def auc_overall():
    lo, hi, x0, x1 = 0.80, 0.92, 200, W - 20
    row, top = 30, 12
    height = top + len(OVERALL) * row + 40
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Overall AUC per model">']
    out += _axis(lo, hi, [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92], x0, x1, top, height - 36)
    y = top + 16
    for name, v in OVERALL:
        out.append(f'<text class="label" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{escape(name)}</text>')
        out += _range_mark(v, lo, hi, x0, x1, y, 1, f"{name}: AUC {_fmt(v)}")
        y += row
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">AUC: non-identical matches vs look-alike streets (axis starts at 0.80)</text>')
    out.append("</svg>")
    table = _table(["Model", "AUC"], [[n, _fmt(v)] for n, v in OVERALL])
    return _figure("One number per model",
                   "Chance that a random true (non-identical) match scores above a random look-alike-street fake. Test set; neural ranges span three seeds.",
                   "\n".join(out), "", table)


# ---------------------------------------------------------------- sentinel vs zero-fill (Table 4, random forest)
def missing_tradeoff():
    cats = [("True matches found", 164, 27.4, 11.6), ("Same-street non-matches rejected", 344, 82.0, 95.9)]
    x0, x1, top, bar, inner, group_gap = 240, W - 60, 10, 18, 4, 26
    height = top + len(cats) * (2 * bar + inner + group_gap) + 44
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Random forest on pairs with one house number missing: sentinel vs zero-fill">']
    out += _axis(0, 100, [0, 25, 50, 75, 100], x0, x1, top, height - 40)
    y = top + 6
    for label, n, sent, zero in cats:
        out.append(f'<text class="label" x="{x0 - 10}" y="{y + bar + 2}" text-anchor="end">{escape(label)}</text>')
        out.append(f'<text class="label muted" x="{x0 - 10}" y="{y + bar + 16}" text-anchor="end">n = {n}</text>')
        for k, (setting, val) in enumerate([("Sentinel (−1)", sent), ("Zero-fill (0)", zero)]):
            yy = y + k * (bar + inner)
            w = _x(val, 0, 100, x0, x1) - x0
            r = min(4, w / 2)
            path = (f"M{x0},{yy} H{x0 + w - r} Q{x0 + w},{yy} {x0 + w},{yy + r} V{yy + bar - r} "
                    f"Q{x0 + w},{yy + bar} {x0 + w - r},{yy + bar} H{x0} Z")
            out.append(f'<g class="mark"><title>{escape(setting)}: {val:.1f}% ({label.lower()})</title>'
                       f'<path class="bar s{k + 1}" d="{path}"/>'
                       f'<text class="value" x="{x0 + w + 6:.1f}" y="{yy + bar - 5}">{val:.1f}%</text></g>')
        y += 2 * bar + inner + group_gap
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">% correct</text>')
    out.append("</svg>")
    legend = _legend([("Sentinel: missing number marked “not comparable” (−1)", 1), ("Zero-fill: missing number read as 0", 2)])
    table = _table(["Pairs with one number missing", "n", "Sentinel", "Zero-fill"], [[c, n, f"{s}%", f"{z}%"] for c, n, s, z in cats])
    return _figure("Same text, opposite labels",
                   "Random forest on test pairs where exactly one address has a house number. Changing how the missing number is encoded moves errors from one row to the other.",
                   "\n".join(out), legend, table)


CHARTS = {"auc_groups": auc_groups, "auc_overall": auc_overall, "missing_tradeoff": missing_tradeoff}
