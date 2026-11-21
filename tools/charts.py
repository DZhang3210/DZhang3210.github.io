"""Inline-SVG charts for the posts. Values are the paper's test-set numbers (reproduced by
github.com/DZhang3210/address-matching-by-pair-type, results/paper_tables.ipynb). Updated to the paper v2 (corrected data, all models retrained).

Each chart: CSS-variable colours (light/dark in style.css), visible text labels on every row
(two series colours sit below 3:1 contrast, so colour is never the only identity), a <title>
tooltip on every mark, and a table view underneath (<details>).
"""
from html import escape

W = 680                         # viewBox width; the SVG scales to the column
MODEL_SLOT = {"XGBoost": 1, "Siamese BiLSTM": 5, "ESIM": 4, "Bi-encoder": 3, "Cross-encoder": 2}


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
     {"XGBoost": (0.919, 0.919), "Siamese BiLSTM": (0.914, 0.921), "ESIM": (0.898, 0.917),
      "Bi-encoder": (0.927, 0.934), "Cross-encoder": (0.908, 0.928)}),
    ("House number missing on one side", 164,
     {"XGBoost": (0.962, 0.962), "Siamese BiLSTM": (0.730, 0.810), "ESIM": (0.699, 0.790),
      "Bi-encoder": (0.692, 0.724), "Cross-encoder": (0.775, 0.804)}),
    ("Two different house numbers", 41,
     {"XGBoost": (0.774, 0.774), "Siamese BiLSTM": (0.904, 0.945), "ESIM": (0.835, 0.871),
      "Bi-encoder": (0.964, 0.973), "Cross-encoder": (0.916, 0.937)}),
]


def auc_groups(groups=GROUPS):
    lo, hi, x0, x1 = 0.5, 1.0, 190, W - 20
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k = len(MODEL_SLOT)
    row, title_h, gap, top = 22, 30, 18, 4
    height = top + len(groups) * (title_h + k * row + gap) + 34
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="AUC of true matches against look-alike streets, by house-number group">']
    y = top
    for label, n, vals in groups:
        out.append(f'<text class="group" x="0" y="{y + 16}">{escape(label)} <tspan class="muted">({n} matches)</tspan></text>')
        y += title_h
        band_top, band_bot = y - 10, y + (k - 1) * row + 10
        for t in ticks:   # gridlines only inside this group's rows, so they never cross the titles
            x = _x(t, lo, hi, x0, x1)
            out.append(f'<line class="{"chance" if t == 0.5 else "grid"}" x1="{x:.1f}" y1="{band_top}" x2="{x:.1f}" y2="{band_bot}"/>')
        for model in MODEL_SLOT:
            v = vals[model]
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
    return _figure("AUC by house-number group",
                   "How well each model ranks true non-identical matches above look-alike streets, split by what happened to the house number. Test set; neural ranges span three seeds.",
                   "\n".join(out), legend, table)


# ---------------------------------------------------------------- overall AUC (Table 15)
OVERALL = [("XGBoost", (0.920, 0.920)), ("DistilBERT cross-encoder", (0.888, 0.904)),
           ("DistilBERT bi-encoder", (0.883, 0.893)), ("Siamese BiLSTM", (0.878, 0.893)),
           ("ESIM (fastText)", (0.861, 0.890))]


def auc_overall():
    lo, hi, x0, x1 = 0.84, 0.94, 200, W - 20
    row, top = 30, 12
    height = top + len(OVERALL) * row + 40
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Overall AUC per model">']
    out += _axis(lo, hi, [0.84, 0.86, 0.88, 0.90, 0.92, 0.94], x0, x1, top, height - 36)
    y = top + 16
    for name, v in OVERALL:
        out.append(f'<text class="label" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{escape(name)}</text>')
        out += _range_mark(v, lo, hi, x0, x1, y, 1, f"{name}: AUC {_fmt(v)}")
        y += row
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">AUC: non-identical matches vs look-alike streets (axis starts at 0.84)</text>')
    out.append("</svg>")
    table = _table(["Model", "AUC"], [[n, _fmt(v)] for n, v in OVERALL])
    return _figure("One number per model",
                   "Chance that a random true (non-identical) match scores above a random look-alike street. Test set; neural ranges span three seeds.",
                   "\n".join(out), "", table)


# ---------------------------------------------------------------- missing-number matches found (Tables 3, 4, 6, 15, B1)
FOUND = [   # (model, all non-identical matches found, matches with a number on one side only found), % at 0.5
    ("Random forest", (84.0, 84.0), (96.3, 96.3)),
    ("XGBoost", (83.6, 83.6), (94.5, 94.5)),
    ("Siamese BiLSTM", (87.0, 89.3), (69.5, 75.6)),
    ("ESIM (fastText)", (87.3, 90.3), (71.3, 75.6)),
    ("DistilBERT bi-encoder", (89.0, 90.9), (69.5, 72.6)),
    ("DistilBERT cross-encoder", (80.3, 83.0), (63.4, 66.5)),
]


def _pct(v):
    lo, hi = v
    return f"{lo:.1f}%" if lo == hi else f"{lo:.1f}–{hi:.1f}%"


def missing_found():
    lo, hi, x0, x1 = 50, 100, 200, W - 20
    row, top = 30, 12
    height = top + len(FOUND) * row + 40
    series = [("All non-identical matches (843)", 1), ("House number on one side only (164)", 2)]
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Share of matches found at 0.5, all non-identical vs house number missing on one side">']
    out += _axis(lo, hi, [50, 60, 70, 80, 90, 100], x0, x1, top, height - 36)
    y = top + 16
    for name, allv, miss in FOUND:
        out.append(f'<text class="label" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{escape(name)}</text>')
        for (lab, slot), v in zip(series, (allv, miss)):
            out += _range_mark(v, lo, hi, x0, x1, y, slot, f"{name}, {lab.lower()}: {_pct(v)} found")
        y += row
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">% of matches found at the 0.5 threshold (axis starts at 50)</text>')
    out.append("</svg>")
    table = _table(["Model", "All non-identical", "Number on one side only"],
                   [[n, _pct(a), _pct(m)] for n, a, m in FOUND])
    return _figure("Matches with a missing house number",
                   "Share of true matches found at 0.5: all non-identical matches, and the 164 where only one side has a house number. Test set; neural ranges span three seeds.",
                   "\n".join(out), _legend(series), table)


CHARTS = {"auc_groups": auc_groups, "auc_overall": auc_overall, "missing_found": missing_found}
