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


# ---------------------------------------------------------------- two series per model, grouped (tree / neural)
def _paired(rows, series, lo, hi, ticks, fmt, axis_label, title, sub, table_head):
    """rows: [(group, model, v1, v2)] with v = (min, max); one row per model, two marks per row."""
    x0, x1 = 200, W - 20
    row, top, head, gap = 30, 4, 34, 14
    groups = []
    for g, *r in rows:
        if not groups or groups[-1][0] != g:
            groups.append((g, []))
        groups[-1][1].append(r)
    height = top + sum(head + len(r) * row + gap for _, r in groups) + 30
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">']
    y = top
    for g, models in groups:
        # header: small caps label over a hairline, like a table section
        out.append(f'<text class="note" x="0" y="{y + 18}" letter-spacing="0.08em">{escape(g.upper())}</text>')
        out.append(f'<line class="grid" x1="0" y1="{y + 26}" x2="{x1}" y2="{y + 26}"/>')
        y += head
        for t in ticks:   # gridlines only inside the group's rows
            x = _x(t, lo, hi, x0, x1)
            out.append(f'<line class="grid" x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + len(models) * row}"/>')
        y += row // 2
        for name, *vals in models:
            out.append(f'<text class="label" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{escape(name)}</text>')
            for (lab, slot), v in zip(series, vals):
                out += _range_mark(v, lo, hi, x0, x1, y, slot, f"{name}, {lab}: {fmt(v)}")
            y += row
        y += gap - row // 2
    for t in ticks:
        x = _x(t, lo, hi, x0, x1)
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 18}" text-anchor="middle">{t:g}</text>')
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 2}" text-anchor="middle">{escape(axis_label)}</text>')
    out.append("</svg>")
    table = _table(table_head, [[name, fmt(a), fmt(b)] for _, name, a, b in rows])
    return _figure(title, sub, "\n".join(out), _legend(series), table)


def _pct(v):
    lo, hi = v
    return f"{lo:.1f}%" if lo == hi else f"{lo:.1f}–{hi:.1f}%"


# ---------------------------------------------------------------- missing-number matches found (Tables 3, 4, 6, 15, B1)
FOUND = [   # (group, model, all non-identical matches found, matches with a number on one side only found), % at 0.5
    ("Tree models", "Random forest", (84.0, 84.0), (96.3, 96.3)),
    ("Tree models", "XGBoost", (83.6, 83.6), (94.5, 94.5)),
    ("Neural models", "Siamese BiLSTM", (87.0, 89.3), (69.5, 75.6)),
    ("Neural models", "ESIM (fastText)", (87.3, 90.3), (71.3, 75.6)),
    ("Neural models", "DistilBERT bi-encoder", (89.0, 90.9), (69.5, 72.6)),
    ("Neural models", "DistilBERT cross-encoder", (80.3, 83.0), (63.4, 66.5)),
]


def missing_found():
    return _paired(FOUND, [("All non-identical matches (843)", 1), ("House number on one side only (164)", 2)],
                   50, 100, [50, 60, 70, 80, 90, 100], _pct,
                   "% of matches found at the 0.5 threshold (axis starts at 50)",
                   "Matches with a missing house number",
                   "Share of true matches found at 0.5: all non-identical matches, and the 164 where only one side has a house number. Test set; neural ranges span three seeds.",
                   ["Model", "All non-identical", "Number on one side only"])


# ---------------------------------------------------------------- AUC with and without missing-number matches (Table 13)
AUC_SPLIT = [   # (group, model, AUC all non-identical vs look-alike, AUC without the 164 missing-number matches)
    ("Tree models", "XGBoost", (0.920, 0.920), (0.910, 0.910)),
    ("Neural models", "Siamese BiLSTM", (0.878, 0.893), (0.913, 0.923)),
    ("Neural models", "ESIM (fastText)", (0.861, 0.890), (0.894, 0.914)),
    ("Neural models", "DistilBERT bi-encoder", (0.883, 0.893), (0.930, 0.937)),
    ("Neural models", "DistilBERT cross-encoder", (0.888, 0.904), (0.908, 0.928)),
]


def auc_without_missing():
    return _paired(AUC_SPLIT, [("All non-identical matches (843)", 1), ("Without missing-number matches (679)", 3)],
                   0.84, 0.96, [0.84, 0.88, 0.92, 0.96], _fmt,
                   "AUC: non-identical matches vs look-alike streets (axis starts at 0.84)",
                   "Look-alike streets, with and without missing-number matches",
                   "How well each model ranks true non-identical matches above look-alike streets: all of them, and without the 164 where one side has no house number. Test set; neural ranges span three seeds.",
                   ["Model", "All non-identical", "Without missing-number matches"])


# ---------------------------------------------------------------- Youden's J against the threshold (paper Figure 3)
# J = share of non-identical matches accepted - share of look-alike streets accepted, validation,
# computed from the saved DistilBERT probabilities (address-matching-by-pair-type, predictions/distilbert_*/probs_val_seed_*.npy).
J_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.994, 0.996, 0.998, 0.999]
J_CURVES = {
    "Bi-encoder": [
         [0.362, 0.386, 0.405, 0.417, 0.431, 0.436, 0.449, 0.45, 0.457, 0.462, 0.473, 0.476, 0.479, 0.479, 0.483, 0.485, 0.489, 0.492, 0.492, 0.493, 0.494, 0.498, 0.5, 0.501, 0.502, 0.504, 0.506, 0.511, 0.511, 0.508, 0.511, 0.513, 0.515, 0.514, 0.518, 0.519, 0.521, 0.523, 0.528, 0.529, 0.532, 0.533, 0.536, 0.54, 0.541, 0.541, 0.542, 0.543, 0.546, 0.547, 0.548, 0.549, 0.552, 0.554, 0.556, 0.557, 0.557, 0.557, 0.559, 0.559, 0.562, 0.562, 0.564, 0.565, 0.567, 0.571, 0.572, 0.573, 0.575, 0.577, 0.578, 0.581, 0.583, 0.583, 0.584, 0.585, 0.585, 0.587, 0.588, 0.589, 0.589, 0.591, 0.593, 0.591, 0.593, 0.593, 0.596, 0.6, 0.602, 0.608, 0.614, 0.613, 0.62, 0.622, 0.626, 0.627, 0.638, 0.645, 0.651, 0.649, 0.656, 0.661, 0.663, 0.662, 0.662, 0.66, 0.659, 0.672] ,
         [0.215, 0.259, 0.287, 0.312, 0.325, 0.332, 0.343, 0.348, 0.349, 0.362, 0.368, 0.375, 0.377, 0.386, 0.392, 0.398, 0.4, 0.405, 0.408, 0.41, 0.412, 0.417, 0.418, 0.421, 0.422, 0.427, 0.429, 0.433, 0.436, 0.441, 0.444, 0.448, 0.449, 0.45, 0.451, 0.457, 0.46, 0.461, 0.462, 0.465, 0.467, 0.467, 0.471, 0.473, 0.474, 0.473, 0.475, 0.478, 0.478, 0.48, 0.48, 0.483, 0.486, 0.486, 0.491, 0.493, 0.495, 0.496, 0.497, 0.5, 0.503, 0.507, 0.507, 0.508, 0.511, 0.515, 0.516, 0.516, 0.517, 0.519, 0.524, 0.526, 0.527, 0.532, 0.534, 0.536, 0.536, 0.54, 0.542, 0.544, 0.545, 0.546, 0.551, 0.554, 0.561, 0.562, 0.559, 0.567, 0.567, 0.574, 0.578, 0.58, 0.582, 0.592, 0.598, 0.602, 0.617, 0.616, 0.619, 0.622, 0.626, 0.634, 0.645, 0.653, 0.662, 0.663, 0.644, 0.624] ,
         [0.372, 0.396, 0.403, 0.415, 0.429, 0.44, 0.448, 0.45, 0.452, 0.455, 0.455, 0.457, 0.459, 0.464, 0.466, 0.468, 0.472, 0.474, 0.478, 0.482, 0.489, 0.491, 0.492, 0.494, 0.496, 0.498, 0.498, 0.498, 0.498, 0.5, 0.505, 0.508, 0.507, 0.51, 0.513, 0.513, 0.519, 0.519, 0.521, 0.521, 0.521, 0.523, 0.524, 0.527, 0.53, 0.53, 0.532, 0.533, 0.533, 0.535, 0.536, 0.541, 0.543, 0.545, 0.546, 0.551, 0.553, 0.554, 0.557, 0.563, 0.566, 0.564, 0.568, 0.569, 0.571, 0.572, 0.573, 0.573, 0.575, 0.575, 0.577, 0.581, 0.582, 0.585, 0.585, 0.585, 0.583, 0.586, 0.586, 0.589, 0.592, 0.591, 0.593, 0.594, 0.601, 0.601, 0.605, 0.609, 0.61, 0.615, 0.618, 0.619, 0.622, 0.624, 0.63, 0.631, 0.632, 0.636, 0.64, 0.65, 0.657, 0.661, 0.671, 0.678, 0.677, 0.672, 0.671, 0.661] ,
    ],
    "Cross-encoder": [
         [0.51, 0.612, 0.641, 0.662, 0.678, 0.687, 0.693, 0.699, 0.7, 0.704, 0.707, 0.708, 0.709, 0.714, 0.716, 0.72, 0.721, 0.721, 0.723, 0.724, 0.727, 0.727, 0.727, 0.729, 0.73, 0.73, 0.732, 0.733, 0.736, 0.739, 0.741, 0.742, 0.742, 0.743, 0.744, 0.745, 0.745, 0.745, 0.745, 0.745, 0.745, 0.744, 0.744, 0.744, 0.744, 0.745, 0.744, 0.744, 0.743, 0.745, 0.747, 0.748, 0.748, 0.747, 0.748, 0.748, 0.748, 0.747, 0.747, 0.748, 0.75, 0.751, 0.751, 0.75, 0.751, 0.754, 0.753, 0.753, 0.754, 0.755, 0.754, 0.755, 0.755, 0.755, 0.754, 0.754, 0.754, 0.755, 0.756, 0.756, 0.758, 0.759, 0.759, 0.756, 0.756, 0.754, 0.755, 0.756, 0.755, 0.758, 0.759, 0.759, 0.761, 0.759, 0.762, 0.76, 0.761, 0.761, 0.759, 0.76, 0.756, 0.757, 0.753, 0.756, 0.747, 0.738, 0.71, 0.648] ,
         [0.455, 0.492, 0.501, 0.61, 0.637, 0.657, 0.662, 0.667, 0.668, 0.673, 0.677, 0.678, 0.68, 0.681, 0.681, 0.681, 0.682, 0.683, 0.682, 0.687, 0.685, 0.687, 0.688, 0.688, 0.687, 0.689, 0.695, 0.695, 0.699, 0.699, 0.7, 0.703, 0.703, 0.704, 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.706, 0.706, 0.706, 0.705, 0.703, 0.703, 0.704, 0.704, 0.703, 0.703, 0.703, 0.703, 0.704, 0.704, 0.703, 0.704, 0.705, 0.705, 0.706, 0.705, 0.706, 0.707, 0.707, 0.708, 0.708, 0.71, 0.709, 0.711, 0.71, 0.71, 0.711, 0.71, 0.711, 0.712, 0.713, 0.715, 0.72, 0.723, 0.724, 0.724, 0.725, 0.723, 0.726, 0.726, 0.725, 0.725, 0.722, 0.719, 0.715, 0.717, 0.717, 0.715, 0.707, 0.706, 0.703, 0.705, 0.698, 0.684, 0.652, 0.517, 0.34, 0.0, 0.0, 0.0] ,
         [0.478, 0.53, 0.579, 0.608, 0.634, 0.65, 0.658, 0.675, 0.699, 0.705, 0.708, 0.71, 0.714, 0.719, 0.722, 0.725, 0.726, 0.727, 0.728, 0.729, 0.729, 0.73, 0.731, 0.734, 0.732, 0.736, 0.735, 0.738, 0.743, 0.743, 0.744, 0.748, 0.751, 0.752, 0.753, 0.754, 0.754, 0.754, 0.755, 0.755, 0.755, 0.758, 0.757, 0.759, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.761, 0.761, 0.761, 0.762, 0.761, 0.761, 0.762, 0.762, 0.76, 0.762, 0.763, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.759, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.761, 0.762, 0.763, 0.76, 0.76, 0.76, 0.759, 0.76, 0.761, 0.762, 0.76, 0.763, 0.766, 0.766, 0.765, 0.763, 0.766, 0.766, 0.763, 0.759, 0.76, 0.754, 0.756, 0.756, 0.743, 0.731, 0.706, 0.228, 0.0, 0.0] ,
    ],
}


def youden_curves():
    x0, x1, top, bot = 50, W - 20, 10, 250
    xlo, xhi, ylo, yhi = 0.0, 1.0, 0.0, 0.8
    height = bot + 44
    def px(t): return _x(t, xlo, xhi, x0, x1)
    def py(j): return bot - (j - ylo) / (yhi - ylo) * (bot - top)
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Youden\'s J against the threshold, bi-encoder vs cross-encoder">']
    for j in [0, 0.2, 0.4, 0.6, 0.8]:
        out.append(f'<line class="grid" x1="{x0}" y1="{py(j):.1f}" x2="{x1}" y2="{py(j):.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{py(j) + 4:.1f}" text-anchor="end">{j:g}</text>')
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        out.append(f'<text class="tick" x="{px(t):.1f}" y="{bot + 18}" text-anchor="middle">{t:g}</text>')
    for (name, curves), slot in zip(J_CURVES.items(), (1, 2)):
        for k, c in enumerate(curves):
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(t):.1f},{py(j):.1f}" for i, (t, j) in enumerate(zip(J_THRESHOLDS, c)))
            best = max(c)
            out.append(f'<g class="mark"><title>{name}, seed {k}: best J {best:.2f} at threshold {J_THRESHOLDS[c.index(best)]:g}</title>'
                       f'<path class="line s{slot}" d="{d}"/></g>')
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{height - 4}" text-anchor="middle">Threshold</text>')
    out.append("</svg>")
    rows = []
    for name, curves in J_CURVES.items():
        for k, c in enumerate(curves):
            best = max(c)
            rows.append([name, k, f"{best:.2f}", f"{J_THRESHOLDS[c.index(best)]:g}", f"{c[J_THRESHOLDS.index(0.5)]:.2f}"])
    table = _table(["Model", "Seed", "Best J", "At threshold", "J at 0.5"], rows)
    return _figure("Same ranking, different score spread",
                   "Youden's J (share of non-identical matches accepted minus share of look-alike streets accepted) at every threshold. Validation set; one line per seed.",
                   "\n".join(out), _legend([("Bi-encoder", 1), ("Cross-encoder", 2)]), table)


CHARTS = {"missing_found": missing_found, "auc_without_missing": auc_without_missing, "youden_curves": youden_curves}
