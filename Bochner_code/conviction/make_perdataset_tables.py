#!/usr/bin/env python3
"""
Per-dataset breakdown of the conviction forward sweep, for the supplementary.

conviction_summary_tables.py answers "how well does the effect reproduce on a new
benchmark?" and so averages the datasets together, reporting a between-dataset SE.
This script answers the complementary question — "where does that aggregate come
from, and how stable is it inside one dataset?" — by keeping the datasets apart and
reporting the SE ACROSS FOLDS of a single dataset.

That is the ONLY statistical difference, and it is deliberate:

    conviction_summary_tables._aggregate   mean over folds -> mean over datasets
                                           spread = SE across datasets
    this script                            mean over folds of ONE dataset
                                           spread = SE across folds

So `_aggregate` is deliberately NOT reused here — it would collapse the dataset axis
that is the whole point of these tables. Everything upstream of the spread IS reused
(_fold_drops, _rel_drop, _FLOOR, STRATEGY_LABEL), so the per-fold drops entering these
tables are byte-identical to the ones entering the aggregated tables and the two
families cannot silently drift apart.

The quantity in every cell is the same relative drop used everywhere else in the
section, normalised by the headroom above chance:

    drop%  =  100 * (baseline - metric) / (baseline - floor),   floor = 0.5 for AUC, 0 else

Two tables are produced, each covering the four selections that matter for the
high/low contrast, with the degree baseline listed first so the conviction row reads
against it:

    perdataset_masking-zero_high.tex   High degree,  High conviction
    perdataset_masking-zero_low.tex    Low degree,   Low conviction

Cornell and Wisconsin are excluded by default, as everywhere else in the conviction
section: their training pools are too small for a percentage budget to be meaningful
(a 1% budget is a single node), so eight datasets are reported.

The LaTeX here is emitted directly rather than through conviction_summary_tables'
_latex_table, which builds a different shape: [H] float, caption after the tabular,
one leading `l` column. These tables are [htbp], caption first, and `llccccc` with a
two-level row index (dataset, selection).

Run from the repo root:
    python Bochner_code/conviction/make_perdataset_tables.py
"""

import os
import sys
import argparse

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUTS = os.path.join(ROOT, 'Bochner_code', 'conviction', 'outputs')

# Reused rather than reimplemented so the per-fold drops feeding these tables are
# identical to the ones feeding the aggregated tables.
from conviction_summary_tables import _read, _fold_drops, STRATEGY_LABEL

# ── constants ─────────────────────────────────────────────────────────────────

BUDGETS = [0.01, 0.05, 0.10, 0.20, 0.30]

# Degree first, conviction second: the degree row is the baseline the conviction row
# is meant to be read against, and the paper's discussion compares them in that order.
SELECTIONS = {
    'high': ['deg_high', 'high'],
    'low':  ['deg_low',  'low'],
}

# Display names. The datasets are otherwise identified by their directory slug.
DATASET_LABEL = {
    'amazon_ratings': 'Amazon Ratings',
    'citeseer':       'Citeseer',
    'cora':           'Cora',
    'minesweeper':    'Minesweeper',
    'pubmed':         'Pubmed',
    'questions':      'Questions',
    'roman_empire':   'Roman Empire',
    'tolokers':       'Tolokers',
    'cornell':        'Cornell',
    'wisconsin':      'Wisconsin',
}

_CAPTION = (
    r"\textbf{{Masking (zero), per dataset, {side} selections.}} Relative drop (\%) "
    r"in the headline metric, accuracy or ROC-AUC on Minesweeper, Tolokers and "
    r"Questions, expressed as a percentage of the baseline's headroom above chance, "
    r"as a function of the fraction of the training pool perturbed. Conviction score "
    r"\texttt{{{score}}}. Unlike the aggregated tables, each cell is the mean over the "
    r"ten folds of that dataset and the quoted spread is the standard error "
    r"\emph{{across folds}}, so it measures how stable the effect is within a dataset "
    r"and not how well it reproduces across datasets."
)


# ── collection ────────────────────────────────────────────────────────────────

def collect(score, model, normalised, mask_mode, exclude=()):
    """{dataset: {(strategy, budget): array of per-fold drops}}.

    Reads the aggregated per-dataset CSVs and lets _fold_drops pick up each one's
    sibling *_folds.csv, exactly as collect_forward does — so a dataset whose folds
    file is missing degrades to a single fold mean rather than vanishing.
    """
    base = os.path.join(OUTPUTS, 'per_dataset_tables', 'forward_experiment',
                        'masking', score, model, f'normalised_{normalised}')
    if not os.path.isdir(base):
        sys.exit(f"no such directory: {base}\n"
                 f"run the masking sweep first (exp/scripts/script_forward/"
                 f"run_conviction_masking.sh)")

    out = {}
    for name in sorted(os.listdir(base)):
        if not name.endswith('.csv') or name.endswith('_folds.csv'):
            continue
        if f'_{mask_mode}_' not in name:
            continue
        path = os.path.join(base, name)
        rows = _read(path)
        if not rows or 'strategy' not in rows[0]:
            continue
        dataset = rows[0]['dataset']
        if dataset in exclude:
            continue
        out[dataset] = _fold_drops(path, rows[0]['metric'], 'strategy', 'pct')
    return out


def _mean_se_over_folds(arr):
    """Mean and standard error ACROSS FOLDS for one (dataset, strategy, budget).

    This is the cell statistic these tables exist to report, and the one place they
    depart from conviction_summary_tables._aggregate, which takes its spread across
    datasets instead. Returns None when a cell has no finite fold.
    """
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    se = float(vals.std(ddof=1)) / np.sqrt(vals.size) if vals.size > 1 else 0.0
    return {'mean': float(vals.mean()), 'se': se, 'n_folds': int(vals.size)}


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def _cell(stat):
    return "--" if stat is None else f"${stat['mean']:.2f} \\pm {stat['se']:.2f}$"


def latex_table(data, side, score, mask_mode, out_dir):
    """One table: datasets as row groups, budgets as columns.

    The dataset name is printed on its group's first row only and the following row
    starts with a bare `&`, which is what makes the two selections read as a pair
    rather than as eight unrelated rows.
    """
    strategies = SELECTIONS[side]
    body = []
    for dataset in sorted(data):
        for i, strat in enumerate(strategies):
            label = DATASET_LABEL.get(dataset, dataset) if i == 0 else ''
            cells = [_cell(_mean_se_over_folds(data[dataset][(strat, b)]))
                     if (strat, b) in data[dataset] else "--"
                     for b in BUDGETS]
            body.append(f"{label} & {STRATEGY_LABEL[strat]} & " + " & ".join(cells) + r" \\")
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    cols = " & ".join(f"{int(b * 100)}\\%" for b in BUDGETS)
    lines = [
        r"%% auto-generated by make_perdataset_tables.py -- do not edit by hand",
        r"\begin{table}[htbp]",
        rf"\caption{{{_CAPTION.format(side=side, score=score)}}}",
        rf"\label{{tab:conv_perds_{side}}}",
        r"\centering",
        r"\small",
        r"%% \tabcolsep reduced from the 6pt default: this table overflows",
        r"%% Springer's 31pc text block at its natural width.",
        r"\footnotesize\setlength{\tabcolsep}{2pt}%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        rf"Dataset & Selection & {cols} \\",
        r"\midrule",
    ] + body + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"perdataset_masking-{mask_mode}_{side}.tex")
    open(path, 'w').write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--score',      default='last', choices=['last', 'avg', 'avg_z'])
    ap.add_argument('--model',      default='GeneralSheaf')
    ap.add_argument('--normalised', default='true', choices=['true', 'false'])
    ap.add_argument('--mask_mode',  default='zero', choices=['zero', 'mean'])
    ap.add_argument('--exclude',    default='cornell,wisconsin',
                    help='datasets left out of the table (training pool too small '
                         'for a percentage budget to mean anything)')
    ap.add_argument('--out_dir',    default=os.path.join(OUTPUTS, 'summary_tables'))
    cli = ap.parse_args()

    exclude = {d.strip() for d in cli.exclude.split(',') if d.strip()}
    data = collect(cli.score, cli.model, cli.normalised, cli.mask_mode, exclude)
    if not data:
        sys.exit("no per-dataset tables matched — check --model/--normalised/--score")

    n_folds = {s['n_folds'] for d in data.values() for v in d.values()
               if (s := _mean_se_over_folds(v)) is not None}
    print(f"{len(data)} datasets: {', '.join(sorted(data))}")
    print(f"folds per cell: {sorted(n_folds)}")

    for side in ('high', 'low'):
        latex_table(data, side, cli.score, cli.mask_mode, cli.out_dir)


if __name__ == '__main__':
    main()
