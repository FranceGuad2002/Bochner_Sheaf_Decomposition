#!/usr/bin/env python3
"""
The paper's three per-dataset Forman-profile tables (tab:perds_rm_norm,
tab:perds_citeseer_unnorm, tab:perds_questions_joint_unnorm): one block per
metric, training checkpoints on the rows, layers on the columns. A cell is the
metric averaged over the d curvature eigenvalues, given as mean +/- SE across
the folds of that dataset (SE with ddof=1).

Two stages, like the global tables:

  --from_eigs   per-fold values from results/forman_eigs/ (the Zenodo archive),
                written to outputs/per_dataset/forman_perdataset_folds.csv
  (default)     the three LaTeX tables, from that committed CSV, into
                outputs/summary_tables/ — seconds, no archive needed

Run from the repo root:
    python Bochner_code/forman_profile/make_forman_perdataset_tables.py --from_eigs
    python Bochner_code/forman_profile/make_forman_perdataset_tables.py
"""

import argparse
import math
import numpy as np
import pandas as pd

from compute_forman_checkpoints import OUTPUTS, discover_combinations, fold_metrics, _make_map_tag
from generate_forman_tables import _ColorRegistry, _cmap_rgb

FOLDS_CSV = OUTPUTS / "per_dataset" / "forman_perdataset_folds.csv"
OUT_DIR   = OUTPUTS / "summary_tables"

CHECKPOINTS = ["epoch-0", "epoch-1", "epoch-5", "epoch-15", "epoch-50", "epoch-200", "last"]
ROW_LABEL   = {"epoch-0": "0", "epoch-1": "1", "epoch-5": "5", "epoch-15": "15",
               "epoch-50": "50", "epoch-200": "200", "last": "final"}

# Correlations on RdBu over [-1, 1]; distances on YlOrRd from 0 to a fixed
# maximum, or to the block maximum where the scale is not bounded (None).
CORR = ("RdBu_r", -1.0, 1.0)
METRICS = {
    "spearman":    (r"Spearman $\rho$",              CORR),
    "pearson":     (r"Pearson $r$",                  CORR),
    "wasserstein": (r"$W_1$",                        ("YlOrRd", 0.0, 0.9)),
    "rmse":        (r"RMSE",                         ("YlOrRd", 0.0, 0.6)),
    "nrmse":       (r"NRMSE",                        ("YlOrRd", 0.0, 1.8)),
    "wa_top":      (r"$W_1$, topological",           ("YlOrRd", 0.0, 0.9)),
    "wa_zero":     (r"$W_1$, zero",                  ("YlOrRd", 0.0, None)),
    "pearson_top": (r"Pearson $r$, topological",     CORR),
    "nrmse_top":   (r"NRMSE, topological",           ("YlOrRd", 0.0, 1.8)),
    "rmse_zero":   (r"RMSE, zero",                   ("YlOrRd", 0.0, None)),
}
# Unnormalized content reuses the names 'pearson' and 'nrmse' for comparisons
# with the topological reference; the tables label them explicitly.
UNNORM_RENAME = {"pearson": "pearson_top", "nrmse": "nrmse_top"}

TABLES = [
    dict(name="perds_rm_norm", dataset="roman_empire", model="GeneralSheaf", normalised="true",
         map_type=None, norm_content=True,
         title="GeneralSheaf, normalized: Roman Empire",
         metrics=["spearman", "pearson", "wasserstein", "rmse", "nrmse"],
         caption="Normalized GeneralSheaf on Roman Empire, $d=5$, five layers, against the "
                 "topological reference."),
    dict(name="perds_citeseer_unnorm", dataset="citeseer", model="GeneralSheaf", normalised="false",
         map_type=None, norm_content=False,
         title="GeneralSheaf, unnormalized: Citeseer",
         metrics=["wa_top", "wa_zero", "pearson_top", "nrmse_top", "rmse_zero"],
         caption="Unnormalized GeneralSheaf on Citeseer, $d=5$, four layers, against the "
                 "topological and the zero reference."),
    dict(name="perds_questions_joint_unnorm", dataset="questions", model="JointSheafParams",
         normalised="false", map_type="identity", norm_content=False,
         title="JointSheafParams, unnormalized: Questions",
         metrics=["wa_top", "wa_zero", "pearson_top", "nrmse_top", "rmse_zero"],
         caption="Unnormalized JointSheafParams on Questions, $d=5$, four layers, started at the "
                 "identity sheaf, against the topological and the zero reference. In the first "
                 "layer the profile is constant, so its correlation is undefined."),
]


# ── stage 1: per-fold values from the raw eigenvalues ─────────────────────────
def harvest():
    rows = []
    for t in TABLES:
        map_tag = _make_map_tag(t["model"], t["map_type"])
        combos  = [c for c in discover_combinations(t["model"], t["normalised"], map_tag)
                   if c["dataset"] == t["dataset"]]
        if len(combos) != 1:
            raise SystemExit(f"{t['name']}: expected one run of {t['dataset']} under "
                             f"results/forman_eigs/, found {len(combos)}")
        per_fold = fold_metrics(combos[0], t["norm_content"])
        for ckpt, m in per_fold.items():
            for key, values in m.items():
                if key == "folds":
                    continue
                metric = key if t["norm_content"] else UNNORM_RENAME.get(key, key)
                # average over the d eigenvalue channels; NaN (a constant
                # profile has no correlation) propagates to a blank cell
                cell = values.mean(axis=2)
                for fi, fold in enumerate(m["folds"]):
                    for layer in range(cell.shape[1]):
                        rows.append(dict(table=t["name"], metric=metric, checkpoint=ckpt,
                                         layer=layer + 1, fold=fold, value=cell[fi, layer]))
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FOLDS_CSV, index=False)
    print(f"wrote {len(rows)} rows -> {FOLDS_CSV}")


# ── stage 2: LaTeX ─────────────────────────────────────────────────────────────
def _frac(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0


def _colorbar(cmap, vmin, vmax, registry, n_segments=12, height_cm=2.6, width_cm=0.18):
    seg = height_cm / n_segments
    lines = [r"\smash{\begin{tikzpicture}[baseline=(current bounding box.north)]"]
    for i in range(n_segments):
        name = registry.name_for(_cmap_rgb((i + 0.5) / n_segments, cmap))
        lines.append(rf"\fill[{name}] (0,{i * seg:.4f}) rectangle ({width_cm},{(i + 1) * seg:.4f});")
    lines.append(rf"\draw[line width=0.3pt] (0,0) rectangle ({width_cm},{height_cm:.4f});")
    for y, v, anchor in ((height_cm, vmax, "north west"), (height_cm / 2, (vmin + vmax) / 2, "west"),
                         (0, vmin, "south west")):
        lines.append(rf"\node[anchor={anchor},font=\tiny,inner sep=1pt] at ({width_cm},{y:.4f}) {{{v:.2f}}};")
    lines.append(r"\end{tikzpicture}}")
    return "\n".join(lines)


def render(t, df, registry):
    layers = int(df["layer"].max())
    stats = (df.groupby(["metric", "checkpoint", "layer"])["value"]
               .agg(mean="mean", se=lambda v: v.std(ddof=1) / math.sqrt(len(v))))
    ncols = layers + 2
    out = [r"\begin{table}[p]", rf"\caption{{{t['caption']}}}", rf"\label{{tab:{t['name']}}}",
           r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.15}%",
           r"\begin{tabular}{l" + r">{\centering\arraybackslash}p{1.55cm}" * layers + "l}",
           r"\toprule", rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textbf{{{t['title']}}}}} \\",
           rf" & \multicolumn{{{layers}}}{{c}}{{Layer}} & \\", rf"\cmidrule(l){{2-{layers + 1}}}",
           "Epoch & " + " & ".join(str(i) for i in range(1, layers + 1)) + r" & \\"]
    for bi, metric in enumerate(t["metrics"]):
        label, (cmap, vmin, vmax) = METRICS[metric]
        if vmax is None:
            vmax = float(np.nanmax([stats.loc[(metric, c, l), "mean"]
                                    for c in CHECKPOINTS for l in range(1, layers + 1)]))
        out += [r"\midrule", rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{{label}}}}} \\"]
        for ri, ckpt in enumerate(CHECKPOINTS):
            cells = []
            for layer in range(1, layers + 1):
                mean, se = stats.loc[(metric, ckpt, layer)]
                if not np.isfinite(mean):
                    cells.append("")
                    continue
                name = registry.name_for(_cmap_rgb(_frac(mean, vmin, vmax), cmap))
                cells.append(rf"\cellcolor{{{name}}}${mean:.2f}_{{\pm{se:.2f}}}$")
            bar = _colorbar(cmap, vmin, vmax, registry) if ri == 0 else ""
            out.append(f"{ROW_LABEL[ckpt]} & " + " & ".join(cells) + f" & {bar} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return out


def write_tables():
    df = pd.read_csv(FOLDS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        registry = _ColorRegistry(t["name"])
        body = render(t, df[df["table"] == t["name"]], registry)
        text = "\n".join([f"% auto-generated by Bochner_code/forman_profile/"
                          f"make_forman_perdataset_tables.py", registry.definitions()] + body) + "\n"
        path = OUT_DIR / f"{t['name']}.tex"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_eigs", action="store_true",
                    help="recompute the per-fold CSV from results/forman_eigs/ first")
    if ap.parse_args().from_eigs:
        harvest()
    write_tables()
