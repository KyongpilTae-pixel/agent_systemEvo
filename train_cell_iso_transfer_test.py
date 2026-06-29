"""Cell-level cross-domain ISO transferability test — train→FDA (followup of P2).

P2 (`train_iso_transfer_test.py`) tested only a *global* (cell-pooled) isotonic
calibration transfer and found ~0 effect (train-global-ISO PASS 82 ≈ routed raw
82). It could not test per-cell ISO because the train-domain inference output
(`output_dtw_aggregate_train_domain/dtw_per_conc.csv`) had an **empty**
`organism_group` column — the analyze step resolved 0/27128 rows against the FDA
Clinical mapping CSV (wrong project).

This followup resolves that: the train sample_ids map 558/558 against
`2024_0709_traintest_allInfo.csv` (the canonical train mapping already referenced
by `organism_normalize.py`). With organism_group recovered we can fit a *per-cell*
isotonic calibrator on the train domain and apply it to the matching FDA cells.

Conditions compared on the FDA per-conc (routed) table:
  - raw            : FDA raw model_pred, no calibration
  - fda_cell_iso   : per-cell ISO fit on FDA, applied to FDA (in-sample; the
                     operational `routed_iso_t65` family — upper bound)
  - train_cell_iso : per-cell ISO fit on TRAIN, applied to matching FDA cells
                     (identity fallback for non-train cells) — the cross-domain test
  - train_global_iso : single global ISO fit on TRAIN (P2 control)

CAVEAT (reported): train-domain inference is *in-sample* (same data the model was
trained on), so train model_pred is far sharper than FDA (train model_pred AUROC
~0.999 vs FDA ~0.95). A train-fit ISO is therefore close to a step at the
decision boundary and may transfer poorly. This is itself a finding.

Outputs (agent_system/output/cross_domain_iso_cell/):
  - metrics_summary.csv      macro/overall Brier + NG/G-miss per condition & threshold
  - per_cell_transfer.csv    per-cell Brier/NG-miss under raw vs train-ISO vs FDA-ISO
  - dtw_per_conc_routed_train_cell_iso.csv   calibrated per_conc for downstream PASS
  - train_cell_iso_transfer.html             report
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from claudeCode.isotonic_lookup import PerCellIsotonicLookup
from claudeCode.organism_normalize import normalize_organism_group

ROOT = Path("/home/kptae/project/qnt_algorithm")
TRAIN_PER_CONC = ROOT / "claudeCode/output_dtw_aggregate_train_domain/dtw_per_conc.csv"
FDA_PER_CONC = ROOT / "claudeCode/output_subset_eval/dtw_per_conc_routed.csv"
TRAIN_MAP = "/data/dRAST30_prepare_csv/2024_0709_traintest_allInfo.csv"

MODEL_COL = "dtw_model_pred"
LABEL_COL = "gt_gng"
OG = "organism_group"
AMR = "antimicrobial"


def enrich_train_organism(train_csv: Path, map_csv: str) -> pd.DataFrame:
    """Load train per_conc, replace its empty organism_group with the mapping
    CSV's value, normalized to FDA-canonical labels."""
    df = pd.read_csv(train_csv, low_memory=False)
    if OG in df.columns:
        df = df.drop(columns=[OG])
    mp = pd.read_csv(
        map_csv, usecols=lambda c: c in ("sample_id", OG), low_memory=False
    ).drop_duplicates("sample_id")
    mp["sample_id"] = mp["sample_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.merge(mp, on="sample_id", how="left")
    df[OG] = df[OG].map(normalize_organism_group)
    return df


def fit_global_iso(df: pd.DataFrame) -> IsotonicRegression:
    m = df[MODEL_COL].notna() & df[LABEL_COL].notna()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(df.loc[m, MODEL_COL].astype(float), df.loc[m, LABEL_COL].astype(int))
    return iso


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def miss_counts(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[int, int]:
    pred = (p >= thr).astype(int)
    ng_miss = int(((pred == 0) & (y == 1)).sum())  # NG called G → VME-ish
    g_miss = int(((pred == 1) & (y == 0)).sum())    # G called NG → ME-ish
    return ng_miss, g_miss


def per_cell_brier_table(df: pd.DataFrame, score_cols: dict[str, str]
                         ) -> pd.DataFrame:
    rows = []
    sub_all = df.dropna(subset=[OG, AMR, LABEL_COL])
    for (og, amr), sub in sub_all.groupby([OG, AMR], sort=False):
        y = sub[LABEL_COL].astype(int).to_numpy()
        if len(sub) < 4 or len(set(y.tolist())) < 2:
            continue
        rec = {OG: og, AMR: amr, "n": int(len(sub)),
               "n_ng": int((y == 1).sum()), "n_g": int((y == 0).sum())}
        for name, col in score_cols.items():
            s = sub[col].astype(float).to_numpy()
            m = np.isfinite(s)
            if m.sum() < 4:
                rec[f"brier_{name}"] = np.nan
                rec[f"ngmiss_{name}"] = np.nan
                continue
            rec[f"brier_{name}"] = brier(s[m], y[m])
            ng, _ = miss_counts(s[m], y[m], 0.5)
            rec[f"ngmiss_{name}"] = ng
        rows.append(rec)
    return pd.DataFrame(rows)


def _fig_uri(fig, dpi=130) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_per_conc", default=str(TRAIN_PER_CONC))
    p.add_argument("--fda_per_conc", default=str(FDA_PER_CONC))
    p.add_argument("--train_map", default=TRAIN_MAP)
    p.add_argument("--output_dir",
                   default="agent_system/output/cross_domain_iso_cell")
    p.add_argument("--thresholds", default="0.5,0.65")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # ---- 1) train domain: enrich organism + fit per-cell & global ISO -------
    print("[load+enrich] train per_conc")
    train = enrich_train_organism(Path(args.train_per_conc), args.train_map)
    print(f"  train rows={len(train)}  org resolved={train[OG].notna().sum()}")
    train_lk = PerCellIsotonicLookup.fit(train, min_rows=30, min_per_class=3)
    print(f"  train per-cell ISO cells = {len(train_lk.cells)}")
    iso_train_global = fit_global_iso(train)

    # ---- 2) FDA: normalize organism + fit per-cell ISO (in-sample) ----------
    print("[load] FDA per_conc (routed)")
    fda = pd.read_csv(args.fda_per_conc, low_memory=False)
    fda[OG] = fda[OG].map(normalize_organism_group)
    fda_lk = PerCellIsotonicLookup.fit(fda, min_rows=30, min_per_class=3)
    print(f"  FDA per-cell ISO cells = {len(fda_lk.cells)}")

    # cell overlap
    tcells = set(train_lk.cells.keys())
    fcells = set(fda_lk.cells.keys())
    fda_present = set(
        fda.dropna(subset=[OG, AMR]).groupby([OG, AMR]).groups.keys())
    overlap = tcells & fda_present
    print(f"  train-fit cells={len(tcells)} FDA-present={len(fda_present)} "
          f"transferable overlap={len(overlap)}")

    # ---- 3) build calibrated score columns on FDA ---------------------------
    fda = fda.dropna(subset=[MODEL_COL]).copy()
    fda["score_raw"] = fda[MODEL_COL].astype(float).clip(0, 1)
    fda = fda_lk.calibrate_dataframe(fda, out_col="score_fda_cell_iso")
    fda = train_lk.calibrate_dataframe(fda, out_col="score_train_cell_iso")
    m = fda[MODEL_COL].notna()
    fda.loc[m, "score_train_global_iso"] = iso_train_global.predict(
        fda.loc[m, MODEL_COL].astype(float))

    # mark whether row's cell got a real train-ISO recalibration (vs identity)
    fda["train_cell_covered"] = fda.apply(
        lambda r: (str(r[OG]), str(r[AMR])) in tcells, axis=1)

    # ---- 4) aggregate metrics ----------------------------------------------
    cond = {
        "raw": "score_raw",
        "fda_cell_iso": "score_fda_cell_iso",
        "train_cell_iso": "score_train_cell_iso",
        "train_global_iso": "score_train_global_iso",
    }
    valid = fda.dropna(subset=[LABEL_COL])
    y_all = valid[LABEL_COL].astype(int).to_numpy()
    # subset to the transferable cells (apples-to-apples for the transfer story)
    cov = valid[valid["train_cell_covered"]]
    y_cov = cov[LABEL_COL].astype(int).to_numpy()

    summ_rows = []
    for name, col in cond.items():
        for scope, sub, ys in (("all_rows", valid, y_all),
                               ("covered_cells_only", cov, y_cov)):
            s = sub[col].astype(float).to_numpy()
            msk = np.isfinite(s)
            row = {"condition": name, "scope": scope, "n": int(msk.sum()),
                   "brier": brier(s[msk], ys[msk])}
            for thr in thresholds:
                ng, g = miss_counts(s[msk], ys[msk], thr)
                row[f"ng_miss@{thr}"] = ng
                row[f"g_miss@{thr}"] = g
            summ_rows.append(row)
    summ = pd.DataFrame(summ_rows)
    summ_csv = out_dir / "metrics_summary.csv"
    summ.to_csv(summ_csv, index=False)
    print(f"\n[saved] {summ_csv}")
    print(summ.to_string(index=False))

    # ---- 5) per-cell table (covered cells) ----------------------------------
    pcell = per_cell_brier_table(
        cov, {"raw": "score_raw", "fda_iso": "score_fda_cell_iso",
              "train_iso": "score_train_cell_iso"})
    pcell["d_brier_train"] = pcell["brier_train_iso"] - pcell["brier_raw"]
    pcell["d_brier_fda"] = pcell["brier_fda_iso"] - pcell["brier_raw"]
    pcell["d_ngmiss_train"] = pcell["ngmiss_train_iso"] - pcell["ngmiss_raw"]
    pcell["d_ngmiss_fda"] = pcell["ngmiss_fda_iso"] - pcell["ngmiss_raw"]
    pcell = pcell.sort_values("d_brier_train")
    pcell_csv = out_dir / "per_cell_transfer.csv"
    pcell.to_csv(pcell_csv, index=False)
    print(f"[saved] {pcell_csv}  ({len(pcell)} cells)")

    # ---- 6) export calibrated per_conc for downstream PASS -------------------
    cal_out = fda.copy()
    cal_out[MODEL_COL] = fda["score_train_cell_iso"].astype(float)
    cal_csv = out_dir / "dtw_per_conc_routed_train_cell_iso.csv"
    cal_out.drop(columns=[c for c in cal_out.columns if c.startswith("score_")]
                 + ["train_cell_covered"], errors="ignore").to_csv(
        cal_csv, index=False)
    print(f"[saved] {cal_csv}  (for verify_method_sir_pipeline)")

    # ---- 7) manifest + HTML -------------------------------------------------
    manifest = {
        "train_per_conc": args.train_per_conc,
        "fda_per_conc": args.fda_per_conc,
        "train_map": args.train_map,
        "n_train_cells": len(tcells),
        "n_fda_cells": len(fcells),
        "n_transferable_overlap": len(overlap),
        "thresholds": thresholds,
        "outputs": {
            "metrics_summary": str(summ_csv),
            "per_cell_transfer": str(pcell_csv),
            "calibrated_per_conc": str(cal_csv),
        },
        "caveat": ("train inference is in-sample; train model_pred is far "
                   "sharper than FDA so train-fit ISO is near a step at the "
                   "boundary."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _write_html(out_dir, summ, pcell, train, fda, len(overlap), len(tcells),
                len(fda_present), thresholds)
    print(f"\n[done] report -> {out_dir / 'train_cell_iso_transfer.html'}")


def _write_html(out_dir, summ, pcell, train, fda, n_overlap, n_tcells,
                n_fda_present, thresholds):
    import matplotlib.pyplot as plt

    # plot A: train vs FDA model_pred score distribution (sharpness caveat)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(train[MODEL_COL].dropna(), bins=40, color="#c62828", alpha=0.8)
    ax[0].set_title("TRAIN model_pred (in-sample, sharp)")
    ax[0].set_xlabel("model_pred")
    ax[1].hist(fda[MODEL_COL].dropna(), bins=40, color="#1565c0", alpha=0.8)
    ax[1].set_title("FDA model_pred")
    ax[1].set_xlabel("model_pred")
    uri_dist = _fig_uri(fig)

    # plot B: Brier by condition (covered cells)
    cc = summ[summ["scope"] == "covered_cells_only"].set_index("condition")
    order = ["raw", "train_global_iso", "train_cell_iso", "fda_cell_iso"]
    briers = [cc.loc[c, "brier"] for c in order]
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
    cols = ["#888", "#ef6c00", "#c62828", "#2e7d32"]
    ax2.bar(order, briers, color=cols)
    for i, v in enumerate(briers):
        ax2.text(i, v, f" {v:.4f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Brier on covered cells (lower = better)")
    ax2.set_ylabel("Brier")
    ax2.grid(alpha=0.3, axis="y")
    uri_brier = _fig_uri(fig2)

    def tbl(df, cols, fmt):
        h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        rows = ""
        for _, r in df.iterrows():
            rows += "<tr>" + "".join(
                f"<td>{fmt(c, r[c])}</td>" for c in cols) + "</tr>"
        return f"<table><thead>{h}</thead><tbody>{rows}</tbody></table>"

    def f_summ(c, v):
        if isinstance(v, float):
            return f"{v:.4f}" if "brier" in c else f"{v:.0f}"
        return str(v)

    summ_cols = ["condition", "scope", "n", "brier"] + \
        [f"ng_miss@{t}" for t in thresholds] + [f"g_miss@{t}" for t in thresholds]
    summ_html = tbl(summ, summ_cols, f_summ)

    worst = pcell.head(12)
    best = pcell.sort_values("d_brier_train", ascending=False).head(8)

    def f_cell(c, v):
        if isinstance(v, float):
            return f"{v:+.3f}" if c.startswith("d_") else f"{v:.3f}"
        return str(v)
    cell_cols = [OG, AMR, "n", "brier_raw", "brier_train_iso", "brier_fda_iso",
                 "d_brier_train", "d_brier_fda", "d_ngmiss_train"]
    worst_html = tbl(worst, cell_cols, f_cell)
    best_html = tbl(best, cell_cols, f_cell)

    cc_raw = cc.loc["raw"]
    cc_tr = cc.loc["train_cell_iso"]
    cc_fda = cc.loc["fda_cell_iso"]
    verdict = (
        "transfer 효과 거의 없음 (train-cell-ISO Brier ≈ raw)"
        if abs(cc_tr["brier"] - cc_raw["brier"]) < 0.5 * abs(
            cc_fda["brier"] - cc_raw["brier"])
        else "train-cell-ISO가 raw 대비 유의미한 calibration 개선")

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Cell-level cross-domain ISO transfer (train→FDA)</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;background:#fafafa;margin:0;color:#222}}
 .c{{max-width:1080px;margin:0 auto;padding:24px 32px 80px}}
 h1{{font-size:21px;border-bottom:2px solid #2979ff;padding-bottom:6px}}
 h2{{font-size:16px;color:#1565c0;margin-top:26px}}
 table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}}
 th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
 th{{background:#f0f0f0}}
 td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}
 code{{background:#eee;padding:1px 5px;border-radius:3px}}
 .box{{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;margin-top:10px}}
 .verdict{{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;margin-top:12px}}
 img{{max-width:100%}}
</style></head><body><div class="c">
<h1>Cell-level cross-domain ISO transfer — train → FDA</h1>
<p>P2의 global ISO transfer(~0 효과) followup. train-domain inference의 비어있던
<code>organism_group</code>을 <code>2024_0709_traintest_allInfo.csv</code>로 복원
(558/558)하여 <b>per-cell</b> isotonic을 train에서 fit → FDA에 적용.</p>

<div class="box">
<b>cell 커버리지</b><br>
train-fit per-cell ISO = <b>{n_tcells}</b> cells · FDA present = {n_fda_present} ·
transferable overlap = <b>{n_overlap}</b> cells.
</div>

<div class="verdict"><b>결론:</b> {verdict}.<br>
covered cells Brier — raw {cc_raw['brier']:.4f} · train-cell-ISO
{cc_tr['brier']:.4f} · FDA-cell-ISO {cc_fda['brier']:.4f}.</div>

<h2>1. Caveat — train inference는 in-sample (분포가 다름)</h2>
<p>train model_pred는 학습 데이터라 분리도가 비정상적으로 높아 (거의 0/1 양극단)
ISO가 0.5 부근 step에 수렴. FDA의 연속적 model_pred 분포와 달라 transfer가
구조적으로 불리.</p>
<img src="{uri_dist}">

<h2>2. 조건별 macro 지표</h2>
<img src="{uri_brier}">
{summ_html}
<p style="font-size:11px;color:#888">scope=covered_cells_only 는 train-fit cell이
존재하는 FDA row만 (apples-to-apples). all_rows 는 미커버 cell은 identity
fallback(=raw).</p>

<h2>3. train-ISO가 가장 악화시킨 cell (Brier 증가 상위 12)</h2>
{worst_html}

<h2>4. train-ISO가 그나마 개선한 cell (상위 8)</h2>
{best_html}

<p style="color:#888;font-size:11px;margin-top:28px">Generated by
<code>python -m agent_system.train_cell_iso_transfer_test</code>.</p>
</div></body></html>"""
    (out_dir / "train_cell_iso_transfer.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
