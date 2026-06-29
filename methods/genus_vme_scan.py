"""genus-iso 가 organism-iso 대비 VME 를 새로/추가로 만드는 cell 스캔 (안전성 위험 목록).

배경: genus 단위 isotonic 은 aggregate held-out PASS +4 지만, species 차이를 뭉개므로 특정
cell 에서 R 균주를 S 로 오판(VME)할 수 있다(예 S.aureus×LNZ). 운영 채택 전 §6 필수 점검.

per-cell 로 organism-iso vs genus-iso 의 VME isolate 수를 비교해 ΔVME>0 cell 을 위험 목록으로.
in-sample(배포 자산 기준)과 OOF(held-out, 정직) 둘 다.

입력: 기존 eval 산출 (organism_insample, genus_insample, flip_org_oof, flip_gen_oof).
산출: output/genus_vme_risk.csv + 콘솔 요약.
"""
from __future__ import annotations

import pandas as pd

from agent_system.methods import base as B

OUT = B.PKG_DIR / "output"
KEY = ["organism_group", "antimicrobial"]
RUNS = {  # (organism_dir, genus_dir) per mode
    "insample": ("organism_insample", "genus_insample"),
    "oof": ("flip_org_oof", "flip_gen_oof"),
}


def _percell(dirname: str) -> pd.DataFrame:
    ev = pd.read_csv(OUT / dirname / "method_eval_model_pred.csv")
    ev["isR"] = ev["bmd_sir"] == "R"
    g = ev.groupby(KEY).agg(n=("sample_id", "size"), nR=("isR", "sum"),
                            VME=("isVME", "sum"), ME=("isME", "sum")).reset_index()
    sm = pd.read_csv(OUT / dirname / "method_summary_model_pred.csv")
    g = g.merge(sm[KEY + ["FDA_fail_list"]], on=KEY, how="left")
    return g


def main() -> None:
    frames = []
    for mode, (od, gd) in RUNS.items():
        o = _percell(od).add_suffix("_org").rename(columns={f"{k}_org": k for k in KEY})
        g = _percell(gd).add_suffix("_gen").rename(columns={f"{k}_gen": k for k in KEY})
        m = o.merge(g, on=KEY, how="outer")
        m["mode"] = mode
        m["dVME"] = m["VME_gen"].fillna(0) - m["VME_org"].fillna(0)
        # genus verdict 에 VME 가 새로 등장(organism 엔 없음)
        m["VME_new_in_verdict"] = (m["FDA_fail_list_gen"].astype(str).str.contains("VME")
                                   & ~m["FDA_fail_list_org"].astype(str).str.contains("VME"))
        frames.append(m)
    allm = pd.concat(frames, ignore_index=True)

    risk = allm[allm["dVME"] > 0].copy()
    risk["VME_rate_gen"] = (risk["VME_gen"] / risk["nR_gen"].replace(0, pd.NA)).astype(float)
    cols = (KEY + ["mode", "nR_gen", "VME_org", "VME_gen", "dVME", "VME_rate_gen",
                   "ME_org", "ME_gen", "FDA_fail_list_org", "FDA_fail_list_gen",
                   "VME_new_in_verdict"])
    risk = risk[cols].sort_values(["mode", "dVME"], ascending=[True, False])
    risk.to_csv(OUT / "genus_vme_risk.csv", index=False)

    for mode in ("insample", "oof"):
        sub = allm[allm["mode"] == mode]
        tot_o = int(sub["VME_org"].fillna(0).sum()); tot_g = int(sub["VME_gen"].fillna(0).sum())
        rs = risk[risk["mode"] == mode]
        print(f"\n{'='*82}\n[{mode}] genus VME 위험 — 총 VME isolate organism {tot_o} → genus {tot_g} "
              f"(Δ{tot_g-tot_o:+d})")
        print(f"  ΔVME>0 cell {len(rs)}개 | 그 중 verdict 에 VME 새로 등장 "
              f"{int(rs['VME_new_in_verdict'].sum())}개")
        if len(rs):
            show = rs.copy()
            show["VME"] = show["VME_org"].fillna(0).astype(int).astype(str) + "→" + \
                show["VME_gen"].fillna(0).astype(int).astype(str)
            show["verdict"] = show["FDA_fail_list_org"].astype(str).str[:10] + " → " + \
                show["FDA_fail_list_gen"].astype(str).str[:12]
            print(show[KEY + ["nR_gen", "VME", "VME_rate_gen", "verdict",
                              "VME_new_in_verdict"]].to_string(index=False))
    print(f"\n[saved] {OUT/'genus_vme_risk.csv'}")


if __name__ == "__main__":
    main()
