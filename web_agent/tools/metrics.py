"""평가 수치 조회 도구 (R 등급). 모든 수치는 정본 산출물 파일에서 읽는다 — 새로 계산하지 않는다.

정본 원천 (2026-09-30 확인):
  3.0 임상 셀   claudeCode/output/struct_passfail_all.csv (build_improvable_cells.py, 모델×organism_group×약제, raw@0.5)
  3.0 배포 셀   claudeCode/data/FDA_Clinical_selected_model.xlsx [summary_1]
  3.0 재현성    claudeCode/output/repro/Reproducibility_<model>.xlsx [FDA_summary] · 배포=allinfo/analytical/FDA_Analytical_Reproducibility.xlsx
  2.5 모델요약  drast_25_lrcn/newmodel/data/night_axes25.csv (build_night_axes25.py)
  2.5 임상 셀   _cells25.py (night_axes25 와 같은 경로로 cell_table, 111/111 일치 검증)
  2.5 재현성    drast_25_lrcn/newmodel/output/repro_ivdr.csv (tag=<canon>__common, IVDR: site 4행 n>=90 & EA>=95)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .. import config
from . import tool

R30 = config.ROOT / "claudeCode"
R25 = Path("/home/kptae/project/drast_25_lrcn/newmodel")
F30_CELLS = R30 / "output/struct_passfail_all.csv"
F30_DEPLOY = R30 / "data/FDA_Clinical_selected_model.xlsx"
D30_REPRO = R30 / "output/repro"
F30_DEPLOY_REPRO = Path("/home/kptae/data/allinfo/analytical/FDA_Analytical_Reproducibility.xlsx")
F25_AXES = R25 / "data/night_axes25.csv"
F25_REPRO = R25 / "output/repro_ivdr.csv"
PY25 = "/home/kptae/miniconda3/envs/qnt_algorithm/bin/python"
MAX_ROWS = 40

COLNAMES = {"org": "균종", "antimicrobial": "약제", "n": "검체_패널_수", "EA": "EA_%", "CA": "CA_%",
            "VME": "VME_건수", "ME": "ME_건수", "pass": "FDA_PASS", "reason": "실패_기준", "major": "major_셀"}
VERSION = {"type": "string", "enum": ["3.0", "2.5"], "description": "dRAST 세대. 불분명하면 사용자에게 물을 것"}
DEPLOYED = {"deployed", "운영", "배포", "deployed_model", "same_mic"}


def _mtime(p: Path) -> str:
    return pd.Timestamp(os.path.getmtime(p), unit="s").strftime("%Y-%m-%d %H:%M")


# ---------- 3.0 ----------
@lru_cache(maxsize=1)
def _cells30() -> pd.DataFrame:
    sys.path.insert(0, str(config.ROOT))
    from claudeCode import model_naming as mn
    d = pd.read_csv(F30_CELLS)
    canon = {m: mn.resolve(m) for m in d.model.unique()}
    d["canon"] = d.model.map(canon)
    return d


_ALIAS = {"nc_diff": "ncdiff", "nc-diff": "ncdiff", "complex_diff_next": "cdn", "시드": "s", "seed": "s"}


def _tokens(q: str) -> list[str]:
    """'nc_diff full ctlfix 시드 42' → ['ncdiff','full','ctlfix','s42']"""
    q = q.lower()
    for k, v in _ALIAS.items():
        q = q.replace(k, v)
    q = re.sub(r"\bs\s+(\d+)", r"s\1", q)
    return [t for t in re.split(r"[\s,/]+", q) if t]


def _fuzzy(names, q: str) -> list[str]:
    toks = _tokens(q)
    hits = [n for n in names if all(re.search(rf"(^|[_\-]){re.escape(t)}($|[_\-])", n.lower()) or
                                    (len(t) > 3 and t in n.lower()) for t in toks)]
    return sorted(set(hits), key=len)


def _pick30(model: str) -> str:
    d = _cells30()
    m = model.strip()
    for col in ("model", "canon"):
        hit = d.loc[d[col] == m, "model"]
        if len(hit):
            return hit.iloc[0]
    cands = _fuzzy(d.model.unique(), m)
    cands = cands or sorted(set(d.loc[d.canon.isin(_fuzzy(d.canon.unique(), m)), "model"]))
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"3.0 모델 '{model}' 을 하나로 특정 못함. 후보 {len(cands)}개: {cands[:15]} "
                     "→ 후보 중 하나로 다시 호출하거나 사용자에게 어느 모델인지 물을 것")


def _deployed30() -> pd.DataFrame:
    x = pd.read_excel(F30_DEPLOY, sheet_name="summary_1")
    cols = ["organism_group", "antimicrobial", "major_drug_bug_group", "total", "EAp", "CAp", "VMEp", "MEp",
            "VME", "ME", "FDA_fail_list"]
    x = x[[c for c in cols if c in x.columns]].copy()
    # 정본(achievement/build_initial_vs_current_pass.py:104)과 같은 판정: 통과 셀은 문자열 'PASS'
    x["pass"] = x["FDA_fail_list"].astype(str).str.strip().eq("PASS")
    return x


# ---------- 2.5 ----------
@lru_cache(maxsize=1)
def _axes25() -> pd.DataFrame:
    return pd.read_csv(F25_AXES)


def _pick25(model: str) -> str:
    a = _axes25()
    if model in set(a.canon):
        return model
    cands = _fuzzy(a.canon, model)
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"2.5 모델 '{model}' 을 하나로 특정 못함. 후보 {len(cands)}개: {cands[:15]} "
                     "→ 후보 중 하나로 다시 호출하거나 사용자에게 어느 모델인지 물을 것")


@lru_cache(maxsize=64)
def _cells25(canon: str) -> tuple[pd.DataFrame, str]:
    r = subprocess.run([PY25, str(Path(__file__).with_name("_cells25.py")), "cells", canon],
                       capture_output=True, text=True, timeout=600)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if "error" in out:
        raise ValueError(out["error"])
    return pd.DataFrame(out["table"]), out["source"]


# ---------- 공통 ----------
_ALL = {"", "all", "전체", "*", "모두", "none", "null"}


def _norm(v):
    return None if v is None or str(v).strip().lower() in _ALL else str(v).strip()


def _filter(d: pd.DataFrame, org_col: str, organism: str | None, drug: str | None) -> pd.DataFrame:
    organism, drug = _norm(organism), _norm(drug)
    if organism:
        d = d[d[org_col].astype(str).str.contains(organism, case=False, regex=False)]
    if drug:
        d = d[d["antimicrobial"].astype(str).str.upper() == drug.upper()]
    return d


def _cell_frame(version: str, model: str) -> tuple[pd.DataFrame, dict]:
    """(표준화된 셀 표[org, antimicrobial, n, EA, CA, VME, ME, pass, reason], 출처정보)"""
    if version == "3.0":
        if model.lower() in DEPLOYED:
            x = _deployed30()
            pct = lambda v: pd.to_numeric(v, errors="coerce") * 100  # noqa: E731 — summary_1 은 0~1 비율
            f = pd.DataFrame({"org": x.organism_group, "antimicrobial": x.antimicrobial, "n": x.total,
                              "EA": pct(x.EAp), "CA": pct(x.CAp), "VME": x.VME, "ME": x.ME, "pass": x["pass"],
                              "reason": x.FDA_fail_list})
            return f, {"평가셋": "fda2023 임상(FDA2023 clinical)", "model": "deployed(운영 라우팅+brightness)", "source": str(F30_DEPLOY), "mtime": _mtime(F30_DEPLOY),
                       "note": "summary_1 전체 365행(스크리닝 약제 포함). 성과표의 '운영 166/317'은 MIC 셀만 센 별도 집계"}
        k = _pick30(model)
        x = _cells30()
        x = x[x.model == k]
        f = pd.DataFrame({"org": x.organism_group, "antimicrobial": x.antimicrobial, "n": x.n, "EA": x.EA,
                          "CA": x.CA, "VME": x.VME, "ME": x.ME, "pass": x["pass"], "reason": x.reason,
                          "major": x.major})
        return f, {"평가셋": "fda2023 임상(FDA2023 clinical)", "model": k, "canonical": x.canon.iloc[0], "source": str(F30_CELLS), "mtime": _mtime(F30_CELLS),
                   "note": "raw@0.5 (iso 보정 전)"}
    c = _pick25(model)
    t, src = _cells25(c)
    f = pd.DataFrame({"org": t.microbial_id, "antimicrobial": t.antimicrobial, "n": t.n, "EA": t.EA, "CA": t.CA,
                      "VME": t.VME, "ME": t.ME, "pass": t.passed, "reason": t.get("reason")})
    return f, {"평가셋": "2.5 임상(d170 Test)", "model": c, "source": src, "mtime": _mtime(Path(src))}


@tool("list_models",
      "평가 결과가 있는 모델 목록과 모델별 요약(PASS 셀 수 등)을 돌려준다. 모델 이름을 모르거나 후보를 좁힐 때 먼저 쓴다.",
      {"type": "object", "properties": {"version": VERSION,
                                        "query": {"type": "string", "description": "이름 부분 문자열 필터(선택)"}},
       "required": ["version"]})
def list_models(version: str, query: str = ""):
    if version == "3.0":
        d = _cells30()
        g = d.groupby(["model", "canon"]).agg(cells=("pass", "size"), pass_cells=("pass", "sum"),
                                              VME=("VME", "sum"), ME=("ME", "sum")).reset_index()
        if query:
            keep = set(_fuzzy(g.model, query)) | set(g.loc[g.canon.isin(_fuzzy(g.canon, query)), "model"])
            g = g[g.model.isin(keep)]
        g = g.sort_values("pass_cells", ascending=False)
        dp = _deployed30()
        deployed = {"model": "deployed", "설명": "운영 시스템(라우팅+brightness 10h), cell_metrics(model='deployed')로 상세 조회",
                    "cells": len(dp), "pass_cells": int(dp["pass"].sum()),
                    "VME": int(pd.to_numeric(dp.VME, errors="coerce").sum()), "ME": int(pd.to_numeric(dp.ME, errors="coerce").sum()),
                    "note": "summary_1 365행(스크리닝 포함) — 구조모델 313셀과 분모가 다름"}
        return {"version": "3.0", "운영_시스템": deployed, "source": str(F30_CELLS), "mtime": _mtime(F30_CELLS),
                "note": "임상 FDA2023, raw@0.5. 운영 시스템은 model='deployed' 로 cell_metrics 조회",
                "n_models": len(g), "models": g.head(MAX_ROWS).to_dict("records")}
    a = _axes25()
    cols = ["canon", "group", "clin_pass", "clin_cells", "clin_EA", "clin_CA", "clin_VME", "clin_ME",
            "repro_pass", "repro_drugs", "config"]
    if query:
        a = a[a.canon.isin(_fuzzy(a.canon, query))]
    a = a.sort_values("clin_pass", ascending=False)[cols].round(2)
    return {"version": "2.5", "source": str(F25_AXES), "mtime": _mtime(F25_AXES),
            "note": "clin_EA/CA 는 패널 단위 %, clin_VME/ME 는 건수, repro_pass 는 PASS 약제 수",
            "n_models": len(a), "models": a.head(MAX_ROWS).to_dict("records")}


@tool("cell_metrics",
      "한 모델의 셀(균종×약제) 단위 **임상 평가셋** 지표(n, EA%, CA%, VME/ME 건수, FDA PASS 여부와 실패 사유)를 조회한다. "
      "균종·약제로 거를 수 있다. 3.0 운영 시스템은 model='deployed'. "
      "재현성은 reproducibility 도구. stability(안정성) 등 다른 평가셋 결과는 현재 어떤 도구에도 없다 — 없다고 답할 것.",
      {"type": "object",
       "properties": {"version": VERSION,
                      "model": {"type": "string", "description": "모델 이름(정식/legacy/부분 문자열) 또는 deployed"},
                      "organism": {"type": "string", "description": "균종 부분 문자열(선택). 3.0=organism_group, 2.5=microbial_id"},
                      "drug": {"type": "string", "description": "약제 약어(선택), 예: CTX, MP"},
                      "only": {"type": "string", "enum": ["all", "pass", "fail"], "description": "기본 all"}},
       "required": ["version", "model"]})
def cell_metrics(version: str, model: str, organism: str | None = None, drug: str | None = None, only: str = "all"):
    f, info = _cell_frame(version, model)
    f = _filter(f, "org", organism, drug)
    if f.empty:
        return {**info, "error": f"조건에 맞는 셀 없음 (organism={organism}, drug={drug}). 균종/약제 철자를 확인할 것"}
    base = {"필터_전_셀_수": len(f), "필터_전_PASS_셀_수": int((f["pass"] == True).sum())}  # noqa: E712
    if only == "pass":
        f = f[f["pass"] == True]  # noqa: E712
    elif only == "fail":
        f = f[f["pass"] == False]  # noqa: E712
    summ = {**base, "표시_필터": only, "셀_수": len(f), "PASS_셀_수": int((f["pass"] == True).sum()),  # noqa: E712
            "VME_건수_합": int(pd.to_numeric(f.VME, errors="coerce").fillna(0).sum()),
            "ME_건수_합": int(pd.to_numeric(f.ME, errors="coerce").fillna(0).sum()),
            "검체_패널_수_합": int(pd.to_numeric(f.n, errors="coerce").fillna(0).sum())}
    rows = f.round(2).rename(columns=COLNAMES).head(MAX_ROWS).to_dict("records")
    return {**info, "version": version, "summary": summ, "rows_shown": len(rows),
            "rows_truncated": max(len(f) - len(rows), 0), "cells_table": rows}


@tool("reproducibility",
      "모델의 재현성 평가 결과(약제별 PASS)를 조회한다. 3.0=FDA analytical(약제×site, 'All sites' 행이 최종), "
      "2.5=IVDR(site 4행 모두 n>=90 & EA>=95). 3.0 운영 시스템은 model='deployed'.",
      {"type": "object", "properties": {"version": VERSION, "model": {"type": "string"},
                                        "drug": {"type": "string", "description": "약제 약어(선택)"}},
       "required": ["version", "model"]})
def reproducibility(version: str, model: str, drug: str | None = None):
    if version == "3.0":
        if model.lower() in DEPLOYED:
            f, name = F30_DEPLOY_REPRO, "deployed"
        else:
            k = _pick30(model) if not (D30_REPRO / f"Reproducibility_{model}.xlsx").exists() else model
            f, name = D30_REPRO / f"Reproducibility_{k}.xlsx", k
            if not f.exists():
                return {"error": f"재현성 결과 파일 없음: {f.name}"}
        x = pd.read_excel(f, sheet_name="FDA_summary")
        drug = _norm(drug)
        if drug and x.Antimicrobial.astype(str).str.upper().eq(drug.upper()).sum() == 0:
            return {"error": f"재현성 결과에 약제 '{drug}' 없음. 있는 약제: {sorted(x.Antimicrobial.astype(str).unique())}"}
        if drug:
            x = x[x.Antimicrobial.astype(str).str.upper() == drug.upper()]
        final = x[x.Site.astype(str) == "All sites"]
        return {"version": "3.0", "model": name, "source": str(f), "mtime": _mtime(f),
                "drugs": int(final.Antimicrobial.nunique()),
                "pass_drugs": int(final.Pass.astype(str).isin(["True", "PASS", "Pass"]).sum()),
                "fail_drugs": final.loc[~final.Pass.astype(str).isin(["True", "PASS", "Pass"]), "Antimicrobial"].tolist(),
                "table": x.round(2).head(MAX_ROWS * 2).to_dict("records")}
    c = _pick25(model)
    x = pd.read_csv(F25_REPRO)
    # repro_ivdr.csv 태그는 'lightpanel25' 접두어 없이 저장된 것이 있음
    x = x[x.tag.isin([f"{c}__common", f"{c.removeprefix('lightpanel25')}__common"])]
    if x.empty:
        a = _axes25().set_index("canon").loc[c]
        if pd.isna(a.get("repro_pass")):
            return {"error": f"2.5 재현성 결과 없음: {c}"}
        return {"version": "2.5", "model": c, "source": str(F25_AXES), "mtime": _mtime(F25_AXES),
                "note": "repro_ivdr.csv 에 상세 없음 → night_axes25 요약만", "pass_drugs": int(a.repro_pass),
                "drugs": int(a.repro_drugs), "fail": a.get("repro_fail")}
    drug = _norm(drug)
    if drug:
        if x.Antimicrobial.astype(str).str.upper().eq(drug.upper()).sum() == 0:
            return {"error": f"재현성 결과에 약제 '{drug}' 없음. 있는 약제: {sorted(x.Antimicrobial.astype(str).unique())}"}
        x = x[x.Antimicrobial.astype(str).str.upper() == drug.upper()]
    ok = x.groupby("Antimicrobial").Pass.apply(lambda s: bool(s.astype(str).isin(["True", "PASS"]).all()))
    return {"version": "2.5", "model": c, "source": str(F25_REPRO), "mtime": _mtime(F25_REPRO),
            "drugs": len(ok), "pass_drugs": int(ok.sum()), "fail_drugs": ok[~ok].index.tolist(),
            "table": x.drop(columns=["tag"]).round(2).head(MAX_ROWS * 2).to_dict("records")}


@tool("compare_models",
      "두 모델을 같은 셀 기준으로 비교한다: 공통 셀 수, PASS 전환 양방향(A만 PASS / B만 PASS), 셀별 EA 승패, VME·ME 변화. "
      "모델 비교 질문에는 반드시 이 도구를 쓴다(합계만 비교 금지).",
      {"type": "object", "properties": {"version": VERSION, "model_a": {"type": "string"}, "model_b": {"type": "string"},
                                        "organism": {"type": "string"}, "drug": {"type": "string"}},
       "required": ["version", "model_a", "model_b"]})
def compare_models(version: str, model_a: str, model_b: str, organism: str | None = None, drug: str | None = None):
    fa, ia = _cell_frame(version, model_a)
    fb, ib = _cell_frame(version, model_b)
    fa, fb = _filter(fa, "org", organism, drug), _filter(fb, "org", organism, drug)
    m = fa.merge(fb, on=["org", "antimicrobial"], suffixes=("_a", "_b"))
    for c in ("EA", "VME", "ME"):
        m[f"{c}_a"] = pd.to_numeric(m[f"{c}_a"], errors="coerce")
        m[f"{c}_b"] = pd.to_numeric(m[f"{c}_b"], errors="coerce")
    pa, pb = m["pass_a"] == True, m["pass_b"] == True  # noqa: E712
    d_ea = (m.EA_b - m.EA_a).round(2)
    key = lambda s: (s["org"] + "×" + s["antimicrobial"]).tolist()  # noqa: E731
    big = m.assign(dEA=d_ea).loc[d_ea.abs() >= 5, ["org", "antimicrobial", "n_a", "EA_a", "EA_b", "dEA"]]
    gained, lost = key(m[~pa & pb]), key(m[pa & ~pb])
    A, B = ia["model"], ib["model"]   # 필드명에 모델 이름을 직접 넣어 A/B 방향 혼동을 막는다
    return {"version": version, "모델_출처": {A: ia, B: ib},
            "읽는법": "개수는 *_수 필드를 그대로 인용하고 목록을 직접 세지 말 것. VME·ME 는 적을수록 좋음",
            "공통_셀_수": len(m), f"{A}에만_있는_셀_수": len(fa) - len(m), f"{B}에만_있는_셀_수": len(fb) - len(m),
            f"{A}_전체_PASS_셀_수": int(pa.sum()), f"{B}_전체_PASS_셀_수": int(pb.sum()),
            f"{A}는_실패_{B}는_PASS_셀_수": len(gained), f"{A}는_실패_{B}는_PASS_셀": gained[:MAX_ROWS],
            f"{A}는_PASS_{B}는_실패_셀_수": len(lost), f"{A}는_PASS_{B}는_실패_셀": lost[:MAX_ROWS],
            f"EA가_{B}쪽이_높은_셀_수": int((d_ea > 0).sum()), f"EA가_{A}쪽이_높은_셀_수": int((d_ea < 0).sum()),
            "EA_같은_셀_수": int((d_ea == 0).sum()),
            f"{A}_VME_건수_합": int(m.VME_a.sum()), f"{B}_VME_건수_합": int(m.VME_b.sum()),
            f"{A}_ME_건수_합": int(m.ME_a.sum()), f"{B}_ME_건수_합": int(m.ME_b.sum()),
            "VME_달라진_셀": m.loc[m.VME_a != m.VME_b, ["org", "antimicrobial", "VME_a", "VME_b"]].rename(
                columns={"org": "균종", "antimicrobial": "약제", "VME_a": f"{A}_VME", "VME_b": f"{B}_VME"}).head(MAX_ROWS).to_dict("records"),
            f"EA_차이_큰_셀({B}-{A}, |차이|>=5%p)": big.sort_values("dEA").head(MAX_ROWS).round(2).rename(
                columns={"org": "균종", "antimicrobial": "약제", "n_a": "검체_패널_수", "EA_a": f"{A}_EA_%", "EA_b": f"{B}_EA_%",
                         "dEA": f"{B}-{A}_%p"}).to_dict("records")}
