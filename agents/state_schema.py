"""Multi-agent shared state schema — dataclasses + JSON serialization.

설계 원칙:
  - 모든 필드는 JSON-serializable primitive (str, int, float, bool, list, dict)
    또는 다른 dataclass. → Parquet 디스크 저장 + Claude Agent SDK 도구 호출 시
    자동 JSON schema 생성 가능.
  - `to_dict()` / `from_dict()` 라운드트립 보장.
  - 큰 데이터(parquet)는 path 참조만 저장 (state body는 작게 유지).

이 schema는 8 agent + orchestrator 사이 데이터 contract.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

STATE_DIR = Path("claudeCode/state")


# =====================================================================
# Recipe — score 생성 recipe (model / ensemble / calibrated)
# =====================================================================
@dataclass
class Recipe:
    """단일 scoring recipe — model 단독, ensemble, 또는 routed 조합."""
    recipe_id: str
    recipe_type: str                       # "single" | "ensemble" | "routed_ensemble"
    model_pool: list[str] = field(default_factory=list)
    ensemble_method: Optional[str] = None  # "rank_average" | "prob_mean" | "z_score" | None
    weights: list[float] = field(default_factory=list)
    apply_on_buckets: list[str] = field(default_factory=list)  # ["shifted"] or ["all"]
    fallback_recipe_id: Optional[str] = None
    calibration: Optional[str] = None      # "isotonic_per_cell" | "pooled_organism_iso" | None
    threshold: float = 0.75
    source_agent: str = "?"                # "A".."H"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    honest_eval: Optional[dict] = None     # {method, mean_auroc, std, ci_95}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        return cls(**d)


# =====================================================================
# MetricsBundle — per-cell SIR 평가 결과 묶음
# =====================================================================
@dataclass
class CellMetric:
    organism_group: str
    antimicrobial: str
    recipe_id: str
    n_rows: int
    n_NG: int
    n_G: int
    n_S: int
    n_R: int
    auroc: float
    EAp: float
    CAp: float
    MEp: float
    VMEp: float
    mEp: float
    fda_pass: bool
    fail_rule: str            # "PASS" or "EA,CA,..."


@dataclass
class MetricsBundle:
    recipe_id: str
    n_cells: int
    fda_pass_count: int
    macro_auroc: float
    global_auroc: float
    pareto: dict              # {"avg_EA":..., "sum_ME":..., "sum_VME":...}
    cells: list[CellMetric] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "cells"},
                "cells_csv_ref": f"per_cell_metrics.parquet#recipe={self.recipe_id}"}


# =====================================================================
# Validation — Overfit Sentinel 산출
# =====================================================================
@dataclass
class ValidationResult:
    recipe_id: str
    status: str               # "transferable" | "fragile" | "unknown"
    method: str               # "GroupKFold(5)" | "LOSO" | "nested_cv"
    mean_auroc: float
    std: float
    ci_95: list[float]        # [low, high]
    reason: str = ""          # if fragile, why


# =====================================================================
# Curation — Training Data Optimizer 산출
# =====================================================================
@dataclass
class CurationRecipe:
    recipe_id: str
    strategy: str             # "object_area_filter" | "rebalance_R" | "organism_normalize" | ...
    parameters: dict = field(default_factory=dict)
    target_drug: Optional[str] = None
    target_organism: Optional[str] = None
    description: str = ""


@dataclass
class CuratedDatasetManifest:
    recipe_id: str
    output_parquet: str
    n_rows_before: int
    n_rows_after: int
    n_rows_dropped: int
    rows_added: int = 0
    expected_shift_gap: Optional[float] = None
    notes: str = ""


# =====================================================================
# Manifest — orchestrator commit 단위
# =====================================================================
@dataclass
class OperationalManifest:
    cycle_id: int
    committed_at: str
    recipe_id: str
    agent_lineage: list[str]   # ["F", "H", "A", "B", "E", "C", "D", "G"]
    validated_metrics: dict    # {"fda_pass": 58, "auroc_global": 0.965, ...}
    operational_assets: dict   # {"model_lookup": path, ...}
    user_approved: bool
    diff_from_previous: dict = field(default_factory=dict)


# =====================================================================
# AgentResult — 모든 agent 메소드의 공통 반환 형식
# =====================================================================
@dataclass
class AgentResult:
    agent: str                # "A".."H" or "G"
    method: str               # e.g. "propose_routing"
    success: bool
    proposals: list[dict] = field(default_factory=list)   # 자유 schema
    audit: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)    # 생성된 파일 path

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# SharedState — agents 간 공유 상태 (디스크 backed)
# =====================================================================
@dataclass
class SharedState:
    """모든 agent가 read/write하는 공유 상태.

    Note: 큰 데이터(per-row scores)는 parquet path만 보유. body는 작게 유지해
    JSON serialize 가능.
    """
    state_dir: Path = STATE_DIR
    cycle_id: int = 0
    per_row_scores_parquet: Optional[str] = None
    per_cell_metrics_parquet: Optional[str] = None
    recipes: dict[str, dict] = field(default_factory=dict)   # recipe_id → Recipe.to_dict()
    blacklist: list[dict] = field(default_factory=list)
    shifted_set: list[list[str]] = field(default_factory=list)  # [[organism, drug], ...]
    last_manifest: Optional[dict] = None

    # ---------------- IO ----------------
    @classmethod
    def load(cls, state_dir: Path = STATE_DIR) -> "SharedState":
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        f = state_dir / "shared_state.json"
        if f.exists():
            data = json.loads(f.read_text())
            data["state_dir"] = state_dir
            return cls(**data)
        return cls(state_dir=state_dir)

    def save(self) -> None:
        body = asdict(self)
        body["state_dir"] = str(self.state_dir)
        (Path(self.state_dir) / "shared_state.json").write_text(
            json.dumps(body, indent=2, ensure_ascii=False))

    def register_recipe(self, recipe: Recipe) -> None:
        self.recipes[recipe.recipe_id] = recipe.to_dict()

    def add_to_blacklist(self, pattern_id: str, reason: str) -> None:
        self.blacklist.append({"pattern_id": pattern_id, "reason": reason,
                               "added_at": datetime.now().isoformat()})

    def save_manifest(self, manifest: OperationalManifest) -> Path:
        mdir = Path(self.state_dir) / "manifests"
        mdir.mkdir(parents=True, exist_ok=True)
        f = mdir / f"cycle_{manifest.cycle_id:03d}.json"
        f.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False))
        self.last_manifest = asdict(manifest)
        return f
