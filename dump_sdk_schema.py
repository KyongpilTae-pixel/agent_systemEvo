"""8 agent의 tools() 메타데이터를 SDK-호환 JSON으로 dump.

Claude Agent SDK가 도구를 자동 register할 수 있는 형식:
[
  {
    "name": "<AgentName>__<method>",
    "description": "...",
    "input_schema": {"type": "object", "properties": {...}}
  },
  ...
]

CLI:
  PYTHONPATH=. python -m agent_system.dump_sdk_schema --output agent_system/sdk_tools.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_system.agents.training_data_optimizer import TrainingDataOptimizer
from agent_system.agents.model_curator import ModelCurator
from agent_system.agents.ensemble_designer import EnsembleDesigner
from agent_system.agents.calibration_specialist import CalibrationSpecialist
from agent_system.agents.sir_evaluator import SIRClinicalEvaluator
from agent_system.agents.overfit_sentinel import OverfitSentinel
from agent_system.agents.data_quality_auditor import DataQualityAuditor
from agent_system.agents.routing_coordinator import RoutingCoordinator


ALL_AGENTS = [
    TrainingDataOptimizer, ModelCurator, EnsembleDesigner, CalibrationSpecialist,
    SIRClinicalEvaluator, OverfitSentinel, DataQualityAuditor, RoutingCoordinator,
]

TYPE_MAP = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "list": "array", "dict": "object", "Path": "string",
    "Any": "string", "SharedState": "object",
}


def _to_json_schema(param_type: str) -> dict:
    """Python type annotation string → JSON schema fragment."""
    return {"type": TYPE_MAP.get(param_type, "string")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=Path("agent_system/sdk_tools.json"), type=Path)
    args = p.parse_args()

    tools: list[dict] = []
    agents_info: list[dict] = []
    for cls in ALL_AGENTS:
        agent = cls()
        desc = agent.describe()
        agent_info = {
            "code": desc.code, "name": desc.name, "role": desc.role,
            "n_tools": len(desc.tools),
        }
        agents_info.append(agent_info)
        for tool_desc in desc.tools:
            tools.append({
                "name": f"{desc.name}__{tool_desc.name}",
                "agent_code": desc.code,
                "method": tool_desc.name,
                "description": tool_desc.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        pname: _to_json_schema(ptype)
                        for pname, ptype in tool_desc.parameters.items()
                    },
                    "required": [k for k in tool_desc.parameters.keys()],
                },
            })

    out = {"meta": {"n_agents": len(ALL_AGENTS), "n_tools": len(tools),
                    "generated_at": __import__("datetime").datetime.now()
                                       .isoformat(timespec="seconds")},
           "agents": agents_info, "tools": tools}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"saved -> {args.output}")
    print(f"  agents: {len(ALL_AGENTS)}")
    print(f"  tools : {len(tools)}")
    print()
    print("Agent별 tool 개수:")
    for a in agents_info:
        print(f"  [{a['code']}] {a['name']:25s} : {a['n_tools']} tools")


if __name__ == "__main__":
    main()
