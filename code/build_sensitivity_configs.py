"""Regenerate every implementation-sensitivity configuration from the canonical
controller lock.

Each study below starts from the exact canonical policy entry and overrides
only the scenario field under test, including predictor-bank paths,
stage/allocation fields, and explicit filter flags. A scenario row that equals
the lock therefore reproduces the canonical replay exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

POLICY_LABELS = {
    "PID-g1.20-du0.60": "PID",
    "ADRC-g1.20-du0.95": "OC-PID",
    "C2-g1.20-du0.80": "C2",
    "C7-Core-g0.75-mpc0.14": "C7-Core",
}
SENS_DIR = ROOT / "config" / "implementation_sensitivity"


def locked_entries() -> dict[str, dict]:
    data = json.loads((ROOT / "config" / "selected_controller_configs.json").read_text(encoding="utf-8"))
    return {entry["label"]: entry for entry in data}


def entry(base: dict, label: str, scenario_id: str, block: str, **overrides) -> dict:
    item = copy.deepcopy(base)
    item["block"] = block
    item["label"] = label
    item["scenario_id"] = scenario_id
    item["seed_group"] = scenario_id
    item["comparison_base"] = "PID"
    for key, value in overrides.items():
        item[key] = value
    return item


def write(name: str, configs: list[dict]) -> None:
    path = SENS_DIR / name
    path.write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path} ({len(configs)} configs)")


def main() -> int:
    locked = locked_entries()

    def per_policy(short: str) -> dict:
        return locked[short]

    # Emulator response envelope: 6 scenarios x 4 policies.
    emulator = []
    scenario_overrides = {
        "nominal": {"audit_energy": True},
        "low_response_corner": {"command_gain_scale": 0.5, "cross_coupling_scale": 0.0, "audit_energy": True},
        "high_response_corner": {"command_gain_scale": 1.5, "cross_coupling_scale": 1.5, "audit_energy": True},
        "diagonal_response": {"command_gain_scale": 1.0, "cross_coupling_scale": 0.0, "audit_energy": True},
        "alternative_linear": {"response_variant": "linear", "command_gain_scale": 0.85, "cross_coupling_scale": 0.5, "audit_energy": True},
        "alternative_weakened": {"response_variant": "weakened", "command_gain_scale": 1.0, "cross_coupling_scale": 1.25, "audit_energy": True},
    }
    for scenario, overrides in scenario_overrides.items():
        for key, short in POLICY_LABELS.items():
            emulator.append(entry(locked[key], short, scenario, "emulator_compact", **overrides))
    write("emulator_compact_configs.json", emulator)

    # Amplitude envelope: 4 bounds x 4 policies.
    amplitude = []
    for bound in [0.45, 0.55, 0.65, 0.75]:
        scenario = f"amplitude_{bound:.2f}"
        for key, short in POLICY_LABELS.items():
            amplitude.append(entry(locked[key], short, scenario, "amplitude_envelope", amplitude_bound=bound))
    write("amplitude_envelope_configs.json", amplitude)

    # Anti-windup: 2 modes x 4 policies.
    anti = []
    for mode in ["vectorwise", "per_channel"]:
        scenario = f"anti_windup_{mode}"
        for key, short in POLICY_LABELS.items():
            anti.append(entry(locked[key], short, scenario, "anti_windup", anti_windup_mode=mode))
    write("anti_windup_configs.json", anti)

    # Numerical state guardrail: tighter 6, canonical 8 and looser 12 x 4 policies.
    clip = []
    for scale in [6.0, 8.0, 12.0]:
        scenario = f"clip_{int(scale)}"
        for key, short in POLICY_LABELS.items():
            clip.append(entry(locked[key], short, scenario, "output_clip", output_clip_scale=scale))
    write("output_clip_configs.json", clip)

    no_clip = []
    for key, short in POLICY_LABELS.items():
        no_clip.append(entry(locked[key], short, "no_output_clip", "output_no_clip", output_clip_scale=float("inf")))
    write("output_no_clip_configs.json", no_clip)

    # Relative outer-rate sensitivity: multipliers of each locked gamma.
    locked_gamma = {"PID-g1.20-du0.60": 0.60, "ADRC-g1.20-du0.95": 0.95,
                    "C2-g1.20-du0.80": 0.80, "C7-Core-g0.75-mpc0.14": 0.80}
    rate = []
    for multiplier in [0.75, 1.00, 1.25, 1.50]:
        scenario = f"shell_rate_{multiplier:.2f}"
        for key, short in POLICY_LABELS.items():
            rate.append(entry(
                locked[key],
                f"{short}|rate{multiplier:.2f}",
                scenario,
                "rate_sensitivity",
                outer_du_scale=locked_gamma[key] * multiplier,
            ))
    write("rate_sensitivity_configs.json", rate)

    # Strict single-factor C7 component ablations.  NoMPC changes only the
    # MPC share; NoBlend changes only the fixed 0.10/0.90 estimator blend.
    c7 = locked["C7-Core-g0.75-mpc0.14"]
    architecture = [
        entry(c7, "C7-Core", "architecture_ablation", "architecture_ablation"),
        entry(c7, "C7-NoMPC", "architecture_ablation", "architecture_ablation", mpc_share=0.0),
        entry(
            c7, "C7-NoBlend", "architecture_ablation", "architecture_ablation",
            gate_enabled=False, no_gate_release_gain=False,
        ),
    ]
    for item in architecture:
        item["comparison_base"] = "C7-Core"
    write("architecture_ablation_configs.json", architecture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
