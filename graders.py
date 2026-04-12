"""Task-specific graders for SupplyChainEnv.

Each grader ingests the full episode history and returns a GradeResult with
a normalised score in [0.0, 1.0] and a human-readable breakdown dict.
"""
from __future__ import annotations

import numpy as np

from environment import SupplyChainEnvironment
from models import GradeResult


def grade_reorder_point(history: list[dict]) -> GradeResult:
    env_ref = SupplyChainEnvironment()
    safety = np.array(env_ref.DAILY_DEMAND_MEAN, dtype=np.float64) * 2.0
    n_skus = env_ref.N_SKUS
    if not history:
        return GradeResult(
            score=0.0,
            breakdown={"pairs_ok": 0, "pairs_total": 0},
            task_id="reorder_point",
        )
    pairs_ok = 0
    pairs_total = len(history) * n_skus
    for rec in history:
        inv = rec["inventory"]
        for i in range(n_skus):
            if float(inv[i]) > float(safety[i]):
                pairs_ok += 1
    score = pairs_ok / float(pairs_total) if pairs_total else 0.0
    score = float(np.clip(score, 0.0, 1.0))
    return GradeResult(
        score=score,
        breakdown={"pairs_ok": pairs_ok, "pairs_total": pairs_total},
        task_id="reorder_point",
    )


def grade_vendor_selection(history: list[dict]) -> GradeResult:
    env_ref = SupplyChainEnvironment()
    max_possible = float(env_ref.INITIAL_BUDGET)
    if not history:
        return GradeResult(
            score=0.0,
            breakdown={"cost_score": 0.0, "quality_score": 0.0},
            task_id="vendor_selection",
        )
    last = history[-1]
    actual_spend = float(last.get("cumulative_spend", 0.0))
    normalized_savings = (max_possible - actual_spend) / max_possible
    cost_score = float(np.clip(normalized_savings, 0.0, 1.0))

    qual_ok = 0
    for rec in history:
        inv = rec["inventory"]
        no_stockout = all(float(x) > 0 for x in inv)
        no_disrupted = "disrupted" not in rec.get("supplier_status", [])
        if no_stockout and no_disrupted:
            qual_ok += 1
    quality_score = qual_ok / float(len(history)) if history else 0.0
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    score = 0.6 * cost_score + 0.4 * quality_score
    score = float(np.clip(score, 0.0, 1.0))
    return GradeResult(
        score=score,
        breakdown={
            "cost_score": cost_score,
            "quality_score": quality_score,
            "actual_spend": actual_spend,
            "max_possible_spend": max_possible,
        },
        task_id="vendor_selection",
    )


def grade_disruption_recovery(history: list[dict]) -> GradeResult:
    if not history:
        return GradeResult(
            score=0.0,
            breakdown={
                "steps_to_recovery": None,
                "stockout_count": 0,
                "rerouted_early": False,
            },
            task_id="disruption_recovery",
        )

    disruption_idx: int | None = None
    for i, rec in enumerate(history):
        if rec.get("disruption_active"):
            disruption_idx = i
            break

    if disruption_idx is None:
        base_score = 0.0
        proactive_bonus = 0.0
        return GradeResult(
            score=float(np.clip(base_score + proactive_bonus, 0.0, 1.0)),
            breakdown={
                "steps_to_recovery": None,
                "stockout_count": 0,
                "rerouted_early": False,
            },
            task_id="disruption_recovery",
        )

    window = history[disruption_idx + 1 : disruption_idx + 1 + 8]
    stockout_count = 0
    for rec in window:
        inv = rec["inventory"]
        stockout_count += sum(1 for x in inv if float(x) <= 0)

    steps_no_stockout = 0
    for rec in window:
        inv = rec["inventory"]
        if all(float(x) > 0 for x in inv):
            steps_no_stockout += 1

    recovered_idx: int | None = None
    for j, rec in enumerate(window):
        inv = rec["inventory"]
        if (not rec.get("disruption_active")) and all(float(x) > 0 for x in inv):
            recovered_idx = j
            break

    if recovered_idx is not None:
        base_score = 1.0
    else:
        base_score = (steps_no_stockout / 8.0) * 0.8

    proactive_bonus = 0.0
    early = history[disruption_idx + 1 : disruption_idx + 3]
    rerouted_early = any(
        bool(r.get("reroute_shipment")) or bool(r.get("switch_supplier"))
        for r in early
    )
    if rerouted_early:
        proactive_bonus = 0.1

    score = float(np.clip(base_score + proactive_bonus, 0.0, 1.0))
    return GradeResult(
        score=score,
        breakdown={
            "steps_to_recovery": recovered_idx + 1 if recovered_idx is not None else None,
            "stockout_count": stockout_count,
            "rerouted_early": rerouted_early,
            "base_score": base_score,
            "proactive_bonus": proactive_bonus,
        },
        task_id="disruption_recovery",
    )


def run_all_graders(history: list[dict], task_id: str) -> GradeResult:
    if task_id == "reorder_point":
        return grade_reorder_point(history)
    if task_id == "vendor_selection":
        return grade_vendor_selection(history)
    if task_id == "disruption_recovery":
        return grade_disruption_recovery(history)
    return GradeResult(
        score=0.0,
        breakdown={"error": f"unknown task_id: {task_id}"},
        task_id=task_id,
    )
