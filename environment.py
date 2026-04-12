from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models import StepResult, SupplyChainAction, SupplyChainObservation


TASK_MAX_STEPS: dict[str, int] = {
    "reorder_point": 10,
    "vendor_selection": 20,
    "disruption_recovery": 30,
}


@dataclass
class SupplyChainEnvironment:
    N_SKUS: int = 3
    SKU_IDS: tuple[str, ...] = ("SKU_A", "SKU_B", "SKU_C")
    INITIAL_INVENTORY: tuple[float, ...] = (100.0, 80.0, 60.0)
    DAILY_DEMAND_MEAN: tuple[float, ...] = (12.0, 8.0, 5.0)
    LEAD_TIME_DAYS: int = 3
    INITIAL_BUDGET: float = 50_000.0
    UNIT_COST: dict[str, float] = field(
        default_factory=lambda: {"SKU_A": 10.0, "SKU_B": 25.0, "SKU_C": 50.0}
    )
    HOLDING_COST_RATE: float = 0.02
    EXPEDITE_MULTIPLIER: float = 2.5
    DISRUPTION_STEP: int = 5

    task_id: str = "reorder_point"
    rng: np.random.Generator | None = None
    inventory: np.ndarray = field(init=False)
    pending_orders: np.ndarray = field(init=False)
    demand_forecast: np.ndarray = field(init=False)
    supplier_status: list[str] = field(default_factory=list)
    budget: float = field(init=False)
    day: int = field(init=False)
    disruption_active: bool = field(init=False)
    step_count: int = field(init=False)
    history: list[dict[str, Any]] = field(default_factory=list)
    using_backup_supplier: bool = field(init=False)

    def _empty_state_arrays(self) -> None:
        self.inventory = np.zeros(self.N_SKUS, dtype=np.float64)
        self.pending_orders = np.zeros(
            (self.N_SKUS, self.LEAD_TIME_DAYS), dtype=np.float64
        )
        self.demand_forecast = np.zeros((self.N_SKUS, 7), dtype=np.float64)

    def reset(self, task_id: str, seed: int = 42) -> SupplyChainObservation:
        self.task_id = task_id
        self.rng = np.random.default_rng(seed)
        self._empty_state_arrays()
        for i in range(self.N_SKUS):
            self.inventory[i] = float(self.INITIAL_INVENTORY[i])
        self.supplier_status = ["normal", "normal", "normal"]
        self.budget = float(self.INITIAL_BUDGET)
        self.day = 0
        self.disruption_active = False
        self.step_count = 0
        self.history = []
        self.using_backup_supplier = False

        self._roll_forecast()
        return self._observation()

    def _roll_forecast(self) -> None:
        assert self.rng is not None
        for s in range(self.N_SKUS):
            mu = float(self.DAILY_DEMAND_MEAN[s])
            self.demand_forecast[s, :] = self.rng.poisson(mu, size=7).astype(
                np.float64
            )

    def _observation(self) -> SupplyChainObservation:
        pending_totals = self.pending_orders.sum(axis=1).astype(float).tolist()
        fc = self.demand_forecast.astype(float).tolist()
        return SupplyChainObservation(
            inventory=self.inventory.astype(float).tolist(),
            pending_orders=pending_totals,
            demand_forecast=fc,
            supplier_status=list(self.supplier_status),
            budget_remaining=float(self.budget),
            day=int(self.day),
            disruption_active=bool(self.disruption_active),
        )

    def step(self, action: SupplyChainAction) -> StepResult:
        if self.rng is None:
            self.rng = np.random.default_rng(42)

        order_qty = dict(action.order_qty or {})
        for k in order_qty:
            if k not in self.SKU_IDS:
                return StepResult(
                    observation=self._observation(),
                    reward=-10.0,
                    done=False,
                    info={"error": f"invalid sku: {k}"},
                )

        max_steps = TASK_MAX_STEPS.get(self.task_id, 10)
        self.using_backup_supplier = bool(action.switch_supplier)

        price_mult = 1.1 if self.using_backup_supplier else 1.0
        projected_spend = 0.0
        for sku, qty in order_qty.items():
            if qty <= 0:
                continue
            unit = float(self.UNIT_COST[sku]) * price_mult
            projected_spend += unit * float(qty)
        if float(self.budget) - projected_spend < 0:
            return StepResult(
                observation=self._observation(),
                reward=-10.0,
                done=False,
                info={"error": "insufficient budget for orders"},
            )

        spend_this_step = 0.0
        for sku, qty in order_qty.items():
            if qty <= 0:
                continue
            idx = self.SKU_IDS.index(sku)
            unit = float(self.UNIT_COST[sku]) * price_mult
            cost = unit * float(qty)
            spend_this_step += cost
            self.pending_orders[idx, 0] += float(qty)

        self.budget -= spend_this_step

        if action.expedite:
            expedite_cost = 0.0
            for idx in range(self.N_SKUS):
                qty_row = float(self.pending_orders[idx].sum())
                if qty_row <= 0:
                    continue
                u = float(self.UNIT_COST[self.SKU_IDS[idx]])
                expedite_cost += self.EXPEDITE_MULTIPLIER * u * qty_row
                self.inventory[idx] += qty_row
            self.pending_orders[:] = 0.0
            self.budget -= expedite_cost

        if action.reroute_shipment and self.disruption_active:
            self.budget -= 500.0
            for i, st in enumerate(self.supplier_status):
                if st == "disrupted":
                    self.supplier_status[i] = "normal"
            if not any(s == "disrupted" for s in self.supplier_status):
                self.disruption_active = False

        demand_today = self.demand_forecast[:, 0].astype(np.float64).copy()

        self.inventory += self.pending_orders[:, 0]
        if self.LEAD_TIME_DAYS > 1:
            self.pending_orders[:, : self.LEAD_TIME_DAYS - 1] = self.pending_orders[
                :, 1:
            ]
        self.pending_orders[:, -1] = 0.0

        inv_before_demand = self.inventory.copy()
        fulfilled = np.minimum(inv_before_demand, demand_today)

        self.inventory = np.maximum(self.inventory - demand_today, 0.0)
        inv_after = self.inventory

        self._roll_forecast()

        self.day += 1
        self.step_count += 1

        if (
            self.task_id == "disruption_recovery"
            and self.day == self.DISRUPTION_STEP
        ):
            # Disruption always fires regardless of supplier choice —
            # this ensures the grading scenario is always exercised.
            self.supplier_status[0] = "disrupted"
            self.disruption_active = True

        stockout_penalty = -1.0 * float(np.sum(inv_after <= 0))
        holding_cost = -self.HOLDING_COST_RATE * float(np.sum(inv_after))
        sla_denom = float(np.sum(demand_today)) + 1e-6
        sla_rate = float(np.sum(fulfilled) / sla_denom)
        sla_bonus = sla_rate * 0.5
        budget_penalty = -2.0 if self.budget < 0 else 0.0
        reward = stockout_penalty + holding_cost + sla_bonus + budget_penalty
        reward = float(np.clip(reward, -10.0, 2.0))

        cumulative_spend = float(self.INITIAL_BUDGET - self.budget)

        record = {
            "step_count": self.step_count,
            "day": self.day,
            "inventory": inv_after.astype(float).tolist(),
            "supplier_status": list(self.supplier_status),
            "disruption_active": bool(self.disruption_active),
            "budget_remaining": float(self.budget),
            "cumulative_spend": cumulative_spend,
            "demand_today": demand_today.astype(float).tolist(),
            "order_qty": dict(order_qty),
            "switch_supplier": bool(action.switch_supplier),
            "expedite": bool(action.expedite),
            "reroute_shipment": bool(action.reroute_shipment),
            "stockouts_after": int(np.sum(inv_after <= 0)),
            "reward": reward,
        }
        self.history.append(record)

        done = self.step_count >= max_steps
        info = {"step_count": self.step_count, "max_steps": max_steps}
        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> dict[str, Any]:
        obs = self._observation()
        tail = self.history[-5:] if self.history else []
        return {
            **obs.model_dump(),
            "step_count": self.step_count,
            "task_id": self.task_id,
            "history": self.history,
            "history_tail": tail,
        }
