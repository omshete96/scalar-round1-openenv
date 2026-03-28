from pydantic import BaseModel


class SupplyChainAction(BaseModel):
    order_qty: dict[str, int] = {}  # sku_id -> units to order
    switch_supplier: bool = False
    expedite: bool = False
    reroute_shipment: bool = False


class SupplyChainObservation(BaseModel):
    inventory: list[float]  # length 3
    pending_orders: list[float]  # length 3
    demand_forecast: list[list[float]]  # shape 3x7
    supplier_status: list[str]  # length 3
    budget_remaining: float
    day: int
    disruption_active: bool


class StepResult(BaseModel):
    observation: SupplyChainObservation
    reward: float
    done: bool
    info: dict


class GradeResult(BaseModel):
    score: float  # always in [0.0, 1.0]
    breakdown: dict
    task_id: str
