---
title: Supply Chain Simulation
emoji: 🚛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
short_description: Supply chain procurement and logistics simulation
---
# SupplyChainEnv

## What this environment simulates

This environment simulates day-to-day procurement and logistics: an agent observes inventories, in-transit orders, demand forecasts, supplier health, and budget, then places orders and executes disruption playbooks. Real-world motivation includes **Red Sea shipping delays** (route disruption, rerouting, and expedites) and **semiconductor supply shocks**, where a single supplier or port can force multi-echelon rebalancing under time and budget pressure.

## Observation space

| Field | Type | Description |
| --- | --- | --- |
| `inventory` | `list[float]` | On-hand units for each of three SKUs (length 3). |
| `pending_orders` | `list[float]` | Total in-transit units per SKU (summed over lead-time buckets). |
| `demand_forecast` | `list[list[float]]` | Seven-day demand forecast per SKU (shape 3×7). |
| `supplier_status` | `list[str]` | Lane status per supplier index: `normal`, `delayed`, or `disrupted`. |
| `budget_remaining` | `float` | Remaining procurement budget after orders, fees, and expedites. |
| `day` | `int` | Simulation day counter (advances once per `step`). |
| `disruption_active` | `bool` | Whether an active disruption is flagging the network (e.g., port strike). |

## Action space

| Field | Type | Description |
| --- | --- | --- |
| `order_qty` | `dict[str, int]` | Units to order keyed by `SKU_A` / `SKU_B` / `SKU_C`. |
| `switch_supplier` | `bool` | Use backup supplier this step (unit cost **+10%**, disruption bypass on trigger). |
| `expedite` | `bool` | Deliver all pending immediately at **2.5×** unit cost. |
| `reroute_shipment` | `bool` | When disrupted, pay flat **$500** and clear `disrupted` statuses. |

## Reward function

Let \(I_a\) be on-hand inventory **after** serving today’s demand, \(d\) today’s demand vector, and \(I_b\) on-hand **before** demand is subtracted (after deliveries). Define element-wise fulfilled flow \(\phi = \min(I_b, d)\).

\[
\text{reward} = \text{clip}\Bigl(
-|\{i : I_{a,i} \le 0\}|
- 0.02 \sum_i I_{a,i}
+ 0.5 \frac{\sum_i \phi_i}{\sum_i d_i + 10^{-6}}
+ \mathbb{1}[\text{budget} < 0]\cdot(-2)
,\,-10,\,2\Bigr)
\]

- **Stockout term:** \(-1\) per SKU at or below zero after demand.
- **Holding term:** \(-0.02\) times total remaining inventory (carrying cost).
- **SLA bonus:** up to \(+0.5\) from the fraction of today’s demand covered from on-hand after receipt.
- **Budget breach:** \(-2\) if budget goes negative (orders, expedites, reroute, etc.).

## Tasks

| Task ID | Difficulty | Max Steps | Description |
| --- | --- | --- | --- |
| `reorder_point` | easy | 10 | Keep all SKUs **above** safety stock \(2 \times\) mean daily demand every step. |
| `vendor_selection` | medium | 20 | Graded on spend efficiency vs budget and step-wise service / no-`disrupted` quality. |
| `disruption_recovery` | hard | 30 | Port strike on **day 5**: primary lane disrupted; recover within the grading window. |

## Baseline scores (seed=42)

| Task | Score |
| --- | --- |
| reorder_point | ~0.78 |
| vendor_selection | ~0.54 |
| disruption_recovery | ~0.31 |

## Setup

```bash
pip install -e .
uvicorn server.app:app --port 7860
```

## Environment variables

| Variable | Value |
| --- | --- |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | your Groq API key |

## Running inference

```bash
API_BASE_URL=https://api.groq.com/openai/v1 \
MODEL_NAME=llama-3.3-70b-versatile \
HF_TOKEN=<your-groq-key> \
python inference.py
```
