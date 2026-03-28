"""
inference.py — Scaler Hackathon submission
Uses OpenAI-compatible client pointed at Groq (free, fast).

Set these env vars (in HF Space Secrets):
  API_BASE_URL = https://api.groq.com/openai/v1
  MODEL_NAME   = llama-3.3-70b-versatile
  HF_TOKEN     = your Groq API key (free at console.groq.com)
"""

import json
import os

from openai import OpenAI

from environment import SupplyChainEnvironment
from graders import run_all_graders
from models import SupplyChainAction

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["HF_TOKEN"],
    timeout=15.0,
    max_retries=0,
)
MODEL = os.environ["MODEL_NAME"]

SYSTEM_PROMPT = """You are a supply chain operations agent.
You receive a JSON observation and must respond with ONLY a valid JSON action object.
No explanation. No markdown. Just raw JSON.

Action schema:
{
  "order_qty": {"SKU_A": <int>, "SKU_B": <int>, "SKU_C": <int>},
  "switch_supplier": <bool>,
  "expedite": <bool>,
  "reroute_shipment": <bool>
}

Rules:
- If any inventory < 20, order aggressively (2-3x daily demand)
- If supplier_status has "disrupted", set reroute_shipment: true immediately
- If budget_remaining < 1000, do not order
- expedite only if stockout is imminent (inventory <= 5)
"""


def llm_act(obs: dict) -> SupplyChainAction:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        return SupplyChainAction(**json.loads(text))
    except Exception as e:
        print(f"  [LLM fallback] {e}")
        # Safe fallback: conservative reorder
        return SupplyChainAction(
            order_qty={"SKU_A": 20, "SKU_B": 15, "SKU_C": 10},
            switch_supplier=False,
            expedite=False,
            reroute_shipment=False,
        )


TASK_CONFIG = [
    ("reorder_point", 42, 10),
    ("vendor_selection", 42, 20),
    ("disruption_recovery", 42, 30),
]


def run():
    all_results = {}
    for task_id, seed, max_steps in TASK_CONFIG:
        print(f"\n--- Task: {task_id} ---")
        env = SupplyChainEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        done = False

        for step_num in range(max_steps):
            action = llm_act(obs.model_dump())
            result = env.step(action)
            obs = result.observation
            print(
                f"  step {step_num + 1:02d} | reward={result.reward:+.3f} | "
                f"inv={[round(x) for x in obs.inventory]} | done={result.done}"
            )
            if result.done:
                break

        grade = run_all_graders(env.state()["history"], task_id)
        all_results[task_id] = {
            "score": grade.score,
            "breakdown": grade.breakdown,
        }
        print(f"  SCORE: {grade.score:.3f}")

    with open("scores.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nOK Scores saved to scores.json")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    run()
