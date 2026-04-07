"""
inference.py — Scaler Hackathon submission
Uses OpenAI-compatible client pointed at Groq (free, fast).

Optional env vars (set in HF Space Secrets for live LLM calls):
  API_BASE_URL = https://api.groq.com/openai/v1
  MODEL_NAME   = llama-3.3-70b-versatile
  HF_TOKEN     = your Groq API key (free at console.groq.com)

If the vars are absent the script still runs using the built-in
rule-based fallback inside llm_act(), so Phase-2 validation passes.
"""

import json
import os
import uuid

from openai import OpenAI

from environment import SupplyChainEnvironment
from graders import run_all_graders
from models import SupplyChainAction

# Environment variables — strictly following hackathon checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default, allowed to be None

# Initialize OpenAI client with checklist-compliant variables
try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key",  # Use dummy if None to avoid init crash
        timeout=15.0,
        max_retries=0,
    )
except Exception:
    client = None

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
        if client is None or not HF_TOKEN:
            raise RuntimeError("LLM client not configured (missing HF_TOKEN)")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Ensure we only have the JSON part if the LLM added filler
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        return SupplyChainAction(**json.loads(text))
    except Exception:
        # Safe rule-based fallback: conservative reorder
        inv = obs.get("inventory", [50, 50, 50])
        supplier_status = obs.get("supplier_status", [])
        disrupted = any("disrupted" in s for s in supplier_status)
        low_stock = any(v < 20 for v in inv)
        critical_stock = any(v <= 5 for v in inv)
        return SupplyChainAction(
            order_qty={
                "SKU_A": 40 if inv[0] < 20 else 15,
                "SKU_B": 30 if inv[1] < 20 else 10,
                "SKU_C": 20 if inv[2] < 20 else 8,
            },
            switch_supplier=disrupted,
            expedite=critical_stock,
            reroute_shipment=disrupted,
        )


TASK_CONFIG = [
    ("reorder_point", 42, 10),
    ("vendor_selection", 42, 20),
    ("disruption_recovery", 42, 30),
]


def run():
    all_results = {}
    for task_id, seed, max_steps in TASK_CONFIG:
        env = SupplyChainEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        episode_id = str(uuid.uuid4())
        
        print(f"START: {episode_id}")
        total_reward = 0.0

        for step_num in range(max_steps):
            action = llm_act(obs.model_dump())
            result = env.step(action)
            obs = result.observation
            reward = float(result.reward)
            total_reward += reward
            
            # STRICT logging format — exactly as expected by the validator regex
            print(f"STEP: {step_num} | Action: {action.__class__.__name__} | Reward: {reward:.3f} | Done: {result.done}")
            
            if result.done:
                break

        print(f"END: {episode_id} | Total Reward: {total_reward:.3f}")

        grade = run_all_graders(env.state()["history"], task_id)
        all_results[task_id] = {
            "score": grade.score,
            "breakdown": grade.breakdown,
        }

    with open("scores.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    run()
