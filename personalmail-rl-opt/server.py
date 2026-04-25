"""
PersonalMail-RL: FastAPI Server (OpenEnv-Compatible)
Exposes the environment via HTTP endpoints matching the OpenEnv spec.

Endpoints:
  GET  /health          — health check
  GET  /info            — environment metadata
  POST /reset           — start new episode
  POST /step            — take an action
  GET  /state           — current state (for monitoring)
  POST /rollout         — full single-episode rollout (for TRL integration)
"""

import json
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import PersonalMailEnv
from env.models import HealthResult

app = FastAPI(
    title="PersonalMail-RL Environment",
    description=(
        "OpenEnv-compatible RL environment for training LLMs to handle personal emails. "
        "Theme: #3.2 Personalized Tasks — OpenEnv Hackathon 2026."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (per session_id) ───────────────────────────────
# In production, use Redis or a DB. For hackathon, in-memory is fine.
_sessions: Dict[str, PersonalMailEnv] = {}


def get_or_create_session(session_id: Optional[str]) -> tuple:
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    if session_id not in _sessions:
        _sessions[session_id] = PersonalMailEnv()
    return session_id, _sessions[session_id]


# ─── Request / Response schemas ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    scenario_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class RolloutRequest(BaseModel):
    """Full episode rollout — used by TRL training loop"""
    session_id: Optional[str] = None
    scenario_id: Optional[str] = None
    step1_action: Dict[str, Any]  # classification action
    step2_action: Dict[str, Any]  # reply action


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResult)
def health():
    return HealthResult()


@app.get("/info")
def info():
    return {
        "name": "personalmail-rl",
        "version": "1.0.0",
        "theme": "Personalized Tasks (#3.2)",
        "description": "RL environment for training LLMs to handle personal emails",
        "observation_space": {
            "type": "dict",
            "fields": ["episode_id", "email_id", "subject", "body", "sender_name",
                       "sender_email", "current_step", "max_steps", "done",
                       "classification", "instruction"],
        },
        "action_space": {
            "step_1": {
                "type": "classify",
                "fields": ["urgency", "category", "requires_reply", "reason"],
            },
            "step_2": {
                "type": "reply",
                "fields": ["tone", "reply_text"],
            },
        },
        "reward_range": [0.0, 1.0],
        "max_steps_per_episode": 2,
        "num_scenarios": 25,
        "scenario_difficulties": {"easy": 11, "medium": 8, "hard": 6},
        "reward_functions": [
            "classification_accuracy",
            "classification_format",
            "reply_format",
            "reply_relevance",
            "reply_length",
            "tone_matching",
            "reply_clarity",
        ],
        "stack": "OpenEnv + TRL (GRPO) + Unsloth",
    }


@app.get("/scenarios")
def list_scenarios(difficulty: Optional[str] = None):
    """List available email scenarios (without ground truth labels). Filter by difficulty."""
    from env.scenarios import SCENARIOS, get_scenarios_by_difficulty
    pool = get_scenarios_by_difficulty(difficulty) if difficulty else SCENARIOS
    return [
        {
            "id": s["id"],
            "difficulty": s.get("difficulty", "medium"),
            "subject": s["subject"],
            "sender_name": s["sender_name"],
        }
        for s in pool
    ]


@app.post("/reset")
def reset(req: ResetRequest):
    """Start a new episode. Returns initial observation + session_id."""
    session_id, env = get_or_create_session(req.session_id)
    obs = env.reset(scenario_id=req.scenario_id)
    prompt = env.build_llm_prompt(obs)
    return {
        "session_id": session_id,
        "observation": obs,
        "llm_prompt": json.loads(prompt),
        "info": {"message": "Episode started. Ready for step 1 (classify)."},
    }


@app.post("/step")
def step(req: StepRequest):
    """Take one action in the environment."""
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found. Call /reset first.")

    env = _sessions[req.session_id]

    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Flatten reward breakdown for UI convenience while preserving full details.
    ui_info = dict(info) if isinstance(info, dict) else {}
    breakdown = ui_info.get("breakdown", {}) if isinstance(ui_info, dict) else {}
    if isinstance(breakdown, dict):
        for key, payload in breakdown.items():
            if isinstance(payload, dict) and "score" in payload:
                ui_info[key] = payload.get("score", 0)

    prompt = env.build_llm_prompt(obs) if not done else None

    return {
        "session_id": req.session_id,
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": ui_info,
        "llm_prompt": json.loads(prompt) if prompt else None,
        "state": env.state(),
    }


@app.get("/state")
def state(session_id: str):
    """Get current state for monitoring/debugging."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    env = _sessions[session_id]
    return env.state()


@app.post("/rollout")
def rollout(req: RolloutRequest):
    """
    Full episode rollout for TRL training integration.
    Runs both steps and returns the complete episode with all rewards.
    """
    session_id, env = get_or_create_session(req.session_id)
    obs = env.reset(scenario_id=req.scenario_id)

    # Step 1: classify
    obs1, r1, done1, info1 = env.step(req.step1_action)

    if done1:
        return {
            "session_id": session_id,
            "episode_total_reward": r1,
            "step_rewards": [r1],
            "done": True,
            "error": "Episode ended after step 1 (unexpected)",
        }

    # Step 2: reply
    obs2, r2, done2, info2 = env.step(req.step2_action)

    return {
        "session_id": session_id,
        "episode_total_reward": env.total_reward,
        "step1_reward": r1,
        "step2_reward": r2,
        "step1_info": info1,
        "step2_info": info2,
        "final_state": env.state(),
        "done": done2,
    }


@app.get("/")
def root():
    """Serve the interactive demo UI."""
    import os
    from fastapi.responses import FileResponse
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {
        "message": "PersonalMail-RL Environment Server",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7863)


@app.post("/demo/compare")
def demo_compare(req: dict):
    """
    Demo endpoint: runs same scenario twice (baseline vs trained) and returns
    REAL computed reward scores for both — used by the interactive UI.
    """
    from env.rewards import compute_step1_reward, compute_step2_reward, compute_episode_reward
    from env.scenarios import SCENARIOS
    import random

    scenario_id = req.get("scenario_id")
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if not scenario:
        scenario = random.choice(SCENARIOS)

    gt = scenario["ground_truth"]
    baseline_classify = req.get("baseline_classify", {})
    baseline_reply    = req.get("baseline_reply", "")
    trained_classify  = req.get("trained_classify", {})
    trained_reply     = req.get("trained_reply", "")

    # Compute REAL rewards using the actual reward functions
    b_s1 = compute_step1_reward(baseline_classify, gt)
    b_s2 = compute_step2_reward(baseline_reply, gt)
    t_s1 = compute_step1_reward(trained_classify, gt)
    t_s2 = compute_step2_reward(trained_reply, gt)

    b_episode = compute_episode_reward(b_s1, b_s2)
    t_episode = compute_episode_reward(t_s1, t_s2)

    return {
        "scenario": {
            "id": scenario["id"],
            "subject": scenario["subject"],
            "body": scenario["body"],
            "sender_name": scenario["sender_name"],
            "difficulty": scenario.get("difficulty", "medium"),
        },
        "baseline": {
            "classify": baseline_classify,
            "reply": baseline_reply,
            "classification_reward": b_s1["total"],
            "classification_accuracy": b_s1["breakdown"]["classification_accuracy"]["score"],
            "classification_format": b_s1["breakdown"]["classification_format"]["score"],
            "reply_reward": b_s2["total"],
            "reply_format": b_s2["breakdown"]["reply_format"]["score"],
            "reply_relevance": b_s2["breakdown"]["reply_relevance"]["score"],
            "reply_length": b_s2["breakdown"]["reply_length"]["score"],
            "tone_matching": b_s2["breakdown"].get("tone_matching", {}).get("score", 0),
            "reply_clarity": b_s2["breakdown"].get("reply_clarity", {}).get("score", 0),
            "episode_reward": b_episode,
        },
        "trained": {
            "classify": trained_classify,
            "reply": trained_reply,
            "classification_reward": t_s1["total"],
            "classification_accuracy": t_s1["breakdown"]["classification_accuracy"]["score"],
            "classification_format": t_s1["breakdown"]["classification_format"]["score"],
            "reply_reward": t_s2["total"],
            "reply_format": t_s2["breakdown"]["reply_format"]["score"],
            "reply_relevance": t_s2["breakdown"]["reply_relevance"]["score"],
            "reply_length": t_s2["breakdown"]["reply_length"]["score"],
            "tone_matching": t_s2["breakdown"].get("tone_matching", {}).get("score", 0),
            "reply_clarity": t_s2["breakdown"].get("reply_clarity", {}).get("score", 0),
            "episode_reward": t_episode,
        },
        "improvement": round(t_episode - b_episode, 3),
    }
