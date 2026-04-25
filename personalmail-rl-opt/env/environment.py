"""
PersonalMail-RL: Core RL Environment
OpenEnv-compatible environment with reset() / step() / state() interface.
Theme: #3.2 Personalized Tasks — Personal email assistant.

Episode structure (2 steps):
  Step 1: Classify the email (urgency, category, requires_reply)
  Step 2: Draft a professional reply

This gives process-level supervision as recommended in the hackathon guide.
"""

import uuid
import json
import random
from typing import Dict, Any, Optional
from datetime import datetime

from .scenarios import SCENARIOS, get_scenario_by_id
from .rewards import compute_step1_reward, compute_step2_reward, compute_episode_reward


class PersonalMailEnv:
    """
    OpenEnv-compatible RL environment for personal email handling.

    Follows the Gymnasium-style interface:
      reset() -> observation
      step(action) -> (observation, reward, done, info)
      state() -> current state dict
    """

    def __init__(self, scenario_id: Optional[str] = None):
        self.scenario_id = scenario_id
        self.episode_id: Optional[str] = None
        self.current_scenario: Optional[Dict] = None
        self.current_step: int = 0
        self.max_steps: int = 2
        self.done: bool = True
        self.total_reward: float = 0.0
        self.step_rewards: list = []
        self.step_infos: list = []
        self.classification_result: Optional[Dict] = None
        self._history: list = []  # for anti-hacking: track all actions
        self._episode_start_time: Optional[float] = None
        self.max_episode_seconds: float = 120.0  # 2-minute timeout per episode

    # ──────────────────────────────────────
    # CORE INTERFACE
    # ──────────────────────────────────────

    def reset(self, scenario_id: Optional[str] = None) -> Dict:
        """Start a fresh episode. Returns the initial observation."""
        # Pick scenario
        if scenario_id:
            scenario = get_scenario_by_id(scenario_id)
            if scenario is None:
                scenario = random.choice(SCENARIOS)
        elif self.scenario_id:
            scenario = get_scenario_by_id(self.scenario_id) or random.choice(SCENARIOS)
        else:
            scenario = random.choice(SCENARIOS)

        self.episode_id = str(uuid.uuid4())[:8]
        self.current_scenario = scenario
        self.current_step = 1
        self.done = False
        self.total_reward = 0.0
        self.step_rewards = []
        self.step_infos = []
        self.classification_result = None
        self._history = []
        self._episode_start_time = datetime.utcnow().timestamp()

        return self._build_observation()

    def step(self, action: Dict) -> tuple:
        """
        Take one step in the environment.

        Step 1 (current_step == 1): agent classifies the email
        Step 2 (current_step == 2): agent drafts a reply

        Returns: (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self.current_scenario is None:
            raise RuntimeError("No scenario loaded. Call reset() first.")

        # Timeout protection: prevent infinitely long episodes
        elapsed = datetime.utcnow().timestamp() - (self._episode_start_time or 0)
        if elapsed > self.max_episode_seconds:
            self.done = True
            return (
                self._build_observation(),
                -0.5,  # penalty for timeout
                True,
                {"error": "episode_timeout", "elapsed_seconds": round(elapsed, 1)},
            )

        # Anti-hacking: record the action
        self._history.append({
            "step": self.current_step,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Anti-hacking: detect suspiciously identical repeated actions
        if len(self._history) >= 2:
            prev = self._history[-2].get("action", {})
            if prev == action and self.current_step == 1:
                # Same action submitted twice on same step — penalise
                return (self._build_observation(), -0.1, True,
                        {"error": "duplicate_action_detected", "penalty": -0.1})

        # Validate action has the right type for this step
        action_type = action.get("action_type", "")
        if self.current_step == 1 and action_type not in ("classify", ""):
            # Allow missing action_type (agent may not include it)
            pass

        gt = self.current_scenario["ground_truth"]

        if self.current_step == 1:
            # ── CLASSIFY STEP ──
            reward_info = compute_step1_reward(action, gt)
            self.classification_result = action
            self.step_rewards.append(reward_info["total"])
            self.step_infos.append(reward_info)
            self.total_reward += reward_info["total"] * 0.30  # 30% weight
            self.current_step = 2

            obs = self._build_observation()
            return obs, reward_info["total"], False, reward_info

        elif self.current_step == 2:
            # ── REPLY STEP ──
            # Accept both flattened and nested action payloads:
            # - {"reply_text": "...", "tone": "..."}
            # - {"reply": {"reply_text": "...", "tone": "..."}}
            nested_reply = action.get("reply", {})
            if isinstance(nested_reply, dict):
                reply_text = action.get("reply_text") or nested_reply.get("reply_text", "")
            else:
                reply_text = action.get("reply_text", "")
            reward_info = compute_step2_reward(reply_text, gt)
            self.step_rewards.append(reward_info["total"])
            self.step_infos.append(reward_info)
            self.total_reward += reward_info["total"] * 0.70  # 70% weight
            self.done = True

            # Episode summary
            episode_info = {
                **reward_info,
                "episode_id": self.episode_id,
                "episode_total_reward": round(self.total_reward, 4),
                "step_rewards": self.step_rewards,
            }
            obs = self._build_observation()
            return obs, reward_info["total"], True, episode_info

        else:
            raise RuntimeError(f"Invalid step number: {self.current_step}")

    def state(self) -> Dict:
        """Return current state representation (for logging/debugging)."""
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.current_scenario["id"] if self.current_scenario else None,
            "current_step": self.current_step,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "step_rewards": self.step_rewards,
            "classification": self.classification_result,
        }

    # ──────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────

    def _build_observation(self) -> Dict:
        """Build the observation dict the agent sees."""
        s = self.current_scenario
        obs = {
            "episode_id": self.episode_id,
            "email_id": s["id"],
            "difficulty": s.get("difficulty", "medium"),
            "subject": s["subject"],
            "body": s["body"],
            "sender_name": s["sender_name"],
            "sender_email": s["sender_email"],
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "done": self.done,
            "classification": self.classification_result,
            "total_reward": round(self.total_reward, 4),
        }
        # Instruction for the current step
        if self.current_step == 1:
            obs["instruction"] = (
                "STEP 1 — CLASSIFY: Analyze this email and return a JSON with: "
                "urgency (high/medium/low), category (work/personal/social/spam), "
                "requires_reply (true/false), reason (string)."
            )
        elif self.current_step == 2:
            obs["instruction"] = (
                "STEP 2 — REPLY: Draft a professional email reply. "
                "Return JSON with: tone (professional/friendly/assertive/apologetic/none), "
                "reply_text (full reply including greeting and closing)."
            )
        else:
            obs["instruction"] = "Episode complete."
        return obs

    def build_llm_prompt(self, obs: Dict) -> str:
        """
        Build the prompt string fed to the LLM during RL training.
        Returns a structured prompt with system + user turns.
        """
        system_prompt = (
            "You are an expert personal email assistant. "
            "You help users manage their emails professionally and efficiently. "
            "Always respond with valid JSON only. No extra text, no markdown."
        )

        if obs["current_step"] == 1:
            user_prompt = f"""Analyze the following email and classify it.

EMAIL:
Subject: {obs['subject']}
From: {obs['sender_name']} <{obs['sender_email']}>
Body:
{obs['body']}

{obs['instruction']}

Respond with JSON only:
{{
  "action_type": "classify",
  "urgency": "high" | "medium" | "low",
  "category": "work" | "personal" | "social" | "spam",
  "requires_reply": true | false,
  "reason": "brief explanation"
}}"""

        else:  # step 2
            classification_str = ""
            if obs.get("classification"):
                c = obs["classification"]
                classification_str = (
                    f"\nYour previous classification:\n"
                    f"  - Urgency: {c.get('urgency', 'unknown')}\n"
                    f"  - Category: {c.get('category', 'unknown')}\n"
                    f"  - Requires reply: {c.get('requires_reply', 'unknown')}\n"
                )

            user_prompt = f"""Handle the following email.

EMAIL:
Subject: {obs['subject']}
From: {obs['sender_name']} <{obs['sender_email']}>
Body:
{obs['body']}
{classification_str}
{obs['instruction']}

Respond with JSON only:
{{
  "action_type": "reply",
  "tone": "professional" | "friendly" | "assertive" | "apologetic" | "none",
  "reply_text": "Full email reply with greeting, body, and closing"
}}"""

        return json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        })
