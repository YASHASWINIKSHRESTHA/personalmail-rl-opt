"""
PersonalMail-RL: Action and Observation Models
OpenEnv-compatible dataclasses for the email RL environment.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class UrgencyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EmailCategory(str, Enum):
    WORK = "work"
    PERSONAL = "personal"
    SOCIAL = "social"
    SPAM = "spam"


class ToneType(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ASSERTIVE = "assertive"
    APOLOGETIC = "apologetic"
    NONE = "none"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"


# ─────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────

class ClassifyAction(BaseModel):
    """Step 1: Agent classifies the email"""
    action_type: ActionType = ActionType.CLASSIFY
    urgency: UrgencyLevel = Field(..., description="Email urgency level")
    category: EmailCategory = Field(..., description="Email category")
    requires_reply: bool = Field(..., description="Does this email need a reply?")
    reason: str = Field(..., description="Brief reason for classification (1-2 sentences)")


class ReplyAction(BaseModel):
    """Step 2: Agent drafts a reply"""
    action_type: ActionType = ActionType.REPLY
    tone: ToneType = Field(..., description="Tone of the reply")
    reply_text: str = Field(..., description="Full email reply text")


class EmailAction(BaseModel):
    """Union action - can be classify or reply"""
    action_type: ActionType
    # Classify fields
    urgency: Optional[UrgencyLevel] = None
    category: Optional[EmailCategory] = None
    requires_reply: Optional[bool] = None
    reason: Optional[str] = None
    # Reply fields
    tone: Optional[ToneType] = None
    reply_text: Optional[str] = None


# ─────────────────────────────────────────
# OBSERVATIONS
# ─────────────────────────────────────────

class EmailObservation(BaseModel):
    """What the agent sees at each step"""
    episode_id: str
    email_id: str
    subject: str
    body: str
    sender_name: str
    sender_email: str
    current_step: int = Field(description="1=classify, 2=reply")
    max_steps: int = 2
    done: bool = False
    # Filled after step 1
    classification: Optional[Dict[str, Any]] = None
    # Reward info (shown after each step)
    last_step_reward: Optional[float] = None
    total_reward: Optional[float] = None


class StepResult(BaseModel):
    """Result of an environment step"""
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result of environment reset"""
    observation: EmailObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class HealthResult(BaseModel):
    status: str = "ok"
    env_name: str = "personalmail-rl"
    version: str = "1.0.0"
