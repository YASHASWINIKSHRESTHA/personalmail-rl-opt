"""
PersonalMail-RL: Reward Functions
5 independent reward functions as recommended by the hackathon guide.
Each returns a value in [0.0, 1.0]. Combined into a weighted total.
"""

import re
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────
# STEP 1 REWARDS: Classification
# ─────────────────────────────────────────

def classification_accuracy_reward(action: Dict, ground_truth: Dict) -> Tuple[float, str]:
    """
    Reward 1: How accurately did the agent classify the email?
    Checks urgency, category, and requires_reply.
    """
    score = 0.0
    breakdown = []

    urgency_match = action.get("urgency") == ground_truth.get("urgency")
    category_match = action.get("category") == ground_truth.get("category")
    reply_match = action.get("requires_reply") == ground_truth.get("requires_reply")

    if urgency_match:
        score += 0.34
        breakdown.append("✅ urgency correct")
    else:
        breakdown.append(f"❌ urgency: got '{action.get('urgency')}', expected '{ground_truth.get('urgency')}'")

    if category_match:
        score += 0.33
        breakdown.append("✅ category correct")
    else:
        breakdown.append(f"❌ category: got '{action.get('category')}', expected '{ground_truth.get('category')}'")

    if reply_match:
        score += 0.33
        breakdown.append("✅ requires_reply correct")
    else:
        breakdown.append(f"❌ requires_reply: got {action.get('requires_reply')}, expected {ground_truth.get('requires_reply')}")

    return round(score, 3), " | ".join(breakdown)


def classification_format_reward(action: Dict) -> Tuple[float, str]:
    """
    Reward 2: Did the agent return a well-formed classification JSON?
    Required fields: urgency, category, requires_reply, reason.
    """
    required_fields = ["urgency", "category", "requires_reply", "reason"]
    present = [f for f in required_fields if action.get(f) is not None]
    score = len(present) / len(required_fields)

    has_reason = bool(action.get("reason", "").strip())
    reason_length = len(action.get("reason", "").split())
    if has_reason and reason_length >= 5:
        score = min(1.0, score + 0.1)

    note = f"{len(present)}/{len(required_fields)} required fields present"
    return round(score, 3), note


# ─────────────────────────────────────────
# STEP 2 REWARDS: Reply Quality
# ─────────────────────────────────────────

def reply_format_reward(reply_text: str) -> Tuple[float, str]:
    """
    Reward 3: Does the reply have proper email format?
    Checks: greeting, body paragraphs, closing sign-off.
    """
    if not reply_text or not reply_text.strip():
        return 0.0, "❌ empty reply"

    score = 0.0
    breakdown = []

    # Check greeting (Dear/Hi/Hello/Hey at start)
    has_greeting = bool(re.search(
        r'^(Dear|Hi|Hello|Hey|Good\s+\w+)',
        reply_text.strip(),
        re.IGNORECASE | re.MULTILINE
    ))
    if has_greeting:
        score += 0.30
        breakdown.append("✅ greeting present")
    else:
        breakdown.append("❌ no greeting")

    # Check body has at least 2 sentences
    sentences = re.split(r'[.!?]+', reply_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 2:
        score += 0.35
        breakdown.append(f"✅ body has {len(sentences)} sentences")
    elif len(sentences) == 1:
        score += 0.15
        breakdown.append("⚠️ body too short (1 sentence)")
    else:
        breakdown.append("❌ body missing")

    # Check closing (Best/Regards/Thanks/Sincerely/Cheers etc.)
    has_closing = bool(re.search(
        r'(Best|Regards|Thanks|Thank you|Sincerely|Cheers|Warm regards|'
        r'Kind regards|Yours truly|With regards)',
        reply_text,
        re.IGNORECASE
    ))
    if has_closing:
        score += 0.35
        breakdown.append("✅ closing present")
    else:
        breakdown.append("❌ no closing")

    return round(score, 3), " | ".join(breakdown)


def reply_relevance_reward(reply_text: str, ground_truth: Dict) -> Tuple[float, str]:
    """
    Reward 4: Does the reply address the email's key intent?
    Checks if must_include_keywords appear in the reply.
    """
    if not reply_text or not reply_text.strip():
        return 0.0, "❌ empty reply"

    keywords = ground_truth.get("must_include_keywords", [])

    # Spam emails don't need replies
    if ground_truth.get("category") == "spam" or not ground_truth.get("requires_reply"):
        return 1.0, "✅ no reply needed for this email type"

    if not keywords:
        return 0.5, "⚠️ no keywords to check (partial credit)"

    reply_lower = reply_text.lower()
    matched = []
    missed = []

    for kw in keywords:
        if kw.lower() in reply_lower:
            matched.append(kw)
        else:
            missed.append(kw)

    score = len(matched) / len(keywords)
    note = f"Keywords matched: {matched} | Missed: {missed}"
    return round(score, 3), note


def tone_matching_reward(reply_text: str, ground_truth: Dict) -> Tuple[float, str]:
    """
    Reward 6: Does the reply tone match the expected tone for this email?
    Uses lexical cues to detect the tone actually used.
    Spam/no-reply emails auto-score 1.0.
    """
    if not reply_text or not reply_text.strip():
        return 0.0, "❌ empty reply"

    expected_tone = ground_truth.get("tone", "professional")

    # Spam / no reply needed — any non-reply scores full
    if expected_tone == "none" or not ground_truth.get("requires_reply", True):
        return 1.0, "✅ no reply needed — tone N/A"

    reply_lower = reply_text.lower()

    tone_signals = {
        "professional": [
            r"\bplease\b", r"\bkindly\b", r"\bregards\b", r"\bsincerely\b",
            r"\bthank you\b", r"\bI confirm\b", r"\bplease find\b", r"\bI would like\b",
        ],
        "friendly": [
            r"\bhey\b", r"\bcheers\b", r"\bsounds (great|fun|good)\b", r"\bcount me in\b",
            r"\bwould love\b", r"\bcatch up\b", r"\bhappy to\b",
        ],
        "assertive": [
            r"\bI (expect|require|demand|insist)\b", r"\bimmediately\b",
            r"\bI (need|want) this\b", r"\bplease (ensure|confirm|action)\b",
            r"\bby (end of|no later)\b",
        ],
        "apologetic": [
            r"\bI (apologize|am sorry|sincerely apologize)\b",
            r"\bwe (apologize|are sorry)\b",
            r"\bdeep(ly)? sorry\b", r"\bregret\b", r"\bI take full responsibility\b",
            r"\bthis (was|is) unacceptable\b",
        ],
    }

    signals = tone_signals.get(expected_tone, tone_signals["professional"])
    matched = sum(1 for sig in signals if re.search(sig, reply_lower, re.IGNORECASE))
    score = min(1.0, matched / max(2, len(signals) * 0.4))

    if score >= 0.75:
        note = f"✅ tone matches expected '{expected_tone}' ({matched} signals found)"
    elif score >= 0.4:
        note = f"⚠️ weak '{expected_tone}' tone signals ({matched} found)"
    else:
        note = f"❌ tone mismatch: expected '{expected_tone}', few signals detected"

    return round(score, 3), note


def reply_length_reward(reply_text: str) -> Tuple[float, str]:
    """
    Reward 5: Is the reply an appropriate length?
    Target: 30-200 words. Too short = no value, too long = inefficient.
    """
    if not reply_text or not reply_text.strip():
        return 0.0, "❌ empty reply"

    words = len(reply_text.split())

    if 40 <= words <= 150:
        score = 1.0
        note = f"✅ perfect length ({words} words)"
    elif 25 <= words < 40:
        score = 0.7
        note = f"⚠️ a bit short ({words} words, target 40-150)"
    elif 150 < words <= 250:
        score = 0.7
        note = f"⚠️ a bit long ({words} words, target 40-150)"
    elif words < 25:
        score = 0.3
        note = f"❌ too short ({words} words)"
    else:
        score = 0.4
        note = f"❌ too long ({words} words, target 40-150)"

    return round(score, 3), note


def reply_clarity_reward(reply_text: str, ground_truth: Dict) -> Tuple[float, str]:
    """
    Reward 7: Is the reply concrete and actionable (not vague/polite fluff)?
    Penalizes empty acknowledgements and rewards specific commitments.
    """
    if not reply_text or not reply_text.strip():
        return 0.0, "❌ empty reply"

    # For spam/no-reply cases, clarity is not required.
    if ground_truth.get("category") == "spam" or not ground_truth.get("requires_reply", True):
        return 1.0, "✅ no reply required"

    text = reply_text.strip()
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    # Detect non-actionable polite fillers commonly used for reward hacking.
    filler_patterns = [
        r"\b(thanks|thank you|noted|okay|ok|got it)\b",
        r"\b(i will get back|will revert|as soon as possible)\b",
    ]
    filler_hits = sum(1 for p in filler_patterns if re.search(p, lower))

    # Signals of concrete action / clarity.
    actionable_patterns = [
        r"\b(by|before|tomorrow|today|at \d{1,2}(:\d{2})?\s?(am|pm)?)\b",  # timeline
        r"\b(confirm|schedule|send|share|prepare|review|follow up|resolve)\b",  # action verbs
        r"\b(please let me know|let me know if)\b",  # coordination phrase
    ]
    actionable_hits = sum(1 for p in actionable_patterns if re.search(p, lower))

    # Very short replies are rarely clear for actionable emails.
    if word_count < 20:
        return 0.2, f"❌ too brief for clarity ({word_count} words)"

    score = 0.4  # base for a non-empty substantive reply
    if actionable_hits >= 1:
        score += 0.25
    if actionable_hits >= 2:
        score += 0.20
    if filler_hits > 0 and actionable_hits == 0:
        score -= 0.25

    # Encourage lexical substance (not repetitive filler).
    unique_ratio = len({w.lower().strip(".,!?") for w in words}) / max(1, word_count)
    if unique_ratio >= 0.55:
        score += 0.15

    score = max(0.0, min(1.0, score))
    note = (
        f"actionable_signals={actionable_hits}, filler_signals={filler_hits}, "
        f"unique_ratio={unique_ratio:.2f}"
    )
    return round(score, 3), note


# ─────────────────────────────────────────
# COMBINED REWARD
# ─────────────────────────────────────────

STEP1_WEIGHTS = {
    "classification_accuracy": 0.7,
    "classification_format": 0.3,
}

STEP2_WEIGHTS = {
    "reply_format": 0.25,
    "reply_relevance": 0.30,
    "reply_length": 0.15,
    "tone_matching": 0.15,
    "reply_clarity": 0.15,
}


def compute_step1_reward(action: Dict, ground_truth: Dict) -> Dict:
    """Compute all rewards for the classification step."""
    acc_score, acc_note = classification_accuracy_reward(action, ground_truth)
    fmt_score, fmt_note = classification_format_reward(action)

    total = (
        STEP1_WEIGHTS["classification_accuracy"] * acc_score
        + STEP1_WEIGHTS["classification_format"] * fmt_score
    )

    return {
        "total": round(total, 4),
        "breakdown": {
            "classification_accuracy": {"score": acc_score, "note": acc_note},
            "classification_format": {"score": fmt_score, "note": fmt_note},
        },
        "step": 1,
    }


def compute_step2_reward(reply_text: str, ground_truth: Dict) -> Dict:
    """Compute all rewards for the reply drafting step."""
    fmt_score, fmt_note = reply_format_reward(reply_text)
    rel_score, rel_note = reply_relevance_reward(reply_text, ground_truth)
    len_score, len_note = reply_length_reward(reply_text)
    tone_score, tone_note = tone_matching_reward(reply_text, ground_truth)
    clarity_score, clarity_note = reply_clarity_reward(reply_text, ground_truth)

    total = (
        STEP2_WEIGHTS["reply_format"] * fmt_score
        + STEP2_WEIGHTS["reply_relevance"] * rel_score
        + STEP2_WEIGHTS["reply_length"] * len_score
        + STEP2_WEIGHTS["tone_matching"] * tone_score
        + STEP2_WEIGHTS["reply_clarity"] * clarity_score
    )

    return {
        "total": round(total, 4),
        "breakdown": {
            "reply_format": {"score": fmt_score, "note": fmt_note},
            "reply_relevance": {"score": rel_score, "note": rel_note},
            "reply_length": {"score": len_score, "note": len_note},
            "tone_matching": {"score": tone_score, "note": tone_note},
            "reply_clarity": {"score": clarity_score, "note": clarity_note},
        },
        "step": 2,
    }


def compute_episode_reward(step1_reward: Dict, step2_reward: Dict) -> float:
    """Combine step 1 and step 2 rewards into an episode total (0-1 scale)."""
    # Step 1 = 30% of episode, Step 2 = 70%
    episode_reward = 0.30 * step1_reward["total"] + 0.70 * step2_reward["total"]
    return round(episode_reward, 4)
