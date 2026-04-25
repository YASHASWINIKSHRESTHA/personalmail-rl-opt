"""
PersonalMail-RL: Email Scenarios Dataset
25 realistic personal/work email scenarios with ground truth labels.
Tagged with difficulty: easy | medium | hard — for curriculum learning.
"""

SCENARIOS = [
    {
        "id": "s001",
        "difficulty": "easy",
        "subject": "URGENT: Project deadline moved to tomorrow 9 AM",
        "body": (
            "Hi,\n\nBad news - the client just called. They need the Q3 analysis report "
            "by tomorrow 9 AM instead of Friday. I know it's last minute. "
            "Can you confirm you'll make it happen? Let me know if you need anything.\n\nBest,\nSarah"
        ),
        "sender_name": "Sarah Johnson",
        "sender_email": "sarah@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "deadline acknowledgment",
            "must_include_keywords": ["confirm", "tomorrow", "Q3"],
        },
    },
    {
        "id": "s002",
        "difficulty": "easy",
        "subject": "Dinner this Friday?",
        "body": (
            "Hey!\n\nWe haven't caught up in ages. A few of us are getting together for "
            "dinner this Friday at 7pm at Spice Garden. Would love for you to join! "
            "Let me know if you're free.\n\nCheers,\nRaj"
        ),
        "sender_name": "Raj Patel",
        "sender_email": "raj.patel@gmail.com",
        "ground_truth": {
            "urgency": "low",
            "category": "social",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "dinner invitation",
            "must_include_keywords": ["Friday", "dinner"],
        },
    },
    {
        "id": "s003",
        "difficulty": "easy",
        "subject": "Your invoice #INV-2024-0892 is overdue",
        "body": (
            "Dear Customer,\n\nThis is a reminder that invoice #INV-2024-0892 for $450.00 "
            "was due on October 15th and remains unpaid. Please make payment at your earliest "
            "convenience to avoid late fees.\n\nAccounts Team\nABC Supplies Ltd"
        ),
        "sender_name": "Accounts Team",
        "sender_email": "accounts@abcsupplies.com",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "payment acknowledgment",
            "must_include_keywords": ["invoice", "payment"],
        },
    },
    {
        "id": "s004",
        "difficulty": "easy",
        "subject": "Re: Team Outing - Feedback Requested",
        "body": (
            "Hi team,\n\nWe're planning a team outing next month and would love your input. "
            "Please vote for your preferred activity: (a) Bowling, (b) Escape Room, or (c) Cooking Class. "
            "Reply by end of day Thursday!\n\nThanks,\nPriya"
        ),
        "sender_name": "Priya Sharma",
        "sender_email": "priya@company.com",
        "ground_truth": {
            "urgency": "medium",
            "category": "work",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "vote for activity",
            "must_include_keywords": ["vote", "activity"],
        },
    },
    {
        "id": "s005",
        "difficulty": "easy",
        "subject": "Congratulations! You've won a free iPhone 15!",
        "body": (
            "Dear Winner,\n\nYou have been selected as the lucky winner of an iPhone 15! "
            "Click here immediately to claim your prize: http://totally-not-scam.xyz/claim "
            "You must act within 24 hours!\n\nLottery Department"
        ),
        "sender_name": "Lottery Department",
        "sender_email": "noreply@lottery-winners.xyz",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "requires_reply": False,
            "tone": "none",
            "key_intent": "spam - no reply needed",
            "must_include_keywords": [],
        },
    },
    {
        "id": "s006",
        "difficulty": "easy",
        "subject": "Mom's birthday party - Saturday?",
        "body": (
            "Hey,\n\nJust wanted to check if you're coming to Mom's birthday party this Saturday at 3pm. "
            "We're doing it at Grandma's house. Bringing your famous biryani?\n\nLove,\nAnanya"
        ),
        "sender_name": "Ananya",
        "sender_email": "ananya.family@gmail.com",
        "ground_truth": {
            "urgency": "medium",
            "category": "personal",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "birthday party confirmation",
            "must_include_keywords": ["Saturday", "birthday"],
        },
    },
    {
        "id": "s007",
        "difficulty": "medium",
        "subject": "Performance review scheduled - Action required",
        "body": (
            "Dear Employee,\n\nYour annual performance review has been scheduled for November 20th at 2 PM. "
            "Please prepare a self-assessment form (attached template) and submit it by November 18th. "
            "Contact HR if you need to reschedule.\n\nHR Department"
        ),
        "sender_name": "HR Department",
        "sender_email": "hr@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "performance review acknowledgment",
            "must_include_keywords": ["review", "November"],
        },
    },
    {
        "id": "s008",
        "difficulty": "medium",
        "subject": "Quick question about the API integration",
        "body": (
            "Hey,\n\nHope you're doing well! I'm working on the payment API integration and "
            "got stuck on the webhook signature verification. Do you have 15 minutes to jump on "
            "a call tomorrow afternoon?\n\nThanks,\nKiran"
        ),
        "sender_name": "Kiran Mehta",
        "sender_email": "kiran@startup.io",
        "ground_truth": {
            "urgency": "medium",
            "category": "work",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "schedule a call",
            "must_include_keywords": ["call", "tomorrow"],
        },
    },
    {
        "id": "s009",
        "difficulty": "easy",
        "subject": "Gym membership renewal - Save 30% this week only",
        "body": (
            "Hi Member,\n\nYour FitZone membership expires in 30 days. "
            "Renew now and save 30% with code RENEW30. Offer valid until Sunday. "
            "Click to renew: https://fitzone.com/renew\n\nFitZone Team"
        ),
        "sender_name": "FitZone Team",
        "sender_email": "newsletter@fitzone.com",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "requires_reply": False,
            "tone": "none",
            "key_intent": "promotional newsletter - no reply needed",
            "must_include_keywords": [],
        },
    },
    {
        "id": "s010",
        "difficulty": "medium",
        "subject": "Flight cancellation - Immediate rebooking needed",
        "body": (
            "Dear Passenger,\n\nWe regret to inform you that your flight AI-204 from Delhi to Bangalore "
            "on November 15th has been cancelled due to operational reasons. "
            "Please contact us at 1800-XXX-XXXX or reply to this email to rebook at no extra charge.\n\n"
            "Air India Customer Service"
        ),
        "sender_name": "Air India Customer Service",
        "sender_email": "service@airindia.in",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "assertive",
            "key_intent": "flight rebooking request",
            "must_include_keywords": ["rebook", "flight", "cancel"],
        },
    },
    {
        "id": "s011",
        "difficulty": "hard",
        "subject": "Conflict: Your meeting overlaps with client call",
        "body": (
            "Hi,\n\nJust noticed you have the design sync scheduled at 3pm tomorrow, "
            "but the client call with Acme Corp is also at 3pm. "
            "Both are important. Which one should I reschedule, or can you handle both somehow?\n\nMegha"
        ),
        "sender_name": "Megha Iyer",
        "sender_email": "megha@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "calendar conflict resolution",
            "must_include_keywords": ["reschedule", "meeting"],
        },
    },
    {
        "id": "s012",
        "difficulty": "easy",
        "subject": "Thank you for a great presentation!",
        "body": (
            "Hi,\n\nJust wanted to say the presentation today was fantastic! "
            "The data visualizations were especially impressive. "
            "The leadership team was very happy. Well done!\n\nBest,\nDeepak"
        ),
        "sender_name": "Deepak Nair",
        "sender_email": "deepak@company.com",
        "ground_truth": {
            "urgency": "low",
            "category": "work",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "acknowledge compliment",
            "must_include_keywords": ["thank"],
        },
    },
    {
        "id": "s013",
        "difficulty": "medium",
        "subject": "Complaint: Wrong order delivered",
        "body": (
            "Hello,\n\nI ordered a blue notebook (Order #45821) but received a red one. "
            "This is the second time this has happened. I need either the correct item sent "
            "or a full refund immediately. Very disappointed.\n\nRegards,\nAmit Shah"
        ),
        "sender_name": "Amit Shah",
        "sender_email": "amit.shah.customer@gmail.com",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "apologetic",
            "key_intent": "resolve order complaint",
            "must_include_keywords": ["apolog", "refund", "resolve", "order"],
        },
    },
    {
        "id": "s014",
        "difficulty": "medium",
        "subject": "Interview invitation - Software Engineer role",
        "body": (
            "Dear Candidate,\n\nThank you for applying to Techcorp! We'd like to invite you for "
            "a technical interview on November 22nd at 10 AM (IST) via Zoom. "
            "Please confirm your availability by November 19th.\n\nRecruiting Team, Techcorp"
        ),
        "sender_name": "Techcorp Recruiting",
        "sender_email": "recruiting@techcorp.com",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "confirm interview availability",
            "must_include_keywords": ["confirm", "interview", "November 22"],
        },
    },
    {
        "id": "s015",
        "difficulty": "easy",
        "subject": "Kids' school annual day - volunteer needed",
        "body": (
            "Dear Parents,\n\nWe are organizing our Annual Day on December 10th and need parent volunteers "
            "for decoration, food stall, and event coordination. "
            "If you're interested, please reply with your preferred role.\n\nSunrise School PTA"
        ),
        "sender_name": "Sunrise School PTA",
        "sender_email": "pta@sunriseschool.edu",
        "ground_truth": {
            "urgency": "low",
            "category": "personal",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "volunteer sign-up",
            "must_include_keywords": ["volunteer", "Annual Day"],
        },
    },
    {
        "id": "s016",
        "difficulty": "hard",
        "subject": "Lease renewal - Response required by Friday",
        "body": (
            "Dear Tenant,\n\nYour apartment lease expires on December 31st. "
            "We'd like to offer a renewal at ₹28,000/month (up from ₹25,000). "
            "Please confirm by this Friday if you'd like to renew, otherwise we'll begin advertising.\n\n"
            "BuildWell Properties"
        ),
        "sender_name": "BuildWell Properties",
        "sender_email": "leasing@buildwell.in",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "assertive",
            "key_intent": "lease renewal decision",
            "must_include_keywords": ["renew", "lease", "Friday"],
        },
    },
    {
        "id": "s017",
        "difficulty": "medium",
        "subject": "Can you cover my shift on Sunday?",
        "body": (
            "Hey,\n\nI have a family emergency and can't make my Sunday shift (10am-6pm). "
            "Could you possibly cover for me? I'll owe you one - happy to cover your next shift!\n\nThanks,\nVinay"
        ),
        "sender_name": "Vinay Kumar",
        "sender_email": "vinay.k@workplace.com",
        "ground_truth": {
            "urgency": "medium",
            "category": "work",
            "requires_reply": True,
            "tone": "friendly",
            "key_intent": "shift coverage request",
            "must_include_keywords": ["Sunday", "shift"],
        },
    },
    {
        "id": "s018",
        "difficulty": "easy",
        "subject": "Monthly newsletter - October Highlights",
        "body": (
            "Hi there!\n\nHere's what happened at TechTalks in October: "
            "3 new webinars, 2 new courses launched, 500 new members joined. "
            "Coming up in November: AI workshop on the 15th. "
            "Unsubscribe: https://techtalks.com/unsubscribe\n\nTechTalks Team"
        ),
        "sender_name": "TechTalks Team",
        "sender_email": "newsletter@techtalks.com",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "requires_reply": False,
            "tone": "none",
            "key_intent": "newsletter - no reply needed",
            "must_include_keywords": [],
        },
    },
    {
        "id": "s019",
        "difficulty": "hard",
        "subject": "Apology needed - I sent the wrong file to the client",
        "body": (
            "Hi,\n\nI accidentally sent the internal pricing document to the client instead of "
            "the final proposal. They've already seen it. I've told my manager. "
            "Can you help me draft an apology email to the client? "
            "Need it in 30 minutes.\n\nPanic mode,\nSneha"
        ),
        "sender_name": "Sneha Rajan",
        "sender_email": "sneha@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "apologetic",
            "key_intent": "help draft urgent apology",
            "must_include_keywords": ["apolog", "help", "draft"],
        },
    },
    {
        "id": "s020",
        "difficulty": "medium",
        "subject": "Recommendation letter request",
        "body": (
            "Dear Professor Rao,\n\nI hope you're well. I'm applying for an MBA program at IIM Bangalore "
            "and would be honored if you could write a recommendation letter for me. "
            "The deadline is December 15th. I'd be happy to provide any details you need.\n\n"
            "Warm regards,\nArjun Verma"
        ),
        "sender_name": "Arjun Verma",
        "sender_email": "arjun.verma@gmail.com",
        "ground_truth": {
            "urgency": "medium",
            "category": "personal",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "recommendation letter request",
            "must_include_keywords": ["recommend", "letter", "MBA"],
        },
    },
]


def get_random_scenario():
    import random
    return random.choice(SCENARIOS)


def get_scenario_by_id(scenario_id: str):
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    return None

# ─── Hard scenarios: ambiguous intent, multi-issue, tone-sensitive ──────────

SCENARIOS += [
    {
        "id": "s021",
        "difficulty": "hard",
        "subject": "Re: Re: Re: The budget issue AND the timeline AND the team",
        "body": (
            "Hi,\n\nOK so to summarise the last three emails: (1) the budget has been cut by 20%, "
            "(2) the client moved the deadline forward by two weeks, AND (3) two team members just "
            "resigned. I need you to send a response to the client that doesn't reveal (2) or (3), "
            "acknowledges (1) diplomatically, and still gives them confidence we'll deliver. "
            "Also CC Priya and BCC Legal. Can you draft that by 5pm today?\n\nRegards,\nAkhil"
        ),
        "sender_name": "Akhil Gupta",
        "sender_email": "akhil@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "draft sensitive client communication under constraints",
            "must_include_keywords": ["budget", "deliver", "confident"],
        },
    },
    {
        "id": "s022",
        "difficulty": "hard",
        "subject": "I think I'm being harassed by a colleague — not sure what to do",
        "body": (
            "Hi,\n\nI'm not sure if I'm overreacting but a colleague has been making comments "
            "about my accent in team meetings and laughing when I mispronounce things. "
            "It happened three times this week. I don't want to escalate and make things "
            "awkward, but it's affecting my confidence. Should I reply to the thread where "
            "it happened or talk to HR directly? What would you advise?\n\nThanks,\nNandini"
        ),
        "sender_name": "Nandini Krishnan",
        "sender_email": "nandini@company.com",
        "ground_truth": {
            "urgency": "high",
            "category": "work",
            "requires_reply": True,
            "tone": "apologetic",
            "key_intent": "support colleague on workplace harassment concern",
            "must_include_keywords": ["HR", "support", "document"],
        },
    },
    {
        "id": "s023",
        "difficulty": "hard",
        "subject": "Your LinkedIn post was offensive — I'm telling your employer",
        "body": (
            "I just saw your post about the recent layoffs at XYZ Corp. I work there and "
            "I found your comment that 'most of the layoffs were probably deserved' deeply "
            "offensive and hurtful to me and my colleagues. I am going to forward this to "
            "your HR department unless you publicly apologize and delete the post "
            "within 24 hours.\n\nVarun Desai"
        ),
        "sender_name": "Varun Desai",
        "sender_email": "varun.desai.personal@gmail.com",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "apologetic",
            "key_intent": "de-escalate threat, offer sincere apology",
            "must_include_keywords": ["apologize", "understand", "remove"],
        },
    },
    {
        "id": "s024",
        "difficulty": "medium",
        "subject": "Bank account suspicious activity — verify or block",
        "body": (
            "Dear Account Holder,\n\nWe have detected unusual login activity on your account "
            "from a device in Frankfurt, Germany at 02:14 AM IST. If this was not you, "
            "please click the secure link below within 2 hours to freeze your account and "
            "prevent further access. If this was you, no action is needed.\n\n"
            "Secure link: https://secure.hdfc-verify-now.net/action\n\n"
            "HDFC Bank Security Team"
        ),
        "sender_name": "HDFC Bank Security",
        "sender_email": "security@hdfc-verify-now.net",
        "ground_truth": {
            "urgency": "low",
            "category": "spam",
            "requires_reply": False,
            "tone": "none",
            "key_intent": "phishing attempt — do not click, mark as spam",
            "must_include_keywords": [],
        },
    },
    {
        "id": "s025",
        "difficulty": "hard",
        "subject": "Salary negotiation follow-up — they came in lower than expected",
        "body": (
            "Hi,\n\nThank you for the offer letter! I'm excited about the role. "
            "However, the salary of ₹18 LPA is lower than what I was expecting (₹22 LPA) "
            "based on my research and my current package of ₹16.5 LPA. "
            "Is there any flexibility? I don't want to lose this opportunity but I also "
            "need to make sure it's the right move financially. "
            "Would love to get on a call to discuss.\n\nWarm regards,\nShreya Bose"
        ),
        "sender_name": "Shreya Bose",
        "sender_email": "shreya.bose@gmail.com",
        "ground_truth": {
            "urgency": "high",
            "category": "personal",
            "requires_reply": True,
            "tone": "professional",
            "key_intent": "respond to salary negotiation request",
            "must_include_keywords": ["salary", "discuss", "call"],
        },
    },
]


def get_random_scenario(difficulty: str = None):
    import random
    pool = SCENARIOS
    if difficulty:
        pool = [s for s in SCENARIOS if s.get("difficulty") == difficulty] or SCENARIOS
    return random.choice(pool)


def get_scenario_by_id(scenario_id: str):
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    return None


def get_scenarios_by_difficulty(difficulty: str):
    return [s for s in SCENARIOS if s.get("difficulty") == difficulty]


def curriculum_sample(step: int, total_steps: int) -> dict:
    """
    Curriculum sampling: return a scenario appropriate for training progress.
    - First 30% of training: easy only
    - 30-70%: easy + medium
    - 70%+: all difficulties
    """
    import random
    progress = step / max(total_steps, 1)
    if progress < 0.30:
        pool = get_scenarios_by_difficulty("easy")
    elif progress < 0.70:
        pool = [s for s in SCENARIOS if s.get("difficulty") in ("easy", "medium")]
    else:
        pool = SCENARIOS
    return random.choice(pool or SCENARIOS)
