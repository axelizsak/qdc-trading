"""The five investor profiles: who each one is and how its money is invested.

From the team's customer archetypes (v1, 17 November 2026): hypotheses built on public data, not validated with
customers. The customer app places each customer in one profile (agents/profiler.py); the investment engine holds
one portfolio per profile. A profile is the typical case, not a risk test: the suitability answers of a customer
can still restrict their plan.
"""
# core: the share of the portfolio in broad, diversified ETFs. The rest is the satellite share: a few themes.
PROFILES = {
    "first_step_starter": {
        "label": "First-step starter",
        "tagline": "Small steps, long runway.",
        "who": "starting out: a student or in a first job, with a small amount and a long time ahead",
        "goal": "start, learn and build the habit",
        "horizon": "10+ years",
        "risk": "Balanced to growth",
        "core": 0.90,
        "rules": ["A savings buffer comes before the first euro invested", "No leveraged or crypto ETFs"],
    },
    "goal_builder": {
        "label": "Goal builder",
        "tagline": "A home first. Wealth after.",
        "who": "employed, often renting, saving for a home and for the long term at the same time",
        "goal": "a home deposit in 2 to 5 years, and long-term wealth",
        "horizon": "2-5 years, then 20+",
        "risk": "Defensive for the home, growth for the rest",
        "core": 0.80,
        "rules": ["The home money moves to safer ETFs as the purchase date nears"],
    },
    "self_directed_trader": {
        "label": "Self-directed trader",
        "tagline": "Your picks, on a solid core.",
        "who": "already invests alone, often at a neobroker or in crypto, and wants control and low costs",
        "goal": "control, low costs and fast gains",
        "horizon": "10+ years",
        "risk": "Growth",
        "core": 0.70,
        "rules": ["Your own picks stay inside the themes share", "Your risk level follows the loss you can bear, "
                                                                "not the one you would like to take"],
    },
    "cautious_saver": {
        "label": "Cautious saver",
        "tagline": "Slow in. Steady on.",
        "who": "has savings and has never invested, for fear of losses and of mistakes",
        "goal": "children, a buffer and a later pension",
        "horizon": "5-15 years",
        "risk": "Defensive to balanced",
        "core": 0.90,
        "rules": ["The money goes in step by step over 6 to 12 months", "The buffer stays in savings"],
    },
    "late_starter": {
        "label": "Late starter",
        "tagline": "Catch up, calmly.",
        "who": "45 or older with larger savings, starting late to close a pension gap",
        "goal": "close the pension gap",
        "horizon": "5-15 years to the pension",
        "risk": "Defensive to balanced",
        "core": 0.90,
        "rules": ["Sales are planned around the yearly EUR 10,000 capital gains exemption"],
    },
}
SLEEVES = ["core", "satellite"]
