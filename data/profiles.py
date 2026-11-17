"""The five investor profiles: who each one is and how its money is invested.

From the team's customer archetypes (v1, 17 November 2026): hypotheses built on public data, not validated with
customers. The customer app places each customer in one profile (agents/profiler.py); the investment engine holds
one portfolio per profile. A profile is the typical case, not a risk test: the suitability answers of a customer
can still restrict their plan.

Each profile has one fixed model portfolio: a share per sleeve (SLEEVES), the same ETFs in the core sleeves for
everyone (SLEEVE_ETFS), and its own themes in the satellites. Nothing in it follows the market: the agents inform
the customer on each ETF, they do not move the weights. The numbers are the team's proposal, to challenge with
backtest/README.md, section 7.
"""
# The sleeves of a model portfolio, in the order the app shows them, and what the customer reads.
SLEEVES = {"core_equity": "Core equity", "bonds": "Bonds", "gold": "Gold", "satellites": "Themes", "cash": "Cash"}
# The ETFs of each core sleeve and their share of the sleeve, the same for every profile.
SLEEVE_ETFS = {
    "core_equity": {"WORLD": 0.70, "EUROPE 600": 0.15, "EMMKT": 0.15},
    "bonds": {"GLOBAL AGG BOND": 0.60, "EUR GOVT BOND": 0.40},
    "gold": {"GOLD": 1.0},
    "cash": {"OVERNIGHT CASH": 1.0},
}
# level: risk level from 1 (lowest) to 5. sleeves: the share of the portfolio in each sleeve. themes: the ETFs of
# the satellites sleeve, in equal parts.
PROFILES = {
    "first_step_starter": {
        "label": "First-step starter",
        "tagline": "Small steps, long runway.",
        "who": "starting out: a student or in a first job, with a small amount and a long time ahead",
        "goal": "start, learn and build the habit",
        "horizon": "10+ years",
        "risk": "Balanced to growth",
        "level": 3,
        "sleeves": {"core_equity": 0.60, "bonds": 0.20, "gold": 0.05, "satellites": 0.10, "cash": 0.05},
        "themes": ["ROBOTICS", "CLEAN ENERGY"],
        "rules": ["A savings buffer comes before the first euro invested", "No leveraged or crypto ETFs"],
    },
    "goal_builder": {
        "label": "Goal builder",
        "tagline": "A home first. Wealth after.",
        "who": "employed, often renting, saving for a home and for the long term at the same time",
        "goal": "a home deposit in 2 to 5 years, and long-term wealth",
        "horizon": "2-5 years, then 20+",
        "risk": "Defensive for the home, growth for the rest",
        "level": 3,
        "sleeves": {"core_equity": 0.45, "bonds": 0.20, "gold": 0.10, "satellites": 0.20, "cash": 0.05},
        "themes": ["SEMIS", "HEALTH CARE"],
        "rules": ["The home money moves to safer ETFs as the purchase date nears"],
    },
    "self_directed_trader": {
        "label": "Self-directed trader",
        "tagline": "Your picks, on a solid core.",
        "who": "already invests alone, often at a neobroker or in crypto, and wants control and low costs",
        "goal": "control, low costs and fast gains",
        "horizon": "10+ years",
        "risk": "Growth",
        "level": 5,
        "sleeves": {"core_equity": 0.55, "bonds": 0.10, "gold": 0.05, "satellites": 0.30, "cash": 0.00},
        "themes": ["SEMIS", "NASDAQ", "BITCOIN", "URANIUM", "DEFENSE"],
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
        "level": 2,
        "sleeves": {"core_equity": 0.30, "bonds": 0.40, "gold": 0.10, "satellites": 0.10, "cash": 0.10},
        "themes": ["HEALTH CARE", "STAPLES"],
        "rules": ["The money goes in step by step over 6 to 12 months", "The buffer stays in savings"],
    },
    "late_starter": {
        "label": "Late starter",
        "tagline": "Catch up, calmly.",
        "who": "45 or older with larger savings, starting late to close a pension gap",
        "goal": "close the pension gap",
        "horizon": "5-15 years to the pension",
        "risk": "Defensive to balanced",
        "level": 2,
        "sleeves": {"core_equity": 0.40, "bonds": 0.30, "gold": 0.10, "satellites": 0.10, "cash": 0.10},
        "themes": ["DIVIDEND LEADERS", "HEALTH CARE"],
        "rules": ["Sales are planned around the yearly EUR 10,000 capital gains exemption"],
    },
}
