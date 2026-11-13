"""Readable ML models on the momentum dataset. Owner: Person 2.

Question asked to each model, ETF by ETF: "will this ETF rise over the next 4 quotes?"
The model gives a probability of a rise. A strong signal is a probability far from the usual one:
  - strong buy  = among the 10 % highest probabilities
  - strong sell = among the 10 % lowest probabilities
The two cut-offs are fixed on the train period (2021-2024), then applied unchanged to the test period.

Two readable models (no black box):
  - a shallow decision tree (3 levels): its output is a set of "if ... then ..." rules
  - a logistic regression: one weight per indicator, on standardized indicators

Usage:  python -m ml_library.momentum_models data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from backtest.momentum_dataset import build_dataset, non_overlapping

# What the models see: the trend answers of the fiche plus the indicators behind them.
# All are known at the close of the day; none uses the future.
FEATURES = [
    "trend_short", "trend_medium", "trend_long", "market_up",          # answers of the fiche
    "dist_sma20", "dist_sma50", "dist_sma200", "sma_50_slope_10",       # distance to the averages
    "ret_10d", "ret_1m", "ret_3m", "ret_6m", "ret_12m_ex_1m",            # momentum
    "trend_quality_90", "adx_14", "efficiency_ratio_20",                # strength of the trend
    "rsi_2", "rsi_14", "zscore_20", "range_position_20", "dist_high_52w",  # stretch, position
    "vol_20", "vol_60", "vol_ratio_20_60", "drawdown_60",               # risk
    "volume_rel_20", "obv_trend_20",                                    # volume
]
STRONG_SHARE = 0.10        # strong buy = top 10 % of probabilities, strong sell = bottom 10 %
TREE_DEPTH = 3             # 3 levels: at most 8 rules, readable
TREE_MIN_LEAF = 1000       # each rule must rest on at least 1000 cases
LOGISTIC_C = 0.05          # strong brake (regularization) so the weights do not learn noise


def make_models() -> dict:
    """The two models. Missing values (ADX of BEL 20, short histories) are filled with the train median."""
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=TREE_MIN_LEAF,
                                                random_state=0))
    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=LOGISTIC_C, max_iter=1000))
    return {"decision_tree": tree, "logistic": logistic}


def evaluate(data: pd.DataFrame, probability: np.ndarray, low_cut: float, high_cut: float) -> dict:
    """On non-overlapping windows: how often strong buys and strong sells were right."""
    data = data.assign(probability=probability)
    data = non_overlapping(data)
    strong_buy = data[data["probability"] >= high_cut]
    strong_sell = data[data["probability"] <= low_cut]
    rest = data[(data["probability"] > low_cut) & (data["probability"] < high_cut)]
    return {
        "auc": roc_auc_score(data["went_up"], data["probability"]),
        "base_hit_pct": 100 * data["went_up"].mean(),
        "strong_buy_cases": len(strong_buy),
        "strong_buy_hit_pct": 100 * strong_buy["went_up"].mean(),
        "strong_buy_return_pct": 100 * strong_buy["forward_return"].mean(),
        "strong_sell_cases": len(strong_sell),
        "strong_sell_hit_pct": 100 * strong_sell["went_up"].mean(),
        "strong_sell_return_pct": 100 * strong_sell["forward_return"].mean(),
        "rest_return_pct": 100 * rest["forward_return"].mean(),
    }


def fit_and_evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train each model on 2021-2024, evaluate on train and on 2025-2026."""
    train = data[data["period"] == "train"]
    test = data[data["period"] == "test"]
    results = []
    fitted = {}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["went_up"])
        fitted[name] = model
        train_probability = model.predict_proba(train[FEATURES])[:, 1]
        test_probability = model.predict_proba(test[FEATURES])[:, 1]
        low_cut = np.quantile(train_probability, STRONG_SHARE)
        high_cut = np.quantile(train_probability, 1 - STRONG_SHARE)
        for period, part, probability in [("train", train, train_probability), ("test", test, test_probability)]:
            row = {"model": name, "period": period}
            row.update(evaluate(part, probability, low_cut, high_cut))
            results.append(row)
    return pd.DataFrame(results), fitted


def tree_rules(model) -> str:
    """The decision tree written as 'if ... then ...' rules."""
    tree = model.named_steps["decisiontreeclassifier"]
    return export_text(tree, feature_names=FEATURES, show_weights=True, decimals=3)


def logistic_weights(model) -> pd.Series:
    """One weight per indicator (indicators standardized): sign = direction, size = importance."""
    weights = model.named_steps["logisticregression"].coef_[0]
    table = pd.Series(weights, index=FEATURES)
    order = table.abs().sort_values(ascending=False).index
    return table[order].round(4)


def main(path: str) -> pd.DataFrame:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    data = build_dataset(pd.read_csv(path))
    results, fitted = fit_and_evaluate(data)
    print("\nModels trained on 2021-2024, checked on 2025-2026 (non-overlapping 4-day windows)")
    print(results.round(3).to_string(index=False))
    print("\nDecision tree rules (value = [cases down, cases up] in the train period)")
    print(tree_rules(fitted["decision_tree"]))
    print("Logistic regression weights (positive = pushes towards a rise)")
    print(logistic_weights(fitted["logistic"]).to_string())
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml_library.momentum_models data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    main(sys.argv[1])
