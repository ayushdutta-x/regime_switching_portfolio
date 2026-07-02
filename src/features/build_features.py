"""
build_features.py
------------------
Phase 2: Turn raw prices into momentum + volatility features that
describe the market's current "state" - these are what we'll feed
into the HMM in Phase 3.

IMPORTANT SCOPE NOTE:
We are NOT doing any global normalization (z-scoring) here that spans
the entire dataset. Per the project spec, any feature scaling must be
computed using only training-window data - that happens later, inside
the walk-forward validation loop (Phase 4). Doing it here, globally,
would be a subtle form of lookahead bias: it would mean day 500's
z-score secretly depends on the mean/std of data from day 5000.
So this script only computes RAW rolling features. Scaling comes later.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "raw_prices.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows in trading days (~21 trading days per month)
MOMENTUM_WINDOWS = {"1w": 5, "1m": 21, "1q": 63}
VOL_WINDOW = 21  # ~1 month, standard choice for short-term realized vol

PORTFOLIO_ASSETS = ["equity", "gold", "bond_proxy"]


def load_raw_prices() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    print(f"Loaded raw prices: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def compute_log_returns(prices: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Log returns: ln(P_t / P_t-1). Time-additive, more symmetric than
    simple returns, standard for this kind of modeling."""
    returns = pd.DataFrame(index=prices.index)
    for col in cols:
        returns[f"{col}_ret"] = np.log(prices[col] / prices[col].shift(1))
    return returns


def compute_momentum_features(returns: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Momentum = rolling MEAN of daily log returns over different windows.
    A positive rolling mean = trending up over that window, negative = trending down.
    We use equity returns as the primary momentum signal since that's
    what "market mood" mostly tracks - but this is a design choice worth
    noting in your README (you could compute momentum per-asset too).
    """
    feats = pd.DataFrame(index=returns.index)
    ret_col = f"{asset}_ret"
    for label, window in MOMENTUM_WINDOWS.items():
        feats[f"momentum_{label}"] = returns[ret_col].rolling(window).mean()
    return feats


def compute_volatility_features(returns: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Volatility = rolling STANDARD DEVIATION of daily log returns.
    This is 'realized volatility' - how much the asset actually swung
    recently, as opposed to VIX which is implied/forward-looking vol
    priced by options markets. We use both as separate, complementary
    features later.
    """
    feats = pd.DataFrame(index=returns.index)
    ret_col = f"{asset}_ret"
    feats[f"volatility_{VOL_WINDOW}d"] = returns[ret_col].rolling(VOL_WINDOW).std()
    return feats


def main():
    prices = load_raw_prices()

    # Step 1: log returns for portfolio assets
    returns = compute_log_returns(prices, PORTFOLIO_ASSETS)

    # Step 2: momentum + volatility features, built off EQUITY returns
    # (equity is our primary "market mood" signal - NIFTY swings are
    # what we mean by Bull/Bear/Crisis in this project)
    momentum = compute_momentum_features(returns, "equity")
    volatility = compute_volatility_features(returns, "equity")

    # Step 3: bring in India VIX as a raw feature too (no transformation
    # needed - VIX is already a volatility measure by construction)
    vix = prices[["india_vix"]].copy()

    # Combine everything into one features table
    features = pd.concat([returns, momentum, volatility, vix], axis=1)

    # Rolling windows create NaNs at the start (e.g. first 63 days can't
    # have a valid quarterly momentum value yet - there's no 63 days of
    # history before day 63). We drop these rather than fill them, since
    # filling would be fabricating data that didn't exist.
    n_before = len(features)
    features = features.dropna()
    n_after = len(features)
    print(f"\nDropped {n_before - n_after} rows with NaN (from rolling window warm-up)")
    print(f"Final feature set: {n_after} rows, {features.index[0].date()} to {features.index[-1].date()}")
    print(f"Columns: {list(features.columns)}")

    out_path = PROCESSED_DIR / "features.csv"
    features.to_csv(out_path)
    print(f"\nSaved features to: {out_path}")

    return features


if __name__ == "__main__":
    main()