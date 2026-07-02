"""
fetch_data.py
-------------
Phase 1: Pull daily price data for our asset universe + India VIX.

WHY THIS FILE EXISTS AS A SEPARATE STEP:
We deliberately separate "fetching raw data" from "processing it" (see
process_data.py, which we'll build next). This is standard practice in
quant research pipelines: raw data should be pulled once, cached to disk,
and never re-touched. Every downstream step (features, HMM, backtest)
reads from the cached raw file. This means:
  1. You don't hammer Yahoo Finance's API every time you re-run your pipeline.
  2. Your results are reproducible - if yfinance changes historical data
     retroactively (it happens), you have a frozen snapshot to point to.
  3. You have a clean point to inspect "is my raw data actually correct"
     before any transformation could hide a bug.

ASSET UNIVERSE (edit TICKERS below if you want to change this):
  - ^NSEI          : NIFTY 50 index - our "equity" asset class
  - GOLDBEES.NS     : Gold ETF on NSE - our "gold" asset class
  - LIQUIDBEES.NS   : Liquid/overnight fund ETF - proxy for "bonds"/cash
  - ^INDIAVIX       : India VIX - NOT a tradeable asset, this is a FEATURE
                       (volatility/fear gauge) for the HMM, not something
                       we hold in the portfolio.
"""

import yfinance as yf
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---- CONFIG ----
TICKERS = {
    "equity": "^NSEI",
    "gold": "GOLDBEES.NS",
    "bond_proxy": "LIQUIDBEES.NS",
}
VIX_TICKER = "^INDIAVIX"

START_DATE = "2011-01-01"
END_DATE = None

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ticker(ticker: str, name: str) -> pd.DataFrame:
    print(f"Fetching {name} ({ticker})...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check the ticker symbol is "
            f"correct and that you have internet access to Yahoo Finance."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": name})

    n_missing = df[name].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} NaN values in {name} raw close prices")

    n_zero_or_negative = (df[name] <= 0).sum()
    if n_zero_or_negative > 0:
        raise ValueError(
            f"{name} has {n_zero_or_negative} zero/negative prices - "
            f"this is corrupted data, investigate before proceeding."
        )

    print(f"  OK: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}, "
          f"{n_missing} missing values")
    return df


def main():
    all_frames = []

    for name, ticker in TICKERS.items():
        df = fetch_ticker(ticker, name)
        all_frames.append(df)

    vix_df = fetch_ticker(VIX_TICKER, "india_vix")
    all_frames.append(vix_df)

    combined = all_frames[0]
    for df in all_frames[1:]:
        combined = combined.join(df, how="inner")

    print(f"\nCombined dataset: {len(combined)} rows, "
          f"{combined.index[0].date()} to {combined.index[-1].date()}")
    print(f"Columns: {list(combined.columns)}")

    remaining_nans = combined.isna().sum().sum()
    if remaining_nans > 0:
        print(f"WARNING: {remaining_nans} NaN values remain after inner join. "
              f"Investigate before proceeding.")

    out_path = RAW_DIR / "raw_prices.csv"
    combined.to_csv(out_path)
    print(f"\nSaved raw price data to: {out_path}")

    return combined


if __name__ == "__main__":
    main()
