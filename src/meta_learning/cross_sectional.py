"""Cross-sectional momentum: rank the universe by trailing return and hold the
top-k (optionally short the bottom-k), rebalancing every ``hold`` bars.

Long-only top-k is runnable on Binance.US spot; long-short is research-only.
Low turnover at hourly/daily horizons is the point: costs amortise, unlike 5m
scalping. Signals are trailing (point-in-time); fills happen at the signal bar's
close with ``fee`` charged on turnover.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src.backtesting.data import load_ohlcv
from src.backtesting.engine import _metrics
from src.backtesting.fees import fee_for_symbol
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe


def backtest_cross_sectional(prices: pd.DataFrame, lookback: int = 168,
                             hold: int = 24, top_k: int = 3, fee: float = 0.001,
                             long_short: bool = False):
    """Equal-weight top-k (and short bottom-k) momentum. ``prices`` is a
    wide DataFrame (index=time, columns=symbols) of closes."""
    if prices.shape[1] < top_k:
        raise ValueError("not enough columns for top_k")
    signal = prices.pct_change(lookback)
    returns = prices.pct_change().fillna(0.0)
    pos = pd.Series(0.0, index=prices.columns)
    equity, curve = 1.0, []
    for i, _ts in enumerate(prices.index):
        equity *= 1.0 + float((pos * returns.iloc[i]).sum())  # earn prior weights
        if i >= lookback and i % hold == 0:
            ranked = signal.iloc[i].dropna().sort_values(ascending=False)
            if len(ranked) >= (2 * top_k if long_short else top_k):
                new = pd.Series(0.0, index=prices.columns)
                longs = ranked.index[:top_k]
                new[longs] = 1.0 / top_k
                if long_short:
                    shorts = ranked.index[-top_k:]
                    new[shorts] = -1.0 / top_k
                equity *= 1.0 - float((new - pos).abs().sum()) * fee  # turnover cost
                pos = new
        curve.append(equity)
    series = pd.Series(curve, index=prices.index, name="equity")
    return _metrics(series, [float(series.iloc[-1] - 1.0)])


def load_price_panel(ch_client, symbols, interval: str = "1h",
                     start: datetime | None = None,
                     end: datetime | None = None) -> pd.DataFrame:
    """Wide close-price panel for symbols that have data; others dropped."""
    cols = {}
    for symbol in symbols:
        try:
            cols[symbol] = load_ohlcv(ch_client, symbol, interval=interval,
                                      start=start, end=end)["close"]
        except ValueError:
            continue
    if not cols:
        raise ValueError("no price data for any symbol")
    return pd.DataFrame(cols).dropna(how="all").ffill().dropna()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum backtest")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--hold", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--long-short", action="store_true")
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--symbols", default="all")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    panel = load_price_panel(ch, symbols, interval=args.interval)
    fee = args.fee if args.fee is not None else fee_for_symbol(ch, symbols[0])
    result = backtest_cross_sectional(panel, lookback=args.lookback, hold=args.hold,
                                      top_k=args.top_k, fee=fee,
                                      long_short=args.long_short)
    print(f"cross-sectional [{args.interval}] n={panel.shape[1]} cols "
          f"lookback={args.lookback} hold={args.hold} top_k={args.top_k} "
          f"long_short={args.long_short} fee={fee * 1e4:.1f}bps")
    print(f"  total_return={result.total_return * 100:.2f}%  sharpe={result.sharpe:.3f}  "
          f"max_dd={result.max_drawdown * 100:.1f}%  trades={result.num_trades}")


if __name__ == "__main__":
    main()
