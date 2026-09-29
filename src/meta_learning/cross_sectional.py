"""Cross-sectional momentum: rank the universe by trailing return and hold the
top-k (optionally short the bottom-k), rebalancing every ``hold`` bars.

Long-only top-k is runnable on Binance.US spot; long-short is research-only.
Low turnover at hourly/daily horizons is the point: costs amortise, unlike 5m
scalping. Signals are trailing (point-in-time); fills happen at the signal bar's
close with ``fee`` charged on turnover.

The optional ``vol_target`` overlay scales exposure *down* when the strategy's
own trailing volatility is above target (causal: exposure at bar t uses vol up to
t-1). It caps drawdown without changing the signal.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime

import pandas as pd

from src.backtesting.data import load_ohlcv
from src.backtesting.engine import _derive_periods_per_year, _metrics
from src.backtesting.fees import fee_for_symbol
from src.ingestion.schemas import get_clickhouse_client
from src.ingestion.universe import universe


def apply_vol_target(net: pd.Series, target_annual: float, vol_window: int = 168,
                     periods_per_year: float | None = None) -> pd.Series:
    """Scale returns down when trailing annualised vol exceeds ``target_annual``.

    Exposure for bar t is computed from returns strictly before t, so this is
    causal and adds no look-ahead.
    """
    if periods_per_year is None:
        periods_per_year = _derive_periods_per_year(net.index)
    min_periods = max(2, vol_window // 4)
    realized = net.rolling(vol_window, min_periods=min_periods).std() \
        * math.sqrt(periods_per_year)
    exposure = (target_annual / realized).clip(upper=1.0).fillna(0.0)
    return (net * exposure.shift(1).fillna(0.0)).rename(net.name)


def backtest_cross_sectional(prices: pd.DataFrame, lookback: int = 168,
                             hold: int = 24, top_k: int = 3, fee: float = 0.001,
                             long_short: bool = False, vol_target: float | None = None,
                             vol_window: int = 168):
    """Equal-weight top-k (and short bottom-k) momentum. ``prices`` is a wide
    DataFrame (index=time, columns=symbols) of closes."""
    if prices.shape[1] < top_k:
        raise ValueError("not enough columns for top_k")
    signal = prices.pct_change(lookback)
    returns = prices.pct_change().fillna(0.0)
    pos = pd.Series(0.0, index=prices.columns)
    net = []
    for i in range(len(prices)):
        bar = float((pos * returns.iloc[i]).sum())
        cost = 0.0
        if i >= lookback and i % hold == 0:
            ranked = signal.iloc[i].dropna().sort_values(ascending=False)
            if len(ranked) >= (2 * top_k if long_short else top_k):
                new = pd.Series(0.0, index=prices.columns)
                new[ranked.index[:top_k]] = 1.0 / top_k
                if long_short:
                    new[ranked.index[-top_k:]] = -1.0 / top_k
                cost = float((new - pos).abs().sum()) * fee
                pos = new
        net.append(bar - cost)
    net_series = pd.Series(net, index=prices.index)
    if vol_target is not None:
        net_series = apply_vol_target(net_series, vol_target, vol_window)
    equity = (1.0 + net_series).cumprod()
    return _metrics(equity, [float(equity.iloc[-1] - 1.0)])


def select_oos(prices: pd.DataFrame, is_end, grid: list[dict], fee: float = 0.001,
               long_short: bool = False, vol_target: float | None = None):
    """Pick the grid point with the best IS Sharpe, then evaluate it OOS.

    Returns (best_params, is_sharpe, oos_result). The OOS slice is never touched
    during selection.
    """
    is_panel = prices.loc[:is_end]
    best, best_sharpe = None, float("-inf")
    for params in grid:
        result = backtest_cross_sectional(is_panel, fee=fee, long_short=long_short,
                                          vol_target=vol_target, **params)
        if result.sharpe > best_sharpe:
            best, best_sharpe = params, result.sharpe
    oos_panel = prices.loc[is_end:].iloc[1:]  # drop the boundary bar
    oos_result = backtest_cross_sectional(oos_panel, fee=fee, long_short=long_short,
                                          vol_target=vol_target, **best)
    return best, best_sharpe, oos_result


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


def _print(label: str, result) -> None:
    print(f"  {label:28} total={result.total_return * 100:8.2f}%  "
          f"sharpe={result.sharpe:5.3f}  max_dd={result.max_drawdown * 100:6.1f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum backtest")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--hold", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--long-short", action="store_true")
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--vol-target", type=float, default=None,
                        help="annualised vol cap, e.g. 0.40")
    parser.add_argument("--oos", default=None,
                        help="IS/OOS split date (YYYY-MM-DD); selects on IS, tests OOS")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    panel = load_price_panel(ch, symbols, interval=args.interval)
    fee = args.fee if args.fee is not None else fee_for_symbol(ch, symbols[0])
    print(f"cross-sectional [{args.interval}] n={panel.shape[1]} cols fee={fee * 1e4:.1f}bps "
          f"long_short={args.long_short} vol_target={args.vol_target}")

    if args.oos:
        grid = [{"lookback": lb, "hold": args.hold, "top_k": k}
                for lb in (24, 72, 168, 336) for k in (2, 3)]
        best, is_sharpe, oos = select_oos(panel, pd.Timestamp(args.oos, tz="UTC"), grid,
                                          fee=fee, long_short=args.long_short,
                                          vol_target=args.vol_target)
        print(f"  selected on IS: {best} (IS sharpe {is_sharpe:.3f})")
        _print("OOS", oos)
        return

    base = backtest_cross_sectional(panel, lookback=args.lookback, hold=args.hold,
                                    top_k=args.top_k, fee=fee,
                                    long_short=args.long_short)
    _print("base", base)
    if args.vol_target is not None:
        overlay = backtest_cross_sectional(panel, lookback=args.lookback, hold=args.hold,
                                           top_k=args.top_k, fee=fee,
                                           long_short=args.long_short,
                                           vol_target=args.vol_target)
        _print("vol-target overlay", overlay)


if __name__ == "__main__":
    main()
