"""Backtest engine: VectorBT execution + one shared metrics implementation.

Metrics are computed from equity curves (or an explicit per-bar returns
series for the additive funding path) and per-trade PnL lists -- never
from VectorBT's stats API, so both backtest kinds share identical math.
Annualization defaults to 1h bars (8760 periods/year) but is derived from
the equity index spacing when the caller does not pass periods_per_year.
Undefined metrics (zero variance, zero drawdown, no trades) return 0.0,
never NaN/inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

PERIODS_PER_YEAR = 8760  # 1h bars


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series
    avg_trade_return: float = 0.0  # mean net-of-fee PnL per trade


STD_EPS = 1e-12  # below this, std/downside-std are treated as zero


def _derive_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Periods/year from median index spacing (hourly -> 8760, 7-day -> ~52)."""
    if len(index) < 2:
        return float(PERIODS_PER_YEAR)
    median_hours = index.to_series().diff().dropna().median().total_seconds() / 3600.0
    if median_hours <= 0:
        return float(PERIODS_PER_YEAR)
    return PERIODS_PER_YEAR / median_hours


def _metrics(equity: pd.Series, trade_pnls: list[float],
             returns: pd.Series | None = None,
             total_return: float | None = None,
             periods_per_year: float | None = None) -> BacktestResult:
    if returns is None:
        returns = equity.pct_change().dropna()
    if total_return is None:
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    if periods_per_year is None:
        periods_per_year = _derive_periods_per_year(equity.index)

    std = float(returns.std(ddof=0))
    sharpe = (float(returns.mean() / std * math.sqrt(periods_per_year))
              if std > STD_EPS else 0.0)
    downside = returns[returns < 0]
    dstd = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = (float(returns.mean() / dstd * math.sqrt(periods_per_year))
               if dstd > STD_EPS else 0.0)

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    n = len(returns)
    if n > 0 and total_return > -1.0:
        annualized = (1.0 + total_return) ** (periods_per_year / n) - 1.0
    else:
        annualized = -1.0
    calmar = float(annualized / abs(max_dd)) if max_dd < 0 else 0.0

    wins = [p > 0 for p in trade_pnls]
    win_rate = float(sum(wins) / len(wins)) if wins else 0.0
    avg_trade = float(sum(trade_pnls) / len(trade_pnls)) if trade_pnls else 0.0
    return BacktestResult(
        total_return=total_return, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate,
        num_trades=len(trade_pnls), equity_curve=equity,
        avg_trade_return=avg_trade,
    )


def run_signal_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series,
                        fee: float = 0.001) -> BacktestResult:
    """Long-only signal backtest via VectorBT execution."""
    pf = vbt.Portfolio.from_signals(close, entries, exits, fees=fee,
                                    freq="1h", init_cash=1.0)
    equity = pf.value().rename("equity")
    equity = equity / equity.iloc[0]
    pnls = [float(p) for p in pf.trades.records_readable["PnL"]]
    return _metrics(equity, pnls)


def run_maker_backtest(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
                       fee: float = 0.0, limit_bps: float = 5.0,
                       max_wait: int = 1, taker_fee: float | None = None
                       ) -> BacktestResult:
    """Long-only passive (maker) execution on OHLC bars.

    A signal posts a limit order ``limit_bps`` in the market's favor off the
    signal bar's close (buy below, sell above). The order fills only if a later
    bar trades through it (buy: ``low <= limit``; sell: ``high >= limit``), so
    fill probability and adverse selection are implicit: passive bids fill when
    price dips to them. Unfilled entries are cancelled after ``max_wait`` bars;
    unfilled exits fall back to a market (taker) fill at that bar's close, and
    any open position is force-closed at the last close with the taker fee.
    """
    if taker_fee is None:
        taker_fee = fee
    for series in (entries, exits):
        if not isinstance(series, pd.Series):
            raise TypeError("entries/exits must be pandas Series")
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ent = entries.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    ext = exits.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    off = limit_bps / 1e4
    n = len(df)
    equity = np.ones(n, dtype=float)
    cash, qty, entry_cash = 1.0, 0.0, 1.0
    pending: tuple[str, float, int] | None = None
    trade_pnls: list[float] = []

    for i in range(n):
        if pending is not None:
            side, price, waited = pending
            filled = ((side == "buy" and low[i] <= price)
                      or (side == "sell" and high[i] >= price))
            if filled:
                if side == "buy":
                    entry_cash = cash
                    qty = cash / (price * (1.0 + fee))
                    cash = 0.0
                else:
                    cash = qty * price * (1.0 - fee)
                    qty = 0.0
                    trade_pnls.append(cash / entry_cash - 1.0)
                pending = None
            elif waited + 1 >= max_wait:
                if side == "sell":
                    cash = qty * close[i] * (1.0 - taker_fee)
                    qty = 0.0
                    trade_pnls.append(cash / entry_cash - 1.0)
                pending = None
            else:
                pending = (side, price, waited + 1)

        if pending is None:
            if qty == 0.0 and ent[i]:
                pending = ("buy", close[i] * (1.0 - off), 0)
            elif qty > 0.0 and ext[i]:
                pending = ("sell", close[i] * (1.0 + off), 0)

        equity[i] = cash + qty * close[i]

    if qty > 0.0:
        cash = qty * close[-1] * (1.0 - taker_fee)
        trade_pnls.append(cash / entry_cash - 1.0)
        equity[-1] = cash

    series = pd.Series(equity, index=df.index, name="equity")
    return _metrics(series, trade_pnls, total_return=float(series.iloc[-1] - 1.0))


def run_funding_backtest(funding: pd.Series, threshold: float = 0.0001,
                         fee: float = 0.001,
                         exit_threshold: float | None = None) -> BacktestResult:
    """Funding-rate carry simulation (short perp + long spot).

    Per-bar PnL = funding rate while positioned, on fixed notional; each
    position flip costs 2 * fee (two legs), with entry charged at the window
    start when the series opens positioned. The equity curve is additive,
    equity = 1 + cumsum(pnl), anchored at 1.0 one bar before the window, so
    total_return == equity.iloc[-1] / equity.iloc[0] - 1 == sum(pnl) exactly.
    Sharpe/sortino are computed from the additive per-bar PnL series (with an
    epsilon zero-variance guard), and max drawdown from the additive curve.
    Basis risk assumed zero (documented approximation).

    ``exit_threshold`` (default = ``threshold``) adds hysteresis: enter when
    funding > threshold, exit only when funding < exit_threshold, so a
    noisy series around the entry level doesn't churn positions.
    """
    if exit_threshold is None:
        exit_threshold = threshold
    state, positions = 0, []
    for rate in funding:
        if state == 0 and rate > threshold:
            state = 1
        elif state == 1 and rate < exit_threshold:
            state = 0
        positions.append(state)
    pos = pd.Series(positions, index=funding.index)
    positioned = pos.astype(bool)
    flips = pos.diff().fillna(pos.iloc[0]).abs()
    cost = flips * 2.0 * fee
    pnl = funding.where(positioned, 0.0) - cost
    equity = (1.0 + pnl.cumsum()).rename("equity")
    step = (funding.index[1] - funding.index[0]) if len(funding) > 1 \
        else pd.Timedelta(hours=1)
    anchor = pd.Series([1.0], index=[funding.index[0] - step])
    equity = pd.concat([anchor, equity])

    trade_pnls: list[float] = []
    in_position = False
    accrual = 0.0
    for is_pos, bar_pnl in zip(positioned, pnl):
        if is_pos and not in_position:
            in_position = True
            accrual = float(bar_pnl)  # includes entry cost
        elif is_pos:
            accrual += float(bar_pnl)
        elif in_position:
            in_position = False
            accrual += float(bar_pnl)  # includes exit cost (bar_pnl is -cost here)
            trade_pnls.append(accrual)
    if in_position:
        trade_pnls.append(accrual)  # open position closed at window end
    return _metrics(equity, trade_pnls, returns=pnl,
                    total_return=float(pnl.sum()))
