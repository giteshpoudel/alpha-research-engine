"""Backtest engine: VectorBT execution + one shared metrics implementation.

Metrics are computed from equity curves (or an explicit per-bar returns
series for the additive funding path) and per-trade PnL lists -- never
from VectorBT's stats API, so both backtest kinds share identical math.
Annualization assumes 1h bars (8760 periods/year). Undefined metrics
(zero variance, zero drawdown, no trades) return 0.0, never NaN/inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


STD_EPS = 1e-12  # below this, std/downside-std are treated as zero


def _metrics(equity: pd.Series, trade_pnls: list[float],
             returns: pd.Series | None = None,
             total_return: float | None = None) -> BacktestResult:
    if returns is None:
        returns = equity.pct_change().dropna()
    if total_return is None:
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    std = float(returns.std(ddof=0))
    sharpe = (float(returns.mean() / std * math.sqrt(PERIODS_PER_YEAR))
              if std > STD_EPS else 0.0)
    downside = returns[returns < 0]
    dstd = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = (float(returns.mean() / dstd * math.sqrt(PERIODS_PER_YEAR))
               if dstd > STD_EPS else 0.0)

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    n = len(returns)
    if n > 0 and total_return > -1.0:
        annualized = (1.0 + total_return) ** (PERIODS_PER_YEAR / n) - 1.0
    else:
        annualized = -1.0
    calmar = float(annualized / abs(max_dd)) if max_dd < 0 else 0.0

    wins = [p > 0 for p in trade_pnls]
    win_rate = float(sum(wins) / len(wins)) if wins else 0.0
    return BacktestResult(
        total_return=total_return, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate,
        num_trades=len(trade_pnls), equity_curve=equity,
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


def run_funding_backtest(funding: pd.Series, threshold: float = 0.0001,
                         fee: float = 0.001) -> BacktestResult:
    """Funding-rate carry simulation (short perp + long spot).

    Per-bar PnL = funding rate while positioned, on fixed notional; each
    position flip costs 2 * fee (two legs), with entry charged at the window
    start when the series opens positioned. The equity curve is additive,
    equity = 1 + cumsum(pnl), anchored at 1.0 one bar before the window, so
    total_return == equity.iloc[-1] / equity.iloc[0] - 1 == sum(pnl) exactly.
    Sharpe/sortino are computed from the additive per-bar PnL series (with an
    epsilon zero-variance guard), and max drawdown from the additive curve.
    Basis risk assumed zero (documented approximation).
    """
    positioned = funding > threshold
    pos = positioned.astype(int)
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
