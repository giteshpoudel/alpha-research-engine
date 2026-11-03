# Alt-coin breakout result (2026-11-03)

Pre-registration: [`docs/prereg/2026-11-03-alt-breakout.md`](prereg/2026-11-03-alt-breakout.md)
(committed `21a9611` before this ran). Harness: `src/research/alt_breakout.py`. OOS run once.

## Decision: retire (by the pre-registered rule)

OOS max drawdown **−26.5%** breached the 20% limit. Return and Sharpe passed.

## Results (equal-weight XRP/DOGE/LINK, 1h, net of per-coin costs)

| | Return | Sharpe | Max DD | Trades | Per trade | Win rate |
|---|---|---|---|---|---|---|
| In-sample walk-forward (2023–24) | +29.3% | 0.71 | −21.8% | 145 | +63.0 bps | – |
| **OOS 2025-01 → 2026-11** | **+27.9%** | **0.72** | **−26.5%** | 336 | +26.7 bps | 38% |
| OOS, costs ×2 | +8.3% | 0.31 | −31.7% | 336 | +11.8 bps | 36% |
| Buy-and-hold (OOS) | −51.3% | −0.20 | −73.5% | – | – | – |

Exposure-matched buy-and-hold (12% in market): −3.6%.

- **By coin (OOS):** XRP +93.2% (Sharpe 1.26, DD −27.4%), DOGE +41.6% (0.82, −25.3%),
  LINK −31.1% (−0.61, −43.8%).
- **Crashes:** 2025-10-10: −0.2% vs buy-and-hold −15.8%. Binance hack: 0.0% vs −2.9%.
- Final parameters (2024): XRP N=24 v=1.0 k=3 T=24; DOGE N=48 v=1.0 k=2 T=48;
  LINK N=48 v=1.0 k=3 T=48.

## Pattern across all breakout tests so far

| Test | OOS return | Sharpe | Max DD | Buy-and-hold |
|---|---|---|---|---|
| BTC/ETH/SOL (v1) | +24.5% | 0.80 | −22.8% | −29.8% |
| XRP/DOGE/LINK | +27.9% | 0.72 | −26.5% | −51.3% |

The family keeps doing the same thing out of sample: positive net returns, far better than
holding through a bear market, sidestepping crashes, but 20–27% drawdowns of the *fully
invested* sleeve. The drawdown rule measures the sleeve as if 100% were committed on every
entry; actual drawdown in dollars scales with position size. Any future rule that changes
how drawdown is measured is a new pre-registration and cannot be applied to these results.
LINK lost money; costs ×2 roughly erase the edge, so cheap execution matters.
