# Pre-registration: 1h volatility breakout on alt coins (2026-11-03)

Written and committed **before** any alt-coin result is computed. The breakout family was
chosen from BTC/ETH/SOL work; no alt-coin data has been used with it.

## Question

Does the same 1h volatility breakout (Phase 2 family A) earn a positive net return on
liquid alt coins on Binance.US, out of sample, with each coin's own measured costs?

## Universe (fixed now, from the collector's measurements on 2026-11-03)

A coin is included if, over the last 24h on Binance.US, its median quoted spread is
≤ 5 bps **and** it traded ≥ $100k (USDT + USD pairs). Measured:

| Coin | Median spread | 24h traded | Included |
|---|---|---|---|
| XRP | 1.6 bps | $945k | yes |
| DOGE | 4.0 bps | $130k | yes |
| LINK | 4.5 bps (USDT) | $143k | yes |
| ADA | 6.3 bps | $198k | no (spread) |
| AVAX | 7.0 bps | $15k | no |
| LTC | 5.5 bps (USDT) | $26k | no |
| BNB, DOT | not collected | – | no |

## Data, costs, execution (as in Phase 2)

- Binance.US 1h bars (USDT pairs). Available from 2022-01 (LINK 2022-01-14, XRP 2023-07-14).
- Signals on bar close, fills at the next open, long/flat, whole sleeve in or out.
- Cost per side = taker fee 2 bps + half of the coin's p95 spread (rounded up) + slippage
  2 bps: **XRP 5, DOGE 7, LINK 8 bps**. ×3 on bars in the top decile of trailing 24h
  realized volatility (threshold from in-sample bars only).

## Selection (in-sample, 2022-01 → 2024-12)

Same 24-point grid as Phase 2 (N ∈ {24, 48, 96}, v ∈ {1.0, 1.3}, k ∈ {2, 3},
T ∈ {N, 2N}). Walk-forward per coin: 12-month train, 3-month test, 3-month step; each
fold picks the best train-window net Sharpe. Final parameters per coin: best net Sharpe
on 2024 (the last 12 in-sample months). Trials added: 24 × 3 = 72 (running total 314).

## Out-of-sample (2025-01 → 2026-11-01), run once with the frozen parameters

Equal-weight portfolio of the three coins. Decision rule:
- **Retire** if OOS net return ≤ 0, or Sharpe < 0.5, or max drawdown ≥ 20%.
- Otherwise the coins that are individually net positive OOS become **shadow
  candidates** to add to a forward test alongside breakout v2. Live use then follows the
  v2 forward-test rules (≥ 3 weeks shadow before small live trades).

Reported regardless: in-sample walk-forward, per-coin results, costs ×2, buy-and-hold,
the 2025-10-10 crash and the Binance hack windows.
