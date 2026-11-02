# Pre-registration: Phase 2 intraday strategies (2026-11-02)

Written and committed **before** any result was computed. Changing anything below after
results exist requires a new pre-registration with a new date, and the trial count carries over.

## Question

Is there a long/flat intraday strategy on Binance.US spot (BTC, ETH, SOL) that earns a
positive return net of realistic costs, out of sample, beyond what its market exposure
alone would earn?

## Data

- Bars: Binance.US 5m klines (USDT pairs, USD fallback), resampled to **15m** and **1h**.
  Live trading uses the USD pairs; USDT/USD prices track within a few bps, which is
  immaterial for these signals.
- **In-sample (IS):** 2022-01-01 → 2024-12-31. All selection happens here.
- **Out-of-sample (OOS):** 2025-01-01 → 2026-11-01. Touched **once**, after IS selection
  is frozen. Includes the 2025-10-10 crash and the 2026-10-29 Binance hack.

## Execution and cost model

- Signals use bar close *t*; fills happen at the **open of bar t+1** (one-bar lag).
  Exits are also evaluated on closes; there are no intrabar stop fills.
- Cost per side = taker fee **2 bps** + half-spread + slippage **2 bps**. Half-spread uses
  the conservative end of measured Binance.US USD spreads (p99/2, rounded up):
  **BTC 1, ETH 1, SOL 2 bps**. Base cost per side: **BTC 5, ETH 5, SOL 6 bps**.
- **Stress:** on bars whose trailing 24h realized volatility is in the top decile (decile
  set on IS only), cost per side is **×3**.
- Sensitivity, reported but not used for selection: all costs ×2, and a Kraken-fee
  scenario (taker 40 bps).
- Position: whole sleeve in or out (long/flat), no leverage.

## Strategy families and grids (the only trials)

**A. Volatility breakout (trend after vol expansion).**
- Enter long when the close exceeds the highest high of the prior N bars **and**
  realized vol over N/4 bars ÷ realized vol over 4N bars > v.
- Exit when the close falls below (highest close since entry − k × ATR(N)), or after T bars.
- Grid: N ∈ {24, 48, 96}, v ∈ {1.0, 1.3}, k ∈ {2, 3}, T ∈ {N, 2N} → **24 points**.

**B. Overshoot fade (buy capitulation).**
- Enter long when (close − close[L]) / ATR(48) < −m **and** volume z-score(48) > z.
- Exit when the close recovers half the L-bar drop, the close falls below
  entry − 3 × ATR(48), or after H bars.
- Grid: L ∈ {4, 12}, m ∈ {3, 5}, z ∈ {1, 2}, H ∈ {12, 48} → **16 points**.

Bar sizes {15m, 1h} × 40 grid points × 3 symbols = **240 trials**. The Deflated Sharpe
uses n_trials = 240.

## Selection procedure (IS only)

Walk-forward per (family, bar, symbol): 12-month train, 3-month test, 3-month step.
That gives 8 test folds over 2023-01 → 2024-12. In each fold, pick the grid point with
the best net Sharpe on the train window and record its test-window returns. The
concatenated test returns are the family's IS estimate (selection noise included).

## Pass bar

A (family, bar size) passes IS if, on the equal-weight portfolio of the 3 symbols'
walk-forward test returns:
1. annualized net Sharpe ≥ 1.0;
2. positive in ≥ 5 of 8 folds;
3. ≥ 100 trades in total;
4. Deflated Sharpe probability ≥ 0.95 (n_trials = 240);
5. mean net return per trade ≥ 2 × the average round-trip cost;
6. beats **exposure-matched buy-and-hold** (B&H scaled by the strategy's average time in
   market) on net Sharpe.

Families that pass IS freeze their final parameters by the same rule (train on the last
12 IS months, 2024) and run **once** on OOS. OOS pass:
1. net return > 0 and annualized net Sharpe ≥ 0.5;
2. max drawdown < 20%;
3. beats exposure-matched B&H on net Sharpe;
4. still net positive with costs ×2.

## Crash stress (reported for every family, pass or fail)

Strategy P&L from 1 day before to 3 days after each event: 2022-05-09 (LUNA),
2022-06-13 (Celsius/3AC), 2022-11-08 (FTX), 2024-08-05 (yen-carry unwind),
2025-10-10 (crash), 2026-10-29 (Binance hack). Any single event losing more than 5% of
the sleeve is flagged, even for a family that passes.

## What happens next

- **Pass OOS:** go to shadow trading in chorus-howl for at least 2–4 weeks, then the $100 pilot.
- **Nothing passes:** report that honestly. The Phase 1 canary still runs to measure
  execution. The next research step would be order-flow features once the websocket
  tape has 4–8 weeks of data.
