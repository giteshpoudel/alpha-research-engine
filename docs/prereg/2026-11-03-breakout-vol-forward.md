# Pre-registration: volatility-sized breakout forward test (2026-11-03)

Written before any forward data exists. Batch 1 placed this strategy in the **watch** tier
(`docs/2026-11-03-batch1-results.md`: OOS +26.9%, Sharpe 0.90, max DD −20.1%, 343 days
underwater). Its signals had been seen out of sample before, so only forward data counts.
No new parameter search: trial count stays 337.

## Design (exactly as tested in batch 1)

1. **Coins:** BTC, ETH, SOL, XRP, DOGE, LINK (Binance.US USDT pairs).
2. **Signals:** frozen breakout selections. BTC N=96 v=1.0 k=3 T=192; ETH N=48 v=1.3 k=3
   T=96; SOL N=24 v=1.3 k=3 T=48; XRP N=24 v=1.0 k=3 T=24; DOGE N=48 v=1.0 k=2 T=48;
   LINK N=48 v=1.0 k=3 T=48.
3. **Sizing:** at entry, position = (budget / 6) × min(1, 0.80 / realized vol), with
   realized vol = annualized standard deviation of the last 720 hourly returns. The size
   is fixed until exit. Budget $100, so at most $16.67 per coin, up to 6 coins at once,
   never more than $100 in total.
4. **No drawdown brake** (batch 1 tested none).
5. **Costs per side:** BTC 5, ETH 5, SOL 6, XRP 5, DOGE 7, LINK 8 bps.
6. An entry smaller than the exchange minimum ($10) is skipped and journaled. The ideal
   sleeve still counts it, so skipped entries show up as a gap between shadow and ideal.

## Forward test

Shadow mode in chorus-howl (`--strategy breakvol --run breakvol --quote USDT`) from its
first start, alongside v1, v2 and XRP/DOGE.

## Rules fixed now

**Promote to candidate** after ≥ 3 months and ≥ 15 entries, if the ideal sleeve passes the
framework's candidate checks on forward data: return > 0, Sharpe ≥ 0.5, budget max
drawdown ≤ 15%, max drawdown ≤ half that of equal-weight buy-and-hold of the six coins
over the same period, and no underwater stretch > 182 days (cannot fail before then).

**Go live small** only after promotion, ≥ 3 weeks of shadow fills within 30% of ideal,
no kill-switch trips caused by the strategy or loop, and the user's approval of the
pilot-limit changes it needs (6 coins, up to 6 positions). The user starts live mode.

**Retire** at any point if the ideal sleeve drawdown reaches 20%, or after 3 months if
ideal return ≤ 0.

## Note added 2026-11-12: internet outage

From 2026-11-09 06:21 to 2026-11-12 05:25 UTC the Mac running the shadow runs had no
internet. All five shadow runs tripped their kill switch on stale market data at 06:25 UTC
and stayed paused until the user approved clearing them on 2026-11-12 22:27 UTC (about 3.7
days). Nothing was held, so nothing was sold. How it's handled, decided after the fact and
on the cautious side:

- The trip isn't counted as a "kill-switch trip caused by the strategy or the loop", which
  is the wording above; trips for any other reason still count.
- The paused days don't count toward the minimum shadow period (3 weeks, or 3 months for
  watch-tier runs). Entries the ideal backtest takes while a run is paused have no shadow
  fill to compare against.
- Order-book data from those 3 days is missing and can't be refilled; the order-book test
  waits about 3 days longer.
