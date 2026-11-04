# Pre-registration: maker-first execution experiment (2026-11-03)

Written before any maker-order data exists. This is an execution test, not a strategy test:
it measures what it costs to buy and sell on Binance.US with two order policies.

## Why

Trading costs erased most strategy edges so far. Our account pays **0% maker / 0.02% taker**
on BTC, ETH, SOL, XRP, DOGE and LINK (USDT pairs, checked 2026-11-03). A market order pays
the taker fee plus half the spread: about 2.2 bps per side on ETH, 2.8 on SOL and XRP, 5.4 on
DOGE, 5.7 on LINK. A resting maker order pays neither, but may not fill, and tends to fill
when the price moves against it (adverse selection).

## Policies

- **Taker:** a market order.
- **Maker-first:** a post-only (`LIMIT_MAKER`) order at our side's best price (best bid for
  a buy, best ask for a sell). If the best price moves away, cancel and re-post at the new
  best price, at most once every 15 s. After 60 s, cancel and send a market order for any
  remainder.

## Experiment

- chorus-howl live canary, run by the user: `python -m chorus_howl.exec_test --mode live`.
- **60 round trips** of about $12: buy, hold 3 minutes, sell, pause 10 s. Policies alternate
  every round trip; coins rotate BTC → ETH → SOL in pairs, so each coin gets 10 round trips
  with each policy (20 sides per coin and policy, 60 sides per policy).
- Coins: BTC, ETH, SOL only (the live allow-list). XRP, DOGE and LINK, where spreads are
  widest, need a pilot-limit change first and are out of scope here.
- Existing live limits and stops apply ($5 daily, $25 total, kill switch). Expected total
  cost: under $1.

## Measurements (per side)

- **Shortfall (bps)** = s × (average fill − mid at decision) / mid at decision × 10,000, plus
  fees in bps of notional; s = +1 for buys, −1 for sells. Lower is better.
- Time to complete, share of quantity filled as maker, how often the fallback was needed.
- Mid-price change 60 s after completion, signed in our direction (adverse selection),
  reported for both policies.

## Decision rule (fixed now)

Adopt maker-first as the default for live strategy orders if both hold over the 60
maker-first sides:
1. Mean shortfall is at least **1.0 bps lower** than the 60 taker sides, overall, and not
   worse than taker by more than 1.0 bps on any single coin.
2. At least **90%** of maker-first sides complete within 75 s.

Otherwise keep market orders. Results are reported either way.

## Known limitation

Canary orders go in at random times. Strategy entries happen right after a breakout, when the
price is running away from a resting buy, so maker fill rates there will be lower. If adopted,
strategy orders log the same measurements and the rule is re-checked after 20 strategy sides.
