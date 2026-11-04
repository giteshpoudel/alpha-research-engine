# Old research loop retired (2026-11-03)

The original alpha-research automation was stopped (launchd jobs unloaded, data kept):

| Job | What it did |
|---|---|
| batch | Optuna strategy search every 30 min on 8 workers |
| optimizer | Continuous LLM-driven optimizer (Kimi API); 0 of its last 25 proposals adopted |
| paper | Hourly paper trading of mean reversion / momentum (retired by research 2026-09-29) |
| carry, perpmark | Funding-carry paper trading and Hyperliquid mark snapshots (needs perps; not on Binance.US) |
| retune, signaleval, report | Monthly retune, daily signal evaluation and daily report for the old loop |
| pipeline, hackernews | Reddit / RSS / StockTwits / Hacker News sentiment ingestion |

Already stopped earlier: batch5m, micro, the port-8000 dashboard.

**Kept running:** `stack` (Colima + ClickHouse watchdog), `wscollector` (order book and
trades for order-flow research), `backfill` (daily OHLCV and funding top-up).

Why: the old strategies were retired by pre-registered research; the batch job and optimizer
were the heaviest load on the Mac mini, which crashed under memory pressure the same day;
the optimizer made paid API calls without adopting anything. Research now runs as
pre-registered batches; trading runs in chorus-howl.

Restart any job with `launchctl load ~/Library/LaunchAgents/com.alpha-research.<job>.plist`.
