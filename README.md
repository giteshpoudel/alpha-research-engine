# Alpha Research Engine

An autonomous quantitative research and portfolio analysis system for crypto markets. It ingests market time-series and social sentiment, scores sentiment with local embeddings, computes rolling signal metrics, and backtests trading strategies with strict train/test separation — all on local infrastructure you control.

> **Status:** Active development. Phases 1–3 complete (storage, sentiment pipeline, backtesting framework). Research/backtesting/paper-trading only — no live exchange execution.

## What it does

- **Ingests** crypto price data (OHLCV + funding rates, 2022–present) and social sentiment (StockTwits streams, crypto news RSS) into local databases on a schedule
- **Scores** every post's sentiment (−1…+1) with embedding-distance against bullish/bearish anchors, using a local Ollama model — deterministic, free, no API keys
- **Aggregates** rolling metrics (5m/1h/24h): sentiment polarity index, mention velocity, engagement spikes per ticker
- **Backtests** strategies (Mean Reversion, Funding-Rate Carry, Sentiment-Momentum) on years of data with a strict in-sample/out-of-sample split, and stores results + equity curves for analysis
- **Designed for what's next:** Optuna-based parameter tuning, LangGraph research agents, and a monitoring dashboard (see Roadmap)

## Architecture

```
 StockTwits   RSS news                     Binance.US / Coinbase / Hyperliquid
     │            │                                    │
     ▼            ▼                                    ▼
 ┌─────────────────────┐                    ┌──────────────────┐
 │  ingestion pipeline │                    │  price backfill  │
 │  (5-min schedule)   │                    │  (resumable)     │
 └─────────┬───────────┘                    └────────┬─────────┘
           ▼                                         ▼
      ┌─────────────────────────────────────────────────┐
      │                 ClickHouse (Docker)              │
      │  sentiment_posts · sentiment_metrics             │
      │  ohlcv · funding_rates                           │
      │  backtest_runs · backtest_equity                 │
      └─────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
      Ollama embeds      time-bucket           backtest engine
      (nomic-embed)      aggregator            (VectorBT)
           │                    │                    │
           ▼                    ▼                    ▼
      Qdrant (Docker)    polarity/velocity      Sharpe · Sortino · MaxDD
      768-dim vectors    metrics per ticker     Calmar · Win Rate
```

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Time-series DB | ClickHouse (Docker) |
| Vector DB | Qdrant (Docker) |
| Embeddings | Ollama `nomic-embed-text` (local, 768-dim) |
| Backtesting | VectorBT |
| Price data | Binance.US, Coinbase, Hyperliquid (public APIs) |
| Sentiment data | StockTwits, RSS (public, unauthenticated) |

## Quickstart

**Prerequisites:** Docker runtime (Colima/Docker Desktop), [Ollama](https://ollama.com), Python 3.11+.

```bash
git clone <repo-url> && cd alpha-research-engine
cp .env.template .env                      # defaults work out of the box
ollama pull nomic-embed-text

docker compose up -d                       # ClickHouse + Qdrant

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.ingestion.schemas            # create tables + collection (idempotent)
```

**Load historical data** (2022–present, ~500k candles + ~340k funding records, ~15 min):

```bash
python -m src.ingestion.backfill           # all 12 tickers; resumable, safe to re-run
```

**Run the sentiment pipeline:**

```bash
python -m src.ingestion.pipeline --once        # single cycle
python -m src.ingestion.pipeline --interval 300  # continuous (5-min cycles)
```

**Run backtests:**

```bash
python -m src.backtesting.runner --strategy all --symbols all --window ALL
python -m src.backtesting.runner --strategy mean_reversion --symbols BTC,ETH --window OOS
```

**Tests:** `python -m pytest tests/ -v` (96 tests; integration tests need the Docker stack up).

## How it works

**Guardrails first** (from `docs/PROJECT_BLUEPRINT.MD`): strict in-sample (2022–2024) vs out-of-sample (2025–now) separation — no tuning on test data; sentiment is timestamped at publication time, never ingestion time, so backtests can't see the future; every pipeline stage is idempotent and safe to re-run.

**Sentiment scoring** is embedding distance, not an LLM call: each post is embedded once, and its score is `cos(post, bullish_anchors) − cos(post, bearish_anchors)` clamped to [−1, +1]. Deterministic, fast, free, and the embeddings double as searchable vectors in Qdrant.

**Backtest results** are content-addressed: a run's ID is a hash of strategy + symbol + params + window + dates, so re-running collapses to the same row. Equity curves are stored per run for charting.

## Project layout

```
src/
  ingestion/    # pollers (reddit/rss/stocktwits), scorer, aggregator, backfill
  backtesting/  # data layer, strategies, engine, runner
config/         # launchd plist for scheduled pipeline
docs/           # blueprint + design specs + implementation plans
tests/          # 96 tests: unit + integration against live services
```

## Roadmap

- [x] **Phase 1** — Storage infrastructure (ClickHouse, Qdrant, schemas)
- [x] **Phase 2** — Sentiment pipeline (ingestion, scoring, time-bucket metrics)
- [x] **Phase 3** — Backtesting framework (3 strategies, IS/OOS, benchmark metrics)
- [ ] **Phase 4** — Multi-agent research engine (LangGraph): risk, macro/sentiment, report agents
- [ ] **Phase 5** — Meta-learner: Optuna walk-forward tuning, bandit strategy allocator, paper trading
- [ ] **Dashboard** — web UI for backtest results, paper trading, and KPIs
- [ ] **Historical social dataset** — extends sentiment backtests before 2026

## Notes & limitations

- Binance.com is geo-restricted in some regions; the backfill uses Binance.US (spot) and Hyperliquid (funding). Funding history starts at each coin's Hyperliquid listing (BTC ≈ May 2023).
- Funding-rate arb assumes zero basis risk (no perp price series) — documented approximation.
- Sentiment history is limited to when the pipeline started running; see the dataset roadmap item.

## Disclaimer

For research and educational purposes only. Nothing here is financial advice. Trading crypto involves substantial risk of loss.

## License

[MIT](LICENSE)
