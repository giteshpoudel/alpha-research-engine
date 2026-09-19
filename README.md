# Alpha Research Engine

An autonomous quantitative research and portfolio analysis system for crypto markets. It ingests market time-series and social sentiment, scores sentiment with local embeddings, computes rolling signal metrics, and backtests trading strategies with strict train/test separation — all on local infrastructure you control.

> **Status:** Active development. Phases 1–5 complete (storage, sentiment pipeline, backtesting, LangGraph research agents, meta-learning) plus a monitoring dashboard, paper-trading forward simulation, a sentiment signal-evaluation harness, and a rolling re-tune feedback loop. Research/backtesting/paper-trading only — no live exchange execution.

## What it does

- **Ingests** crypto price data (OHLCV + funding rates, 2022–present) and social sentiment (StockTwits streams, crypto news RSS) into local databases on a schedule
- **Scores** every post's sentiment (−1…+1) with embedding-distance against bullish/bearish anchors, using a local Ollama model — deterministic, free, no API keys
- **Aggregates** rolling metrics (5m/1h/24h): sentiment polarity index, mention velocity, engagement spikes per ticker
- **Backfills** historical social/news discussion (Hacker News via Algolia, archived to 2006, with real engagement) to extend sentiment analysis before live collection began
- **Backtests** strategies on years of data with a strict in-sample/out-of-sample split, and stores results + equity curves. Mean Reversion is the active edge; Funding-Rate Carry and Sentiment-Momentum are retained for research only (see Notes)
- **Forward-tests** the tuned Mean Reversion strategy on live data as a paper-trading account (independent $1 sleeve per symbol), surfaced in a dashboard view
- **Automates research:** Optuna walk-forward tuning, a LangGraph risk/macro/report agent pipeline that also appends deterministic paper-trading and sentiment-signal sections, and a monitoring dashboard

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
| Sentiment data | StockTwits, RSS, Hacker News/Algolia (public, unauthenticated) |

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

**Paper trading** (independent $1 sleeve per symbol; forward step is idempotent):

```bash
python -m src.paper.runner --replay --from 2025-01-01   # rebuild history
python -m src.paper.runner --once                       # hourly top-up + step
```

**Feedback loop** (rolling re-tune on a 36-month IS window with a 7-day embargo; monthly):

```bash
python -m src.meta_learning.retune            # publish a new param version
```

**Historical sentiment backfill** (Hacker News via Algolia; free and archived to 2006):

```bash
python -m src.ingestion.hackernews --start 2024-01-01   # fetch + store posts
python -m src.ingestion.aggregator --bucket 1h --start 2024-01-01 --end 2026-09-19  # metrics
```

**Signal evaluation** (does the social/news feed predict forward returns?):

```bash
python -m src.research.signal_eval             # print + store to ClickHouse
python -m src.research.signal_eval --no-store  # print only
```

**Dashboard** (backtesting, paper-trading, and signal views):

```bash
python -m src.dashboard            # http://127.0.0.1:8000
```

**Tests:** `python -m pytest tests/ -v` (177 tests; integration tests need the Docker stack up).

## How it works

**Guardrails first** (from `docs/PROJECT_BLUEPRINT.MD`): strict in-sample (2022–2024) vs out-of-sample (2025–now) separation — no tuning on test data; sentiment is timestamped at publication time, never ingestion time, so backtests can't see the future; every pipeline stage is idempotent and safe to re-run.

**Sentiment scoring** is embedding distance, not an LLM call: each post is embedded once, and its score is `cos(post, bullish_anchors) − cos(post, bearish_anchors)` clamped to [−1, +1]. Deterministic, fast, free, and the embeddings double as searchable vectors in Qdrant.

**Backtest results** are content-addressed: a run's ID is a hash of strategy + symbol + params + window + dates, so re-running collapses to the same row. Equity curves are stored per run for charting.

## Project layout

```
src/
  ingestion/    # pollers (reddit/rss/stocktwits), scorer, aggregator, backfill
  backtesting/  # data layer, strategies, engine, runner
  meta_learning/# Optuna tuner, causal allocator, OOS comparison
  research/     # LangGraph agents + deterministic collectors + daily report
  dashboard/    # FastAPI monitoring app (backtesting + paper trading)
  paper/        # paper-trading forward simulation (executor, store, runner)
config/         # launchd plists for pipeline, backfill, and paper trading
docs/           # blueprint + design specs + implementation plans
tests/          # 141 tests: unit + integration against live services
```

## Roadmap

- [x] **Phase 1** — Storage infrastructure (ClickHouse, Qdrant, schemas)
- [x] **Phase 2** — Sentiment pipeline (ingestion, scoring, time-bucket metrics)
- [x] **Phase 3** — Backtesting framework (3 strategies, IS/OOS, benchmark metrics)
- [x] **Phase 4** — Multi-agent research engine (LangGraph): risk, macro/sentiment, report agents
- [x] **Phase 5** — Meta-learner: Optuna walk-forward tuning, strategy allocator, OOS comparison
- [x] **Dashboard** — web UI for backtest results and paper trading
- [x] **Paper trading** — forward simulation of tuned Mean Reversion on live data
- [x] **Signal evaluation** — predictive IC / event study for the social feed
- [x] **Feedback loop** — rolling re-tune (versioned params) + per-symbol allocation gating
- [ ] **Live trading** — order routing / execution (out of scope for now)
- [~] **Historical social dataset** — Hacker News backfill shipped (2014+); historical tweets/Reddit still blocked (paid/restricted)

## Notes & limitations

- Binance.com is geo-restricted in some regions; the backfill uses Binance.US (spot) and Hyperliquid (funding). Funding history starts at each coin's Hyperliquid listing (BTC ≈ May 2023).
- Funding-rate arb assumes zero basis risk (no perp price series) — documented approximation.
- Sentiment history is limited to when the pipeline started running; see the dataset roadmap item.
- **Sentiment signal is weak and treated as research/monitoring only.** Polarity is capped by the embedding model (`nomic-embed-text` maps bullish and bearish crypto text to nearly the same region — anchor cosine 0.83, hand-written bull/bear probes span only ±0.13). Predictive IC is small (24h mean polarity vs 24h forward return: +0.13, p=0.04); contrastive anchors and per-ticker normalization were worse. Attention features (`post_count`) show a small contrarian IC (−0.12 at 4h, p=0.004) but an entry/exit overlay on Mean Reversion was a no-op. Funding-Rate Carry loses to fees.

## Disclaimer

For research and educational purposes only. Nothing here is financial advice. Trading crypto involves substantial risk of loss.

## License

[MIT](LICENSE)
