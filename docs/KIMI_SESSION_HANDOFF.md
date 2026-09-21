# Kimi Session Handoff → opencode

Source: Kimi Code session `wd_alpha-research-engine_da2c5a4e1791` (session id
`d39eddd2-40ac-4310-85e0-44cc71aa1bf5`), 23 prompts, ended `completed`.

Purpose: carry the full working context of the Kimi build session into opencode.
Reference this file with `@docs/KIMI_SESSION_HANDOFF.md` when starting work. The
authoritative running ledger is `.superpowers/sdd/progress.md` — read it for
per-task detail. This doc is the condensed state.

## Project snapshot

- **Repo:** `alpha-research-engine` — autonomous quant research & portfolio engine for crypto. Research/backtesting/paper-trading only, **no live exchange execution**.
- **Branch:** `main`, working tree clean. **Local only — not pushed to GitHub** (user's explicit rule: push only when asked).
- **Tests:** 126/126 green (`python -m pytest tests/ -v`; integration tests need the Docker stack up).
- **Status:** Phases 1–5 built and merged, dashboard live, Phase 4 LangGraph agents merged, first daily report generated, ops hardened. Roadmap below is what remains.

## Stack

| Layer | Tech |
| --- | --- |
| Language | Python 3.11 (`.venv` at repo root) |
| Time-series DB | ClickHouse (Docker, `alpha-clickhouse`) |
| Vector DB | Qdrant (Docker, `alpha-qdrant`) |
| Embeddings | Ollama `nomic-embed-text` (768-dim, local/free) |
| Backtesting | VectorBT |
| Tuning | Optuna (walk-forward) |
| Agents | LangGraph |
| LLM report | Kimi/Moonshot primary, Ollama `deepseek-r1:8b` fallback |
| Dashboard | FastAPI + Jinja + Chart.js |

## How to run

```bash
source .venv/bin/activate
docker compose up -d                              # ClickHouse + Qdrant (Colima)
python -m src.ingestion.schemas                   # idempotent DDL
python -m src.ingestion.backfill                  # resumable price/funding backfill
python -m src.ingestion.pipeline --once           # one sentiment cycle
python -m src.backtesting.runner --strategy all --symbols all --window ALL
python -m src.meta_learning.tuner                 # Optuna walk-forward (IS only)
python -m src.meta_learning.compare               # static vs tuned vs allocator (OOS)
python -m src.research.daily_report [--date YYYY-MM-DD]   # daily report
python -m src.dashboard                           # dashboard at http://127.0.0.1:8000
```

## What's built

- **Phase 1 — Storage:** `docker-compose.yml` (loopback-bound ports), `src/ingestion/schemas.py` (ClickHouse tables + Qdrant collection, idempotent).
- **Phase 2 — Sentiment:** pollers (`stocktwits.py`, `rss.py`, `reddit.py`), embedding scorer, time-bucket aggregator (`sentiment_metrics` 5m/1h/24h).
- **Phase 3 — Backtesting:** data layer with strict half-open window discipline, strategies `mean_reversion`, `sentiment_momentum`, `funding_arb`; engine + metrics (Sharpe/Sortino/MaxDD/Calmar/WinRate); `backtesting.runner` stores content-addressed runs + equity curves.
- **Phase 5 — Meta-learning:** `meta_learning/tuner.py` (Optuna walk-forward, IS only), `compare.py` (static vs tuned vs causal allocator on OOS), `allocator.py`; `tuned_params` table.
- **Dashboard:** `src/dashboard/` — overview, strategy pages, run pages, JSON API, `APP_ENV=dev|prod` toggle; live at `http://127.0.0.1:8000` (logs `data/dashboard.log`).
- **Phase 4 — Research agents (LangGraph):** `src/research/` — deterministic collectors → parallel risk/macro agents → report agent → store. Report stored in `research_reports` and `data/reports/YYYY-MM-DD.md`. LLM behind one OpenAI-compatible client (`llm.py`): Kimi primary, Ollama fallback.
- **Ops:** ClickHouse heavy-log disable + memory caps persisted in `config/clickhouse/` (mounted); launchd plists `config/com.alpha-research.pipeline.plist` (5-min) and `config/com.alpha-research.backfill.plist` (6:17am top-up).

## Where things stand

- Data: ~499,664 OHLCV candles (12 tickers, 1h+1d, 2022→now) + ~337,865 funding records; OHLCV current through today; sentiment accumulating every 5 min.
- First real report: `data/reports/2026-09-15.md` (generated via Ollama fallback because `MOONSHOT_API_KEY` was not set).
- Key finding from Phase 5: **Mean Reversion tuning helps OOS on 8/12 symbols** (BNB, LINK flipped profitable; DOT is the overfit cautionary case). **Funding-rate arb is retired from the active roster** — fee drag > carry at 0.1% fees (commit `23cdd2c`; code and history retained, excluded from future tune/compare).
- Disk-full root cause was ClickHouse `trace_log`/`text_log` (~83 GiB), now disabled via mounted config; reclaimed ~94 GiB.

## Guardrails (do not violate)

1. Strict IS (2022–2024) / OOS (2025–now) separation; never tune on OOS.
2. Sentiment timestamped at publication time, never ingestion time (no lookahead).
3. All pipelines idempotent / safe to re-run.
4. Don't push to GitHub unless the user explicitly asks.

## Open follow-ups (from `.superpowers/sdd/progress.md`, Phase 4 final review)

- **Signal-state semantics:** the Risk Agent's "signal states" describe entry-trigger conditions with default params, not tuned held positions — substantive fix if exposure accuracy matters.
- Dashboard `APP_ENV=prod` DB-name inconsistency (`queries.py` `database_name()` uses unprefixed `CLICKHOUSE_DB`) — fix before the prod slot is used.
- Hoist the 12 identical drawdown queries in `collectors.py`.
- Normalize `OLLAMA_HOST` handling; add NaN guards in collectors; rename `current_dd` → `max_dd`.

## Next steps / roadmap

- [ ] **Paper-trading iteration** — tuned MR forward on live data + dashboard "Paper Trading" view (nav slots already reserved).
- [ ] **Historical social dataset** — extend sentiment backtests before 2026 (Reddit was blocked by 403; StockTwits/RSS used instead).
- [ ] **Push to GitHub** — Phase 4 + first report is the showcase milestone; README/LICENSE already added. User wants open-source cleanup + README for profile.
- [ ] Add `MOONSHOT_API_KEY` to `.env` to switch reports from Ollama fallback to Kimi.

## Environment notes

- `.env` present (git-ignored). Current keys include `CLICKHOUSE_*`, `QDRANT_*`, `OLLAMA_HOST`, `TWITTER_BEARER_TOKEN`, `DEEPSEEK_API_KEY`, and a suspiciously-named `kimi_open_code ` (trailing space) — verify whether this is intended to be `MOONSHOT_API_KEY`.
- Colima VM is 4GB/2CPU; ClickHouse was OOM-killed once at 2GB. Consider bumping to 6–8GB for heavy tuning runs.
- Model routing for this repo's opencode sessions (global config): `plan` → `moonshotai/kimi-k3`, `build` → `deepseek/deepseek-v4-flash`.
