# Project Blueprint: Quant Research & Adaptive Portfolio Engine

## 1. Project Goal & Overview
Build an autonomous quantitative research and portfolio analysis system. The system ingests market time-series data, historical/live news feeds, and social sentiment (Twitter/X) to backtest strategies, generate market intelligence, and continuously adjust trading algorithm parameters based on historical performance and sentiment signals.

> **Operational Scope:** Pure research, backtesting, paper trading, and strategy auto-tuning. No live exchange execution.

---

## 2. Recommended Tech Stack & System Infrastructure

| Layer | Recommended Technology | Justification |
| :--- | :--- | :--- |
| **Language & Environment** | Python 3.11+ | Native compatibility with ML, quantitative backtesting, and LLM frameworks. |
| **Time-Series Database** | **ClickHouse** (via Docker) | Unmatched performance for column-store queries on high-frequency market data & sentiment metrics. |
| **Vector Database** | **Qdrant** (via Docker) | Lightweight, fast vector search for semantic query matching over historical tweets and news feeds. |
| **Data Ingestion** | `httpx` + `websockets` + `Redpanda` (optional) | High-throughput async data ingestion pipelines. |
| **Backtesting Engine** | **VectorBT** or **Backtrader** | VectorBT for lightning-fast matrix backtests; Backtrader for event-driven simulation. |
| **Agent Framework** | **LangGraph** | Explicit state management and graph routing for deterministic multi-agent research loops. |
| **Hyperparameter Optimization** | **Optuna** | Efficient Bayesian optimization for strategy parameter tuning. |
| **LLM Models** | Kimi (Moonshot API) / Local Ollama (DeepSeek) | High-context understanding for long-form research and quick local classification. |

---

## 3. Order of Development Plan

### Phase 1: Storage Infrastructure & Data Ingestion Modules
- [ ] Spin up ClickHouse and Qdrant database containers using Docker Compose.
- [ ] Implement data schema for market time-series (OHLCV, funding rates) and sentiment payloads.
- [ ] Build historical data import scripts for:
  - Crypto price data (Binance / Hyperliquid archive or Birdeye).
  - Historical Twitter/X feeds and crypto news datasets.
- [ ] Implement a noise-reduction pipeline: filter bot spam, deduplicate tweets, and extract crypto tickers/entities.

### Phase 2: Sentiment Processing & Signal Generation
- [ ] Build a sentiment scoring worker (using embedding distance or lightweight LLM classifier) to quantify social posts (-1.0 to +1.0).
- [ ] Implement a time-bucket aggregation pipeline (5m, 1h, 24h) computing rolling metrics:
  - Sentiment Polarity Index
  - Volume/Social Engagement Spikes
  - Ticker Mention Velocity

### Phase 3: Core Backtesting Framework
- [ ] Implement foundational quantitative trading strategies (e.g., Sentiment-Adjusted Momentum, Mean Reversion, Funding Rate Arbitrage).
- [ ] Integrate VectorBT/Backtrader to evaluate strategy performance on historical data (2022–Present).
- [ ] Establish benchmark metrics: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, and Win Rate.

### Phase 4: Multi-Agent Research Engine (LangGraph)
- [ ] **Portfolio Risk Agent:** Analyzes exposure, current PnL metrics, and correlation across assets.
- [ ] **Macro & Sentiment Agent:** Summarizes market trends, tweet sentiment divergence, and news catalysts.
- [ ] **Report Generator Agent:** Synthesizes outputs into formatted daily research reports.

### Phase 5: Meta-Learner & Auto-Tuning Feedback Loop
- [ ] Integrate Optuna to run parameter optimization loops across moving windows (e.g., Walk-Forward Optimization).
- [ ] Implement a multi-armed bandit / strategy allocator that shifts weight toward strategies performing best in current market regimes.
- [ ] Run paper-trading simulations comparing static parameter strategies vs. adaptive (tuned) strategies over historical out-of-sample data.

---

## 4. Best Practices & Guardrails

1. **Strict Data Separation:** Always maintain strict isolation between **In-Sample** (training/tuning) and **Out-of-Sample** (validation) historical data to prevent lookahead bias and overfitting.
2. **Deterministic Backtesting:** Ensure sentiment scores are timestamped strictly at their *publication time* (or ingestion time), never allowing future social data to influence historical backtest decisions.
3. **Graceful Rate-Limit Handling:** Use exponential backoff for all external API queries (Twitter, LLM APIs, price indexers).
4. **Idempotent Data Pipelines:** Ensure historical data loaders can be re-run safely without creating duplicate database entries.
5. **Modular Architecture:** Keep data ingestion, backtesting logic, and agent orchestration in separate decoupled Python packages inside `src/`.