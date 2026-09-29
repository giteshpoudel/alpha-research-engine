# Intraday crypto & funding carry — research + build (2026-09-29)

Status: research note + first implementation (delta-neutral funding carry).
Audience: next session / future agent. Supersedes the "10%/day intraday" premise
with an evidence-based set of realistic edges and a roadmap.

## 0. TL;DR

- Our 5m mean-reversion "edge" is not real: it was killed by execution costs
  (realistic per-side cost ~3-5 bps; breakeven was ~3-5 bps) and by a single
  tail event (2025-10-10 crash). **Retire 5m scalar scalping.**
- The literature-supported *real* crypto edges at retail scale are, in order of
  reliability: (1) **delta-neutral funding/basis carry**, (2) **market making /
  liquidity provision**, (3) **hourly/daily ML or cross-sectional momentum**.
- Built and evaluated (1): **~+3.4% to +14.4% APY net**, market-neutral, holding
  the carry continuously. Hourly timing of funding destroys it (see §4).
- Realistic portfolio expectation for an agent that rotates capital across coins
  using our current strategies: **low single digits to low double digits % per
  year**, with meaningful variance — not 10%/day.

## 1. Method and limits

Searched via DuckDuckGo HTML, Springer, SSRN, Cornell (Easley/O'Hara), arXiv and
AIMS Press. Full text retrieved for: Cohen (2023, AIMS Math); Zhu & Cai (2026,
arXiv 2609.04917). Others via abstract/snippet only (SSRN/Springer PDFs and
search engines later blocked by CAPTCHA). Claims below are tagged by confidence.

## 2. What has actually been shown profitable

### 2.1 Funding-rate / basis carry (delta-neutral) — highest reliability
Long spot + short perp harvests the funding rate with ~no directional exposure.

- Hyperliquid funding, our universe, 2023-05 → 2026-09 (hourly), average
  annualised funding: BTC 14.2%, ETH 14.4%, DOGE 17.1%, LINK 15.7%, LTC 12.8%,
  SOL 12.3%, XRP 9.2%, ADA 8.8%, AVAX 8.1%, DOT 7.0%, BNB 3.6%.
- Real desk yields of ~5–50% APY are commonly cited; the review (Zhu & Cai 2026)
  treats perp funding as a distinct cash-flow stream requiring its own modelling.

### 2.2 Market making / liquidity provision — strongest structural edge
- SSRN 5066176 builds auto market-making for perps with an alpha called
  **Bar Portion**, "robust across cryptocurrencies" (abstract-level).
- Easley/O'Hara et al. (2024, Cornell) show **microstructure metrics (liquidity,
  price discovery) predict** price dynamics relevant to market making.
- Fits us: **Binance.US maker fee is 0%**. Our earlier naive-limit test failed
  from adverse selection — the literature's fix is inventory skew + microprice +
  quote fade, not a fixed offset.

### 2.3 Intraday ML on hourly bars — modest but documented
- Cohen (2023): polynomial autoregression, 6mo train / 6mo live-like test, best
  bars **61–67 min**. BTC +15.6% (vs −44.8% B&H), ETH +17.0% (vs −33.6%), BNB
  +9.3%, ADA +4.3% over six months (~0.08%/day).
- Springer (Rev Quant Fin Acc): oscillators at 5–180 min; **RSI best**, beat B&H
  on five majors. Note: these are hourly-to-3h, not 5m scalps.

### 2.4 Also documented
Cross-sectional momentum and attention (Liu & Tsyvinski 2021, RFS); day-of-week
effect (Caporale & Plastun 2019); social sentiment (Garcia & Schweizer 2015).

## 3. The evidence verdict (Zhu & Cai 2026, arXiv 2609.04917)

Across ML/LLM/RL/agents: **"no general AI architecture is shown to deliver
persistent, cross-regime, capacity-aware net alpha."** Progress is upstream
(prediction/representation); failure happens in *translation*: look-ahead,
selection multiplicity, cost, capacity, execution delay. Prescriptions:
point-in-time data/models, decision-aligned objectives (not forecast error),
joint signal–portfolio–execution evaluation, controlled adaptation, prospective
validation, governance proportional to authority.

## 4. Implementation: funding carry (built this session)

- `src/meta_learning/carry.py`: `carry_metrics`, `evaluate`, CLI.
- Engine `run_funding_backtest` gained **hysteresis** (`exit_threshold`), so a
  position is entered above the entry threshold and only exited below the exit
  threshold (backwards compatible; default exit = entry).
- Fees: per-leg cost = spot taker (Binance.US 2 bps) + perp taker
  (`PERP_TAKER_FEE`, Hyperliquid default 4.5 bps); `fee_for_symbol` still used.

Net annualised by rule (funding data = Hyperliquid; per-leg fee 6.5 bps):

| rule | BTC | ETH | DOGE | LINK | LTC | SOL | XRP | DOT | ADA | AVAX | BNB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| always-in (one entry) | 12.2 | 12.2 | 14.4 | 13.4 | 11.2 | 10.7 | 8.6 | 8.2 | 8.1 | 7.5 | 3.4 |
| enter 0 / exit −2e-5 | 5.3 | 1.0 | 5.8 | 6.2 | 3.4 | −14 | −9 | −36 | −27 | −23 | −37 |
| enter 5e-6 / exit −2e-5 | 6.2 | 2.3 | 7.5 | 7.4 | 5.6 | −9 | −4 | −27 | −18 | −14 | −25 |

**Finding:** holding the carry continuously captures roughly the gross funding
minus a negligible entry cost; *timing* funding hour-by-hour pays flip costs and
gives up carry. The real risk of always-in is sustained negative funding, which
needs a regime guard — not hourly hysteresis.

## 5. Rigid roadmap (mapped to this repo)

Tier 1:
1. **Funding carry, productionised**: regime guard via *trailing-mean* funding
   (enter when 24h mean > 0, exit when 7d mean < 0); reconcile Hyperliquid
   funding vs venue we actually trade; track real net APY in paper.
2. **Cost-aware, point-in-time tuning**: put fees+spread+lag inside the objective
   and add a **Deflated Sharpe / trial-count penalty** so grid searches stop
   selecting on luck (currently `DEFAULT_ADOPT_MARGIN` only).
3. **Cross-sectional momentum at daily/hourly** on the 15+ Binance.US symbols
   (low turnover → costs amortise).

Tier 2:
4. **Proper market making** on Binance.US (0% maker): inventory skew, microprice,
   quote fade. Needs L2 depth + trade tape.
5. **Order-flow imbalance / microstructure alpha** (Easley): predicts short-
   horizon drift; trade hourly, not 5m.
6. **Tail/regime gating**: make crash exposure explicit (vol regime, funding
   stress) instead of accidentally harvesting it.

Tier 3:
7. **Prospective paper protocol**: pre-register hypotheses; accept only on
   deflated Sharpe; keep a trial ledger.
8. **Retire 5m scalar MR** to research-only.

Data requests this implies: perp funding + mark/basis (have funding; add mark),
L2 order book, trade tape, open interest / liquidations.

## 6. Expected returns & the "agent rotates coins" question

- Carry (market-neutral): **~7–14% APY** gross-of-frictions, on the perp notional;
  annualised on total capital is lower because both legs are funded. Sustained
  negative-funding regimes can make it negative; a regime guard is required.
- Hourly/daily directional strategies: the documented edge is ~**0.05–0.1%/day**
  idealised, with drawdowns; net of realistic costs many disappear.
- So an autonomous agent rotating capital across coins on *our current* signals
  should be expected to produce **low single-digit to low double-digit % per
  year**, with real risk of flat/negative stretches — **not 10%/day**. The
  10%/day figure implies ~10^2–10^3× leverage or an edge that the public record
  does not support.

## 7. Acceptance guardrails (apply to any new strategy)

- Point-in-time data; OOS/validation-only adoption; deflated Sharpe vs trials.
- Costs inside the objective (fee + half-spread + slippage + execution lag).
- Capacity/turnover limit; reject if breakeven cost < realistic cost.
- Prospective paper period before any live discussion; live code never pushed.
