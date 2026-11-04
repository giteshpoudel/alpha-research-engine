# Strategy scoping: what to test next, and how not to run dry (2026-11-03)

Status: scoping only. Nothing here is tested yet. Each test still gets its own
pre-registration before results, and is judged by the three-tier framework
(`docs/prereg/2026-11-03-decision-framework.md`).

## 1. The problem to avoid

Testing one strategy at a time, waiting weeks for each, and ending with nothing is a
real risk. So far: 5m scalping retired, overshoot fade rejected, breakout v1 and the
alt-coin breakout retired on drawdown. Two breakout variants are in paper trading.

How to avoid running dry:

1. **Test in batches, not one at a time.** Bar-data backtests take minutes. One
   pre-registration can cover 4–5 independent ideas, with the trial count raised for
   all of them at once. A batch per week is realistic.
2. **Run paper tests in parallel.** Paper runs cost nothing. Everything that reaches
   "candidate" or "watch" goes straight into shadow, side by side.
3. **Spread across strategy types.** Breakouts, trend, rotation and reversal fail at
   different times. A working portfolio is more likely to come from 3–4 modest,
   unrelated strategies than from one star. The breakout alone is in the market only
   about 12% of the time.
4. **Learning trades don't wait for the whole pipeline.** v2 reaches its 3-week check
   around 2026-11-24. If it passes, small live trades start then, while other tests run.
5. **Be honest about the odds.** Most published edges shrink after costs and after
   publication. A batch that finds nothing is a valid result, and it costs nothing
   compared with trading something that doesn't work. Volume of good tests is the
   lever we control.

## 2. Constraints that shape the list

- **Venue:** Binance.US spot, as a US resident. **Long or flat only**: no shorting and
  no perpetual futures. Many papers are long-short; here they become long-only rotations,
  which keeps part of the effect at best.
- **Costs:** measured 2 bps fee per side with 0 slippage on BTC; research assumes
  5–8 bps per side. Doubling costs erased most of the alt-coin edge, so turnover matters.
- **Universe:** 11 Binance.US coins with 1h/1d/5m bars since 2022-01 (BTC, ETH, SOL,
  XRP, DOGE, LINK, ADA, AVAX, LTC, DOT, BNB). The tradeable subset is limited by spread
  and volume (XRP, DOGE, LINK, BTC, ETH, SOL pass today).
- **Other data we have:** Hyperliquid funding rates (2023-05 →), order-book samples,
  live trade tape, 90-day trade backfill (finishing), sentiment tables.

## 3. Candidate strategies

Evidence grades: **A** = several peer-reviewed studies, replicated outside crypto too;
**B** = peer-reviewed, crypto-only or mixed; **C** = preprints, single studies or our
own hunch.

| # | Idea | Evidence | Fits long-only spot? | Data ready | Priority |
|---|---|---|---|---|---|
| 1 | Daily trend following, volatility-scaled | A | yes | yes | **Batch 1** |
| 2 | Cross-sectional momentum rotation (hold top-k of 11 coins) | A/B | partly (long leg only) | yes | **Batch 1** |
| 3 | Volatility-managed sizing overlay on existing strategies | A | yes | yes | **Batch 1** |
| 4 | Funding-rate extremes as a spot signal (crowding) | C | yes | yes | **Batch 1** |
| 5 | ML meta-labeling of breakout signals | B/C | yes | yes | **Batch 2** |
| 6 | Gradient-boosted cross-sectional ranker (4h/daily) | B | partly | yes | **Batch 2** |
| 7 | Order-flow imbalance (hours horizon) | C | yes | 90 days only | **Batch 3** (pre-registered 2026-11-03) |
| 8 | Maker (post-only) execution for existing strategies | B | yes | live canary | **Parallel** execution project |
| 9 | Short-term reversal | B (small illiquid coins) | yes | yes | Low: we only trade liquid coins |
| 10 | Time-of-day / weekday effects | B (returns not persistent) | yes | yes | Low: use for execution timing only |
| 11 | Pairs / stat arb | B | **no** (needs shorting) | – | Skip |
| 12 | Funding carry (spot long + perp short) | A | **no** (no US perps here) | – | Skip on this venue |

### Notes per idea

1. **Daily trend following.** The most robust documented effect across asset classes.
   Volatility scaling is the main source of better risk-adjusted returns (Kim, Tse &
   Wald 2016). Crypto studies find momentum beats buy-and-hold with lower downside
   (Borgards 2021). Low turnover, so costs matter less. Best single bet to reach
   "candidate".
2. **Cross-sectional momentum.** Market, size and momentum explain crypto returns, and
   momentum is stronger among large coins (Liu, Tsyvinski & Wu 2022, *Journal of
   Finance*). Caveats: their returns are gross, and 11 coins is a small cross-section.
   Weekly rotation into the top 2–3.
3. **Volatility-managed overlay.** Cut position size when recent volatility is high. This
   targets the drawdowns that retired v1 and the alt-coin test directly. Applied to the
   breakout family and to ideas 1–2 as a pre-registered variant.
4. **Funding extremes.** Very high perp funding means crowded longs. Test whether extreme
   funding (from the Hyperliquid data we already store) predicts weaker spot returns, as
   a filter on entries. Our own hypothesis; grade C.
5. **ML meta-labeling.** The breakout's win rate is ~38%: many small losses, few big wins.
   A second model predicts which breakout signals are worth taking (López de Prado's
   meta-labeling with triple-barrier labels). It improves an existing signal rather than
   searching for a new one, which is where ML has the best odds.
6. **Gradient-boosted ranker.** Rank the 11 coins by expected next-day (or 4h) return from
   engineered features (momentum, volatility, volume, funding). Daily crypto ML studies
   report ~53–54% accuracy, ~58% on the most confident 10% (Jaquart et al. 2022). At
   1–60 minutes, costs made it unprofitable (Jaquart, Dann & Weinhardt 2021).
7. **Order flow.** Recent preprints find order imbalance predicts returns over 4–12 hours
   and that order-flow features generalise across coins. Only 90 days of trade history,
   so this leans on forward data.
8. **Maker execution.** Posting limit orders instead of crossing the spread could cut
   costs on every strategy. Our earlier naive limit-order test lost to adverse selection,
   so this is an execution experiment measured in the live canary, not a strategy.
   Our current fee tier needs checking first.

## 4. Machine learning: where it can help and where it usually fails

**What the evidence says**
- **Raw price forecasting with LSTMs** rarely beats a naive "tomorrow = today" forecast
  out of sample. A walk-forward study of 12 models (including LSTM, XGBoost and N-BEATS)
  found naive forecasts beat all of them on crypto prices. Papers that claim success
  usually rely on short test windows, segmented data or weak baselines.
- **Classification of direction** works slightly: 52–54% daily accuracy, better on
  confident predictions. That's enough for an edge only if turnover is low.
- **Tree models** (random forest, gradient boosting) are at least as good as deep nets on
  this kind of tabular data, need far less data, and are easier to check. Fischer, Krauss
  & Deinert (2019) found a random forest edge after 15 bps costs, but it was long-short,
  1-minute data, over three months of 2018.

**Why your LSTM idea was reasonable but hard**
Crypto prices have a very low signal-to-noise ratio and regimes change. An LSTM has
many parameters and ~40k hourly bars per coin, which aren't independent samples. It
learns the noise. That's the usual failure, not a flaw in the idea.

**How we'd do ML here**
- Predict **outcomes of trades** (did this signal hit its profit target before its stop?),
  not prices.
- Start with **gradient-boosted trees** (LightGBM) on engineered features; runs fine on
  the Mac mini.
- **Purged, embargoed walk-forward** validation, so overlapping labels don't leak.
- Must beat the simple rule it replaces, net of costs, out of sample, once.
- Deep models (LSTM, Transformer, time-series foundation models) only as a comparison
  after the tree baseline works, and with the extra trials counted.
- Every model configuration tried counts toward the trial count.

## 5. "Success stories"

Firms with real long-term records (Renaissance, Jump, Wintermute and other market makers)
win on infrastructure: speed, rebates, inventory management and size. That edge isn't
available to a $100–$500 retail account. Public retail stories rarely have verifiable,
cost-inclusive records, and community strategy libraries (e.g. Freqtrade) are mostly
overfit. They're fine as idea sources. Everything goes through the same pipeline.

What retail can realistically use: low-turnover trend and momentum, sitting out crashes,
and cheap execution. All three are on the list above.

## 6. Proposed schedule

| When | Work | Output |
|---|---|---|
| Week of 11-03 | Batch 1: ideas 1–4, one pre-registration, walk-forward + OOS once | Tiered results; passes go to shadow same day |
| Week of 11-10 | Batch 2: ideas 5–6 (ML), one pre-registration | Tiered results; passes go to shadow |
| ~11-24 | v2 3-week check | First small live learning trades if it passes |
| From ~11-17 | Batch 3: order flow (idea 7) once the backfill is complete | Exploratory + forward test |
| Ongoing | Maker execution experiment (idea 8) in the live canary | Measured cost per side |

The fallback if batches 1–2 find nothing: v2 and XRP/DOGE keep paper trading, ideas move
to forward-only tests, and no money is at risk in the meantime.

## Sources

- Liu, Tsyvinski & Wu (2022), "Common Risk Factors in Cryptocurrency", *Journal of Finance* 77(2): https://papers.ssrn.com/abstract=3379131
- Kim, Tse & Wald (2016), "Time series momentum and volatility scaling", *Journal of Financial Markets* (summary): https://tradingstrategy.ai/docs/learn/momentum.html
- Borgards (2021), momentum in cryptocurrencies, *N. American Journal of Economics and Finance*: https://econpapers.repec.org/RePEc:eee:ecofin:v:57:y:2021:i:c:s1062940821000590
- Jaquart, Dann & Weinhardt (2021), "Short-term bitcoin market prediction via machine learning": https://publikationen.bibliothek.kit.edu/1000150665
- Jaquart, Köpke & Weinhardt (2022), "Machine learning for cryptocurrency market prediction and trading": https://www.econbiz.de/Record/machine-learning-for-cryptocurrency-market-prediction-and-trading-jaquart-patrick/10014433737
- Fischer, Krauss & Deinert (2019), "Statistical Arbitrage in Cryptocurrency Markets": https://mdpi.com/1911-8074/12/1/31
- Baur, Cahill, Godfrey & Liu (2019), Bitcoin time-of-day effects: https://research-repository.uwa.edu.au/en/publications/bitcoin-time-of-day-day-of-week-and-month-of-year-effects-in-retu/
- Aleti & Mizrach, intraday volatility and liquidity periodicity: https://arxiv.org/pdf/2109.12142v2.pdf
- Kozlowski, Puleo & Zhou (2020), cryptocurrency return reversals: https://ideas.repec.org/a/taf/apeclt/v28y2021i11p887-893.html
- Zaremba et al. (2021), "Up or down? Short-term reversal, momentum, and liquidity": https://earsiv.medeniyet.edu.tr/items/23a96c9f-3952-4bdf-aec5-abafb3aaab70
- Bieganowski & Ślepaczuk (2026), "Explainable Patterns in Cryptocurrency Microstructure": https://arxiv.org/abs/2602.00776
- "The Quarter-Hour Effect" (2026), order imbalance in crypto futures: https://arxiv.org/pdf/2607.09426
- Order flow imbalance and bitcoin crashes: https://link.springer.com/article/10.1007/s11156-023-01148-1
- LSTM vs naive forecasts review: https://arxiv.org/html/2606.00071
- López de Prado, *Advances in Financial Machine Learning* (meta-labeling, triple barrier, purged CV), overview: https://tradingstrategy.ai/docs/learn/machine-learning.html
- Earlier internal note: `docs/2026-09-29-intraday-and-carry-research.md`
