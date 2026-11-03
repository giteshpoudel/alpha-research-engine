# Pre-registration: breakout v2 forward test (2026-11-03)

Written before any v2 result exists. The design was chosen **after** seeing the
exploratory OOS result (`docs/2026-11-03-oos-breakout-results.md`), so no historical
window can validate it. The evidence is forward data only. Trial count for any future
Deflated Sharpe: 242.

## Changes from v1 (and only these)

1. **Symbols: ETH and SOL only.** BTC (N=96) lost OOS (−8.5%, Sharpe −0.28).
2. **Parameters unchanged** (frozen 2024 selection, not re-tuned on OOS):
   ETH N=48 v=1.3 k=3 T=96; SOL N=24 v=1.3 k=3 T=48.
3. **Drawdown brake.** The OOS failure was a 22.8% drawdown, so new entries pause when
   the ideal two-symbol sleeve is ≥ 10% below its trailing 30-day (720-bar) peak.
   Open positions still exit by their own rules. The brake lifts automatically when the
   drawdown from the trailing peak is back under 10% (or the old peak rolls out of the
   window). The sleeve equity is the research execution model (next-open fills,
   base costs) computed from 1h bars, so the brake is deterministic and the same in the
   live loop and in the report.
4. **Sizing:** $40 per position, at most 2 (both symbols), inside the $100 pilot limits.

## Forward test

Shadow mode in chorus-howl (`--strategy breakout2 --run breakout2`), from its first start.
The v1 shadow keeps running alongside as a comparison.

## Rules fixed now

**Go live small (learning trades)**, after ≥ 3 weeks of shadow, if all hold:
- shadow fills within 30% of ideal P&L on the same entries (execution works);
- no kill-switch trip caused by the strategy or the loop;
- ideal return over the period > −10%.
Live size starts at the same $40 per position under the existing $5 daily / $25 total
stops. This is a learning pilot, not adoption. The user starts live mode.

**Scale beyond the pilot** only after ≥ 3 months of forward data with ≥ 15 entries, if:
- ideal return > 0 and above exposure-matched buy-and-hold over the same period;
- mean ideal return per entry ≥ 2× round-trip cost;
- max drawdown of the ideal sleeve < 15%;
- live fills within 30% of ideal.
Scaling itself is decided with the user and written up before it happens.

**Retire** at any point if the ideal sleeve drawdown reaches 20%, or after 3 months if
ideal return ≤ 0.
