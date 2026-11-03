# Pre-registration: XRP/DOGE breakout forward test (2026-11-03)

Written before any forward data exists. XRP and DOGE were picked **after** seeing the
alt-coin OOS result (`docs/2026-11-03-alt-breakout-results.md`: XRP +93%, DOGE +42%,
LINK −31%, portfolio retired on a 26.5% drawdown). That choice is post-hoc, so no
historical window can validate it. The evidence is forward data only. No new parameter
search: trial count stays 314.

## Design (and only this)

1. **Symbols: XRP and DOGE** (Binance.US USDT pairs). LINK is dropped because it lost OOS.
2. **Parameters frozen** from the pre-registered 2024 selection, not re-tuned on OOS:
   XRP N=24 v=1.0 k=3 T=24; DOGE N=48 v=1.0 k=2 T=48.
3. **Costs per side** as pre-registered: XRP 5 bps, DOGE 7 bps.
4. **Drawdown brake**, same as breakout v2: new entries pause while the ideal XRP/DOGE
   sleeve is ≥ 10% below its trailing 720-bar (30-day) peak. Open positions exit by
   their own rules. This sleeve is separate from the ETH/SOL v2 sleeve.
5. **Sizing:** $25 per position, at most 2 (both symbols). Shadow only; the sizing is
   chosen so that v2 and this test could share a $100 budget at 4 × $25 later.

## Forward test

Shadow mode in chorus-howl (`--strategy altbreak --run altbreak`), from its first start.
Runs alongside v2; neither changes the other.

## Rules fixed now (same as v2)

**Go live small (learning trades)** after ≥ 3 weeks of shadow, if all hold:
- shadow fills within 30% of ideal P&L on the same entries;
- no kill-switch trip caused by the strategy or the loop;
- ideal return over the period > −10%.
Going live also needs the user's approval of any pilot-limit change it requires
(e.g. 4 positions × $25 across v2 and this test). The user starts live mode.

**Scale beyond the pilot** only after ≥ 3 months of forward data with ≥ 15 entries, if:
ideal return > 0 and above exposure-matched buy-and-hold; mean ideal return per entry
≥ 2× round-trip cost; ideal sleeve max drawdown < 15%; live fills within 30% of ideal.
Scaling is decided with the user and written up first.

**Retire** at any point if the ideal sleeve drawdown reaches 20%, or after 3 months if
ideal return ≤ 0.
