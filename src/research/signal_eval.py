"""Sentiment signal evaluation: does the social/news feed predict forward returns?

No-lookahead discipline: a sentiment bucket is only knowable once it closes, so
every feature is aligned at ``bucket_end`` and forward returns start strictly
after it. All statistics report their sample size and flag small samples as
``insufficient`` — this is an evidence baseline, not a verdict.

Usage:
    python -m src.research.signal_eval [--bucket 1h,24h] [--horizons 1h,4h,24h]
                                       [--symbols all] [--no-store]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.backtesting.data import BUCKET_SECONDS, load_ohlcv
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.tickers import TICKER_ALIASES

FEATURES = ("weighted_score", "mean_score", "velocity", "engagement_ratio", "post_count")
_BUCKET_HOURS = {"5m": 1 / 60, "1h": 1, "24h": 24}
_INSUFFICIENT_IC = 20
_INSUFFICIENT_QUANTILE = 30
_SIGNAL_COLUMNS = ("bucket_size", "feature", "horizon", "method", "value", "n",
                   "tstat", "insufficient", "detail", "computed_at")


# ---------- pure statistics ----------


def rank_ic(x, y) -> float | None:
    """Spearman rank correlation, or None when undefined (n < 3 / zero variance)."""
    xs = pd.Series(x, dtype="float64").reset_index(drop=True)
    ys = pd.Series(y, dtype="float64").reset_index(drop=True)
    mask = xs.notna() & ys.notna()
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return None
    rx, ry = xs.rank(), ys.rank()
    if rx.std(ddof=0) == 0 or ry.std(ddof=0) == 0:
        return None
    return float(rx.corr(ry))


def pooled_ic(panel: pd.DataFrame) -> dict:
    """Cross-sectional Spearman per timestamp, aggregated over time.

    ``panel`` has columns ``ts, x, y``. Returns the mean IC, information ratio,
    t-statistic, and the per-timestamp IC series (used for bootstrap significance).
    """
    per_time = []
    for _, group in panel.dropna(subset=["x", "y"]).groupby("ts"):
        ic = rank_ic(group["x"], group["y"])
        if ic is not None:
            per_time.append(ic)
    series = pd.Series(per_time, dtype="float64")
    n = int(len(series))
    if n == 0:
        return {"ic": 0.0, "ir": 0.0, "tstat": 0.0, "n": 0, "series": series}
    ic = float(series.mean())
    sd = float(series.std(ddof=0))
    ir = ic / sd if sd > 0 else 0.0
    return {"ic": ic, "ir": ir, "tstat": ir * math.sqrt(n), "n": n, "series": series}


def block_bootstrap_p(series, block: int = 1, seed: int = 0, n_boot: int = 2000) -> float:
    """Two-sided bootstrap p-value for a non-zero mean, with circular blocks.

    ``block`` should be the forward-return overlap length so serial correlation
    from overlapping horizons is preserved. Deterministic for a fixed seed.
    """
    arr = np.asarray([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    n = arr.size
    if n < 2:  # a bootstrap test is meaningless on 0-1 observations
        return 1.0
    block = int(max(1, min(block, n)))
    observed = float(arr.mean())
    centered = arr - observed
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    offsets = np.arange(block)[None, :]
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n] % n
        means[i] = centered[idx].mean()
    return float((np.abs(means) >= abs(observed)).mean())


def quantile_spread(values, fwd) -> dict:
    """Forward-return means by feature quantile, plus top-minus-bottom spread."""
    df = pd.DataFrame({"f": pd.Series(values, dtype="float64").reset_index(drop=True),
                       "r": pd.Series(fwd, dtype="float64").reset_index(drop=True)}).dropna()
    n = int(len(df))
    result = {"n": n, "quantiles": [], "means": [], "counts": [], "spread": 0.0,
              "tstat": 0.0, "monotonic": False, "insufficient": n < _INSUFFICIENT_QUANTILE}
    if n < _INSUFFICIENT_QUANTILE:
        return result
    q = 10 if n >= 200 else (5 if n >= 100 else 3)
    try:
        labels = pd.qcut(df["f"].rank(method="first"), q, labels=False)
    except ValueError:
        return result
    groups = df.groupby(labels)["r"]
    means = [float(m) for m in groups.mean().tolist()]
    counts = [int(c) for c in groups.count().tolist()]
    result.update({"quantiles": list(range(len(means))), "means": means, "counts": counts})
    spread = means[-1] - means[0]
    result["spread"] = spread
    top, bottom = df[labels == labels.max()]["r"], df[labels == labels.min()]["r"]
    se = 0.0
    if len(top) > 1 and len(bottom) > 1:
        se = math.sqrt(top.var(ddof=1) / len(top) + bottom.var(ddof=1) / len(bottom))
    result["tstat"] = float(spread / se) if se > 0 else 0.0
    diffs = np.diff(means)
    result["monotonic"] = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return result


# ---------- data assembly ----------


def load_sentiment_features(client, symbol: str, bucket: str = "1h") -> pd.DataFrame:
    """All metric columns for one ticker, indexed at bucket end (no lookahead)."""
    columns = ["bucket_start", "mean_score", "weighted_score", "velocity",
               "engagement_ratio", "post_count"]
    rows = client.query(
        "SELECT bucket_start, mean_score, weighted_score, velocity, engagement_ratio, post_count "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE ticker = {s:String} AND bucket_size = {b:String} ORDER BY bucket_start",
        parameters={"s": symbol, "b": bucket},
    ).result_rows
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns[1:]).set_index(pd.DatetimeIndex([], name="ts"))
    idx = pd.DatetimeIndex(df.pop("bucket_start"))
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    df.index = idx + pd.Timedelta(seconds=BUCKET_SECONDS[bucket])
    df.index.name = "ts"
    return df


def assemble_panel(client, symbol: str, bucket: str,
                   horizons: tuple[int, ...] = (1, 4, 24)) -> pd.DataFrame:
    """Feature rows with forward returns at each horizon (missing bars skipped)."""
    feats = load_sentiment_features(client, symbol, bucket)
    if feats.empty:
        return pd.DataFrame()
    try:
        close = load_ohlcv(client, symbol, interval="1h")["close"]
    except ValueError:
        return pd.DataFrame()
    momentum = close.pct_change(24)
    records = []
    for ts, row in feats.iterrows():
        base = close.get(ts)
        if base is None or not np.isfinite(base):
            continue
        rec: dict = {"ts": ts, "ticker": symbol}
        for feature in FEATURES:
            rec[feature] = float(row[feature]) if pd.notna(row[feature]) else None
        m = momentum.get(ts)
        rec["mom_24h"] = float(m) if m is not None and np.isfinite(m) else None
        for h in horizons:
            future = close.get(ts + pd.Timedelta(hours=h))
            rec[f"fwd_{h}"] = (float(future) / float(base) - 1.0) if (
                future is not None and np.isfinite(future)) else None
        records.append(rec)
    return pd.DataFrame(records)


# ---------- orchestration ----------


def _result(bucket: str, feature: str, horizon: str, method: str, value: float,
            n: int, tstat: float, insufficient: bool, detail: dict) -> dict:
    return {"bucket_size": bucket, "feature": feature, "horizon": horizon,
            "method": method, "value": float(value), "n": int(n),
            "tstat": float(tstat), "insufficient": int(bool(insufficient)),
            "detail": json.dumps(detail, default=str)}


def _coverage_rows(client, buckets: tuple[str, ...]) -> list[dict]:
    rows = []
    for bucket in buckets:
        count, tickers, lo, hi = client.query(
            f"SELECT count(), uniqExact(ticker), min(bucket_start), max(bucket_start) "
            f"FROM {database_name()}.sentiment_metrics FINAL WHERE bucket_size = {{b:String}}",
            parameters={"b": bucket},
        ).result_rows[0]
        rows.append(_result(bucket, "*", "*", "coverage", count or 0, count or 0, 0.0, False,
                            {"tickers": tickers or 0, "start": str(lo), "end": str(hi)}))
    sources = client.query(
        f"SELECT source, count(), countIf(sentiment_score IS NOT NULL), "
        f"min(published_at), max(published_at) "
        f"FROM {database_name()}.sentiment_posts FINAL GROUP BY source ORDER BY source"
    ).result_rows
    total = sum(int(r[1]) for r in sources)
    rows.append(_result("*", "*", "*", "coverage", total, total, 0.0, False,
                        {"sources": [
                            {"source": s, "posts": int(n), "scored": int(sc),
                             "start": str(lo), "end": str(hi)}
                            for s, n, sc, lo, hi in sources]}))
    return rows


def evaluate(client, buckets: tuple[str, ...] = ("1h", "24h"),
             horizons: tuple[int, ...] = (1, 4, 24), features: tuple[str, ...] = FEATURES,
             symbols: tuple[str, ...] | None = None, seed: int = 0) -> list[dict]:
    """Evaluate every (bucket, feature, horizon); returns result rows."""
    symbols = tuple(symbols) if symbols else tuple(TICKER_ALIASES)
    results = _coverage_rows(client, buckets)
    for bucket in buckets:
        panels = [p for p in (assemble_panel(client, s, bucket, horizons) for s in symbols)
                  if not p.empty]
        if not panels:
            continue
        panel = pd.concat(panels, ignore_index=True)
        bucket_hours = _BUCKET_HOURS.get(bucket, 1)
        for feature in features:
            for h in horizons:
                ycol = f"fwd_{h}"
                horizon_label = f"{h}h"
                sub = panel[["ts", "ticker", feature, ycol]].dropna()
                if sub.empty:
                    continue

                pooled = pooled_ic(sub.rename(columns={feature: "x", ycol: "y"}))
                block = max(1, int(math.ceil(h / bucket_hours)))
                pvalue = block_bootstrap_p(pooled["series"], block=block, seed=seed) \
                    if pooled["n"] else 1.0
                results.append(_result(bucket, feature, horizon_label, "ic_pooled",
                                       pooled["ic"], pooled["n"], pooled["tstat"],
                                       pooled["n"] < _INSUFFICIENT_IC,
                                       {"ir": pooled["ir"], "pvalue": pvalue, "block": block}))

                per_ticker = []
                for ticker, group in sub.groupby("ticker"):
                    ic = rank_ic(group[feature], group[ycol])
                    if ic is not None:
                        per_ticker.append({"ticker": ticker, "ic": ic, "n": int(len(group))})
                ics = [r["ic"] for r in per_ticker]
                results.append(_result(
                    bucket, feature, horizon_label, "ic_ts",
                    float(np.mean(ics)) if ics else 0.0, len(per_ticker), 0.0,
                    len(per_ticker) < 5,
                    {"pct_positive": float(np.mean([v > 0 for v in ics])) if ics else 0.0,
                     "per_ticker": per_ticker}))

                qs = quantile_spread(sub[feature], sub[ycol])
                results.append(_result(bucket, feature, horizon_label, "quantile",
                                       qs["spread"], qs["n"], qs["tstat"], qs["insufficient"],
                                       {"quantiles": qs["quantiles"], "means": qs["means"],
                                        "counts": qs["counts"], "monotonic": qs["monotonic"]}))

            dsub = panel[["ts", "ticker", feature, "mom_24h"]].dropna()
            if not dsub.empty:
                ic = rank_ic(dsub[feature], dsub["mom_24h"])
                results.append(_result(bucket, feature, "-", "divergence",
                                       ic if ic is not None else 0.0, int(len(dsub)), 0.0,
                                       len(dsub) < _INSUFFICIENT_IC,
                                       {"note": "rank corr(feature, trailing 24h return)"}))
    return results


def run_evaluation(client, **kwargs) -> list[dict]:
    """Evaluate and persist to signal_eval_results (idempotent via Replace keys)."""
    results = evaluate(client, **kwargs)
    now = datetime.now(timezone.utc)
    data = [[r["bucket_size"], r["feature"], r["horizon"], r["method"], r["value"],
             r["n"], r["tstat"], r["insufficient"], r["detail"], now] for r in results]
    if data:
        client.insert(f"{database_name()}.signal_eval_results", data,
                      column_names=list(_SIGNAL_COLUMNS))
    return results


# ---------- CLI ----------


def _parse_horizons(value: str) -> tuple[int, ...]:
    return tuple(int(v.rstrip("h")) for v in value.split(","))


def _print_report(results: list[dict], buckets: tuple[str, ...],
                  horizons: tuple[int, ...]) -> None:
    print("== coverage ==")
    for r in (r for r in results if r["method"] == "coverage"):
        detail = json.loads(r["detail"])
        if r["bucket_size"] != "*":
            print(f"  {r['bucket_size']:>4}: n={r['n']} tickers={detail.get('tickers')} "
                  f"{detail.get('start')} -> {detail.get('end')}")
        else:
            for s in detail.get("sources", []):
                print(f"  {s['source']:>10}: posts={s['posts']} scored={s['scored']} "
                      f"{s['start']} -> {s['end']}")
    for method, title in (("ic_pooled", "pooled cross-sectional IC"),
                          ("ic_ts", "per-ticker time-series IC")):
        print(f"\n== {title} ==")
        for bucket in buckets:
            for feature in FEATURES:
                for h in horizons:
                    for r in (r for r in results if r["method"] == method
                              and r["bucket_size"] == bucket and r["feature"] == feature
                              and r["horizon"] == f"{h}h"):
                        flag = "  [insufficient]" if r["insufficient"] else ""
                        print(f"  {bucket:>4} {feature:<16} {h:>2}h  "
                              f"value={r['value']:+.3f} n={r['n']:<4} t={r['tstat']:+.2f}{flag}")
    print("\n== quantile spread (top - bottom; overlapping windows, t is naive/indicative) ==")
    for bucket in buckets:
        for feature in FEATURES:
            for h in horizons:
                for r in (r for r in results if r["method"] == "quantile"
                          and r["bucket_size"] == bucket and r["feature"] == feature
                          and r["horizon"] == f"{h}h"):
                    detail = json.loads(r["detail"])
                    print(f"  {bucket:>4} {feature:<16} {h:>2}h  spread={r['value']:+.4f} "
                          f"n={r['n']:<4} t={r['tstat']:+.2f} monotonic={detail.get('monotonic')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment predictive signal")
    parser.add_argument("--bucket", default="1h,24h")
    parser.add_argument("--horizons", default="1h,4h,24h")
    parser.add_argument("--symbols", default="all",
                        help="comma-separated tickers or 'all' (default: all 12)")
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args(argv)

    buckets = tuple(args.bucket.split(","))
    horizons = _parse_horizons(args.horizons)
    symbols = tuple(TICKER_ALIASES) if args.symbols == "all" else tuple(args.symbols.split(","))
    client = get_clickhouse_client()

    runner = evaluate if args.no_store else run_evaluation
    results = runner(client, buckets=buckets, horizons=horizons, symbols=symbols)
    _print_report(results, buckets, horizons)


if __name__ == "__main__":
    main()
