"""Time-bucket sentiment metrics: 5m / 1h / 24h per ticker.

Recompute-and-replace: computing a bucket that already exists replaces it
(ReplacingMergeTree on (bucket_size, bucket_start, ticker)), so the
aggregator is idempotent. Baselines are same-size buckets, so velocity and
engagement_ratio are scale-consistent.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.ingestion.schemas import database_name

BUCKET_SIZES = ("5m", "1h", "24h")
_BUCKET_SECONDS = {"5m": 300, "1h": 3600, "24h": 86400}
_BASELINE_BUCKETS = {"5m": 288, "1h": 24, "24h": 7}

_METRICS_COLUMNS = (
    "bucket_size", "bucket_start", "ticker", "post_count", "mean_score",
    "weighted_score", "engagement_total", "velocity", "engagement_ratio",
    "computed_at",
)


def align_start(ts: datetime, bucket_size: str) -> datetime:
    epoch = int(ts.timestamp())
    secs = _BUCKET_SECONDS[bucket_size]
    return datetime.fromtimestamp(epoch - (epoch % secs), tz=timezone.utc)


def _window_stats(ch_client, start: datetime, end: datetime) -> dict[str, tuple]:
    """Per-ticker (post_count, mean_score, weighted_score, engagement) for a window."""
    rows = ch_client.query(
        f"""
        SELECT
            arrayJoin(tickers) AS ticker,
            count() AS post_count,
            avgIf(sentiment_score, sentiment_score IS NOT NULL) AS mean_score,
            sumIf(sentiment_score * (1 + likes + retweets + replies),
                  sentiment_score IS NOT NULL) AS weighted_sum,
            sumIf(1 + likes + retweets + replies,
                  sentiment_score IS NOT NULL) AS weight_total,
            sum(likes + retweets + replies) AS engagement_total
        FROM {database_name()}.sentiment_posts
        WHERE published_at >= {{start:DateTime64(3)}} AND published_at < {{end:DateTime64(3)}}
          AND notEmpty(tickers)
        GROUP BY ticker
        """,
        parameters={"start": start, "end": end},
    ).result_rows
    stats = {}
    for ticker, post_count, mean_score, wsum, wtotal, engagement in rows:
        if mean_score is None or (isinstance(mean_score, float) and math.isnan(mean_score)):
            scored_mean = 0.0
        else:
            scored_mean = float(mean_score)
        weighted = float(wsum) / float(wtotal) if wtotal else 0.0
        stats[ticker] = (int(post_count), scored_mean, weighted, int(engagement))
    return stats


def compute_metrics(ch_client, bucket_size: str, bucket_start: datetime) -> int:
    """Recompute one bucket for every ticker mentioned in it. Returns rows written."""
    secs = _BUCKET_SECONDS[bucket_size]
    n_baseline = _BASELINE_BUCKETS[bucket_size]
    bucket_end = bucket_start + timedelta(seconds=secs)
    baseline_start = bucket_start - timedelta(seconds=secs * n_baseline)

    current = _window_stats(ch_client, bucket_start, bucket_end)
    if not current:
        return 0
    baseline = _window_stats(ch_client, baseline_start, bucket_start)

    now = datetime.now(timezone.utc)
    data = []
    for ticker, (post_count, mean_score, weighted, engagement) in current.items():
        base_count, _, _, base_engagement = baseline.get(ticker, (0, 0.0, 0.0, 0))
        mean_count = base_count / n_baseline
        mean_engagement = base_engagement / n_baseline
        velocity = post_count / mean_count if mean_count > 0 else 0.0
        engagement_ratio = engagement / mean_engagement if mean_engagement > 0 else 0.0
        data.append([
            bucket_size, bucket_start, ticker, post_count, mean_score, weighted,
            engagement, velocity, engagement_ratio, now,
        ])
    ch_client.insert(
        f"{database_name()}.sentiment_metrics",
        data,
        column_names=list(_METRICS_COLUMNS),
    )
    return len(data)


def run_aggregation(ch_client, now: datetime | None = None) -> int:
    """Compute the just-closed bucket for each size. Returns total rows written.

    The just-closed 24h bucket only changes at day boundaries; recomputing it
    every cycle is harmless because recompute replaces the same key.
    """
    now = now or datetime.now(timezone.utc)
    total = 0
    for size in BUCKET_SIZES:
        bucket_start = align_start(now, size) - timedelta(seconds=_BUCKET_SECONDS[size])
        total += compute_metrics(ch_client, size, bucket_start)
    return total


def backfill_metrics(ch_client, bucket_size: str, start: datetime,
                     end: datetime) -> int:
    """Recompute every non-empty bucket in [start, end) — historical backfill.

    Only buckets that actually contain ticker-tagged posts are touched, so a
    year of sparse history is a few hundred ``compute_metrics`` calls rather
    than every bucket in the range. Idempotent (recompute replaces).
    """
    secs = _BUCKET_SECONDS[bucket_size]
    rows = ch_client.query(
        f"""
        SELECT DISTINCT toStartOfInterval(published_at,
               INTERVAL {secs} SECOND) AS bucket_start
        FROM {database_name()}.sentiment_posts
        WHERE published_at >= {{start:DateTime64(3)}} AND published_at < {{end:DateTime64(3)}}
          AND notEmpty(tickers)
        ORDER BY bucket_start
        """,
        parameters={"start": start, "end": end},
    ).result_rows
    total = 0
    for (bucket_start,) in rows:
        if bucket_start.tzinfo is None:
            bucket_start = bucket_start.replace(tzinfo=timezone.utc)
        total += compute_metrics(ch_client, bucket_size, bucket_start)
    return total


def main(argv: list[str] | None = None) -> None:
    import argparse

    from src.ingestion.schemas import get_clickhouse_client

    parser = argparse.ArgumentParser(description="Backfill historical sentiment metrics")
    parser.add_argument("--bucket", choices=BUCKET_SIZES, default="1h")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    args = parser.parse_args(argv)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    n = backfill_metrics(get_clickhouse_client(), args.bucket, start, end)
    print(f"aggregator: {n} metric rows backfilled ({args.bucket})")


if __name__ == "__main__":
    main()
