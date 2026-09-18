from datetime import datetime, timedelta, timezone

import pytest

from src.ingestion.aggregator import align_start, compute_metrics, run_aggregation
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_align_start():
    ts = datetime(2024, 1, 15, 10, 7, 33, tzinfo=timezone.utc)
    assert align_start(ts, "5m") == datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc)
    assert align_start(ts, "1h") == datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    assert align_start(ts, "24h") == datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    db = database_name()
    client.command(f"ALTER TABLE {db}.sentiment_posts DELETE WHERE startsWith(post_id, 'test:')",
                   settings={"mutations_sync": 1})
    client.command(f"ALTER TABLE {db}.sentiment_metrics DELETE WHERE ticker = 'TEST'",
                   settings={"mutations_sync": 1})


def _seed_post(ch, post_id, tickers, score, likes, replies, published_at):
    ch.insert(
        f"{database_name()}.sentiment_posts",
        [[post_id, "reddit", "tester", "seed", tickers, "en",
          likes, 0, replies, "", published_at, datetime.now(timezone.utc), score]],
        column_names=["post_id", "source", "author", "text", "tickers", "lang",
                      "likes", "retweets", "replies", "url",
                      "published_at", "ingested_at", "sentiment_score"],
    )


def test_compute_metrics_exact_values(ch_client):
    bucket_start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    inside = bucket_start + timedelta(minutes=2)
    # two scored TEST posts in the bucket
    _seed_post(ch_client, "test:agg-1", ["TEST"], 0.5, 10, 0, inside)
    _seed_post(ch_client, "test:agg-2", ["TEST"], -0.5, 30, 10, inside)
    # one unscored TEST post (counts for volume, not scores)
    _seed_post(ch_client, "test:agg-3", ["TEST"], None, 0, 0, inside)
    # baseline: one TEST post in each of the 24 preceding 1h buckets -> mean = 1.0
    for h in range(1, 25):
        _seed_post(ch_client, f"test:agg-base-{h}", ["TEST"], 0.0, 1, 0,
                   bucket_start - timedelta(hours=h))

    assert compute_metrics(ch_client, "1h", bucket_start) >= 1

    rows = ch_client.query(
        f"SELECT post_count, mean_score, weighted_score, engagement_total, "
        f"velocity, engagement_ratio "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' AND bucket_start = {ts:DateTime64(3)}",
        parameters={"ts": bucket_start},
    ).result_rows
    assert len(rows) == 1
    post_count, mean_score, weighted_score, engagement, velocity, eng_ratio = rows[0]
    assert post_count == 3
    assert mean_score == pytest.approx(0.0, abs=1e-6)          # (0.5 + -0.5) / 2
    # weights: p1 = 1+10 = 11, p2 = 1+30+10 = 41 -> (0.5*11 - 0.5*41) / 52
    assert weighted_score == pytest.approx(-15.0 / 52.0, abs=1e-4)
    assert engagement == 50                                     # 10 + (30+10) + 0
    assert velocity == pytest.approx(3.0, abs=1e-4)             # 3 / (24/24)
    assert eng_ratio == pytest.approx(50.0 / 1.0, abs=1e-2)     # 50 / (24*1/24)


def test_run_aggregation_writes_recent_buckets(ch_client):
    bucket_start = align_start(datetime.now(timezone.utc), "1h") - timedelta(hours=1)
    _seed_post(ch_client, "test:agg-run", ["TEST"], 0.25, 2, 1,
               bucket_start + timedelta(minutes=5))
    written = run_aggregation(ch_client)
    assert written >= 1
    rows = ch_client.query(
        f"SELECT post_count FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' AND bucket_start = {ts:DateTime64(3)}",
        parameters={"ts": bucket_start},
    ).result_rows
    assert rows and rows[0][0] == 1


def test_backfill_metrics_only_nonempty_buckets(ch_client):
    from src.ingestion.aggregator import backfill_metrics

    b1 = datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc)
    b2 = datetime(2024, 3, 1, 9, 0, tzinfo=timezone.utc)
    _seed_post(ch_client, "test:bf-1", ["TEST"], 0.2, 1, 0, b1 + timedelta(minutes=1))
    # b2 is intentionally empty
    assert backfill_metrics(ch_client, "1h", b1, b1 + timedelta(hours=3)) >= 1
    count = ch_client.query(
        f"SELECT count() FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' "
        "AND bucket_start >= {s:DateTime64(3)} AND bucket_start <= {e:DateTime64(3)}",
        parameters={"s": b1, "e": b2},
    ).result_rows[0][0]
    assert count == 1  # only the non-empty bucket was computed


def test_compute_metrics_unscored_only_bucket(ch_client):
    bucket_start = datetime(2024, 2, 20, 10, 0, tzinfo=timezone.utc)
    # only an unscored TEST post: avgIf over zero scored rows yields NaN,
    # which must be written as 0.0, same as None
    _seed_post(ch_client, "test:agg-nan", ["TEST"], None, 0, 0,
               bucket_start + timedelta(minutes=2))

    assert compute_metrics(ch_client, "1h", bucket_start) >= 1

    rows = ch_client.query(
        f"SELECT mean_score, weighted_score "
        f"FROM {database_name()}.sentiment_metrics FINAL "
        "WHERE bucket_size = '1h' AND ticker = 'TEST' AND bucket_start = {ts:DateTime64(3)}",
        parameters={"ts": bucket_start},
    ).result_rows
    assert len(rows) == 1
    mean_score, weighted_score = rows[0]
    assert mean_score == 0.0
    assert weighted_score == 0.0
