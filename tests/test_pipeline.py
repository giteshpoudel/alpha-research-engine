import pytest

import src.ingestion.pipeline as pipeline


@pytest.fixture
def staged(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "poll_subreddits", lambda ch: calls.append("reddit") or 3)
    monkeypatch.setattr(pipeline, "poll_feeds", lambda ch: calls.append("rss") or 2)
    monkeypatch.setattr(pipeline, "poll_symbols", lambda ch: calls.append("stocktwits") or 4)
    monkeypatch.setattr(pipeline, "score_pending_posts",
                        lambda ch, qd: calls.append("score") or 5)
    monkeypatch.setattr(pipeline, "run_aggregation",
                        lambda ch: calls.append("aggregate") or 10)
    return calls


def test_run_cycle_order_and_stats(staged):
    stats = pipeline.run_cycle(ch_client=object(), qd_client=object())
    assert staged == ["reddit", "rss", "stocktwits", "score", "aggregate"]
    assert stats == {"reddit": 3, "rss": 2, "stocktwits": 4, "score": 5, "aggregate": 10}


def test_run_cycle_stage_failure_is_isolated(monkeypatch, staged):
    def boom(ch):
        raise RuntimeError("reddit down")
    monkeypatch.setattr(pipeline, "poll_subreddits", boom)
    stats = pipeline.run_cycle(ch_client=object(), qd_client=object())
    assert stats["reddit"] == -1
    assert staged == ["rss", "stocktwits", "score", "aggregate"]  # later stages still ran


def test_main_once_runs_single_cycle(monkeypatch):
    cycles = []
    monkeypatch.setattr(pipeline, "run_cycle", lambda: cycles.append(1))
    pipeline.main(["--once"])
    assert len(cycles) == 1
