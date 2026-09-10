from datetime import datetime, timezone

import pytest

import src.ingestion.backfill as backfill


@pytest.fixture
def jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "backfill_ohlcv",
                        lambda ch, symbols=None, intervals=("1h", "1d"), start=None:
                        calls.append(("binance_us", symbols)) or 100)
    monkeypatch.setattr(backfill, "backfill_matic",
                        lambda ch, intervals=("1h", "1d"), start=None:
                        calls.append(("coinbase_matic", None)) or 50)
    monkeypatch.setattr(backfill, "backfill_funding",
                        lambda ch, coins=None, start=None:
                        calls.append(("hyperliquid_funding", coins)) or 200)
    return calls


def test_run_backfill_all_sources(jobs):
    stats = backfill.run_backfill(object())
    assert [c[0] for c in jobs] == ["binance_us", "coinbase_matic", "hyperliquid_funding"]
    assert stats == {"binance_us": 100, "coinbase_matic": 50, "hyperliquid_funding": 200}
    assert jobs[0][1] is None  # symbols=None -> all 11 binance tickers
    assert jobs[2][1] is None  # coins=None -> all 12


def test_run_backfill_symbol_filter(jobs):
    backfill.run_backfill(object(), symbols=("BTC",))
    assert [c[0] for c in jobs] == ["binance_us", "hyperliquid_funding"]  # no MATIC -> no coinbase job
    assert jobs[0][1] == ("BTC",)
    assert jobs[1][1] == ("BTC",)


def test_run_backfill_source_failure_isolated(monkeypatch, jobs):
    monkeypatch.setattr(backfill, "backfill_matic",
                        lambda ch, intervals=("1h", "1d"), start=None: (_ for _ in ()).throw(RuntimeError("coinbase down")))
    stats = backfill.run_backfill(object())
    assert stats["coinbase_matic"] == -1
    assert stats["hyperliquid_funding"] == 200


def test_main_parses_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(backfill, "run_backfill",
                        lambda ch, symbols=None, intervals=("1h", "1d"), start=None:
                        seen.update(symbols=symbols, intervals=intervals, start=start))
    monkeypatch.setattr(backfill, "get_clickhouse_client", lambda: object())
    backfill.main(["--symbols", "BTC", "ETH", "--intervals", "1h", "--start", "2024-01-01"])
    assert seen["symbols"] == ["BTC", "ETH"]
    assert seen["intervals"] == ("1h",)
    assert seen["start"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
