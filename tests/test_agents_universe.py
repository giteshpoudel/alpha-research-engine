import pytest

from src.ingestion import universe as uni
from src.ingestion.schemas import database_name, get_clickhouse_client

SYM = "TESTCOIN"


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.universe DELETE WHERE symbol = '{SYM}'",
              settings={"mutations_sync": 1})


def test_ensure_and_universe_seeded(ch):
    uni.ensure_universe(ch)
    symbols = uni.universe(ch)
    assert "BTC" in symbols and "DOGE" in symbols


def test_add_and_remove_symbol(ch):
    stats = uni.add_symbol(ch, SYM, aliases=["testcoin"], backfill=False)
    assert stats["enabled"] is True
    assert SYM in uni.universe(ch)
    assert uni.get_symbol(ch, SYM)["aliases"] == ["testcoin"]

    uni.remove_symbol(ch, SYM)
    assert SYM not in uni.universe(ch)                 # enabled only
    assert SYM in uni.universe(ch, enabled_only=False)  # still listed


def test_add_symbol_falls_back_to_coinbase(ch, monkeypatch):
    from datetime import datetime, timezone

    import src.ingestion.binance_us as binance
    import src.ingestion.coinbase as coinbase
    from src.ingestion.market_data import insert_ohlcv

    monkeypatch.setattr(binance, "backfill_ohlcv", lambda *a, **k: 0)

    def fake_coinbase(ch_, symbols=None, intervals=("1h", "1d"), start=None, **k):
        insert_ohlcv(ch_, [{"exchange": "coinbase", "symbol": "TESTCB", "interval": "1h",
                            "ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
                            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                            "volume": 1.0, "quote_volume": 0.0, "num_trades": 0}])
        return 1

    monkeypatch.setattr(coinbase, "backfill_coinbase", fake_coinbase)
    try:
        stats = uni.add_symbol(ch, "TESTCB", backfill=True)
        assert stats["enabled"] is True
    finally:
        ch.command(f"ALTER TABLE {database_name()}.universe DELETE WHERE symbol = 'TESTCB'",
                   settings={"mutations_sync": 1})
        ch.command(f"ALTER TABLE {database_name()}.ohlcv DELETE WHERE symbol = 'TESTCB'",
                   settings={"mutations_sync": 1})


def test_add_symbol_without_data_is_disabled(ch, monkeypatch):
    import src.ingestion.binance_us as binance
    import src.ingestion.coinbase as coinbase
    import src.ingestion.hyperliquid as hyper

    monkeypatch.setattr(binance, "backfill_ohlcv", lambda *a, **k: 0)
    monkeypatch.setattr(coinbase, "backfill_coinbase", lambda *a, **k: 0)
    monkeypatch.setattr(hyper, "backfill_funding", lambda *a, **k: 0)
    stats = uni.add_symbol(ch, "NODATA", backfill=True)
    assert stats["enabled"] is False
    assert "NODATA" not in uni.universe(ch)
    ch.command(f"ALTER TABLE {database_name()}.universe DELETE WHERE symbol = 'NODATA'",
               settings={"mutations_sync": 1})
