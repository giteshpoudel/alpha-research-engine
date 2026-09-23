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


def test_add_symbol_without_data_is_disabled(ch, monkeypatch):
    import src.ingestion.binance_us as binance
    import src.ingestion.hyperliquid as hyper

    monkeypatch.setattr(binance, "backfill_ohlcv", lambda *a, **k: 0)
    monkeypatch.setattr(hyper, "backfill_funding", lambda *a, **k: 0)
    stats = uni.add_symbol(ch, "NODATA", backfill=True)
    assert stats["enabled"] is False
    assert "NODATA" not in uni.universe(ch)
    ch.command(f"ALTER TABLE {database_name()}.universe DELETE WHERE symbol = 'NODATA'",
               settings={"mutations_sync": 1})
