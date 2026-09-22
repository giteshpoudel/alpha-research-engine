import pytest

from src.agents import symbols
from src.ingestion.schemas import database_name, get_clickhouse_client


def test_price_bucket_boundaries():
    assert symbols.price_bucket(0.5) == "$0-2"
    assert symbols.price_bucket(1.999) == "$0-2"
    assert symbols.price_bucket(2.0) == "$2-20"
    assert symbols.price_bucket(19.99) == "$2-20"
    assert symbols.price_bucket(20.0) == "$20+"
    assert symbols.price_bucket(86193.0) == "$20+"


def test_category_and_override():
    assert symbols.category("DOGE") == "meme"
    assert symbols.category("BTC") == "major"
    assert symbols.category("XRP") == "alt"
    assert symbols.category("XRP", {"XRP": "meme"}) == "meme"


def test_pump_risk():
    assert symbols.pump_risk("meme", 0.1) == "high"
    assert symbols.pump_risk("alt", 0.95) == "high"
    assert symbols.pump_risk("alt", 0.75) == "medium"
    assert symbols.pump_risk("major", 0.5) == "low"


def test_classify_shape_and_values():
    row = symbols.classify("DOGE", 0.10, 0.94)
    assert set(row) == {"symbol", "price", "price_bucket", "category",
                        "ann_vol", "pump_risk"}
    assert row["price_bucket"] == "$0-2"
    assert row["category"] == "meme"
    assert row["pump_risk"] == "high"


@pytest.fixture
def ch_client():
    yield get_clickhouse_client()


def test_classify_universe_and_store(ch_client):
    rows = symbols.classify_universe(ch_client, symbols=("BTC", "DOGE"))
    assert {r["symbol"] for r in rows} == {"BTC", "DOGE"}
    assert symbols.store_metadata(ch_client, rows) == 2

    loaded = {r["symbol"]: r
              for r in symbols.load_metadata(ch_client)
              if r["symbol"] in ("BTC", "DOGE")}
    assert loaded["DOGE"]["category"] == "meme"
    assert loaded["BTC"]["price_bucket"] == "$20+"
    assert loaded["BTC"]["pump_risk"] in {"low", "medium", "high"}
