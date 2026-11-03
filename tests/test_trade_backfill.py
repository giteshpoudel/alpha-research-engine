from datetime import datetime, timezone

from src.ingestion import trade_backfill as tb


def test_kraken_rows_use_trade_id_and_taker_side():
    rows = tb.kraken_rows("BTC", [["64650.1", "0.0015", 1785964928.1024, "b", "l", "", 104796236],
                                  ["64650.0", "0.0024", 1785964930.5, "s", "m", "", 104796237]])
    assert rows[0][:3] == ["kraken:USD", "BTC", 104796236]
    assert rows[0][3] == datetime.fromtimestamp(1785964928.1024, tz=timezone.utc)
    assert [r[6] for r in rows] == [0, 1]  # buy taker -> buyer not maker; sell taker -> buyer maker


def test_binance_rows_match_collector_format():
    rows = tb.binance_rows("binance_us:USD", "ETH",
                           [{"a": 7, "p": "2250.5", "q": "0.1", "T": 1785964989106, "m": True}])
    assert rows == [["binance_us:USD", "ETH", 7,
                     datetime.fromtimestamp(1785964989.106, tz=timezone.utc), 2250.5, 0.1, 1]]
