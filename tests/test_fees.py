from src.backtesting import fees


def test_fee_defaults_by_interval():
    assert fees.fee_for_interval("5m") == 0.0002
    assert fees.fee_for_interval("1h") == 0.001
    assert fees.fee_for_strategy("mean_reversion") == 0.001
    assert fees.fee_for_strategy("mean_reversion@5m") == 0.0002


def test_fee_env_override(monkeypatch):
    monkeypatch.setenv("FEE_5M", "0.0001")
    assert fees.fee_for_interval("5m") == 0.0001
    assert fees.fee_for_strategy("breakout@5m") == 0.0001
