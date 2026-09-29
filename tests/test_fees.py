from src.backtesting import fees


def test_exchange_taker_defaults():
    assert fees.fee_for_exchange("binance_us") == 0.0002
    assert fees.fee_for_exchange("coinbase") == 0.006


def test_maker_mode():
    assert fees.fee_for_exchange("binance_us", mode="maker") == 0.0
    assert fees.fee_for_exchange("coinbase", mode="maker") == 0.004


def test_fee_mode_env(monkeypatch):
    monkeypatch.setenv("FEE_MODE", "maker")
    assert fees.fee_for_exchange("binance_us") == 0.0


def test_exchange_env_override(monkeypatch):
    monkeypatch.setenv("FEE_COINBASE_TAKER", "0.005")
    assert fees.fee_for_exchange("coinbase") == 0.005


def test_fee_for_symbol_uses_exchange(monkeypatch):
    monkeypatch.setattr("src.backtesting.data.exchange_for_symbol",
                        lambda ch, s: "coinbase")
    assert fees.fee_for_symbol(None, "PEPE") == 0.006


def test_execution_mode_defaults_taker(monkeypatch):
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("FEE_MODE", raising=False)
    assert fees.execution_mode() == "taker"
    monkeypatch.setenv("EXECUTION_MODE", "maker")
    assert fees.execution_mode() == "maker"
    assert fees.fee_for_exchange("binance_us") == 0.0  # maker side


def test_execution_lag_default_and_env(monkeypatch):
    monkeypatch.delenv("EXECUTION_LAG", raising=False)
    assert fees.execution_lag() == 1
    monkeypatch.setenv("EXECUTION_LAG", "0")
    assert fees.execution_lag() == 0


def test_maker_params_env(monkeypatch):
    monkeypatch.setenv("MAKER_LIMIT_BPS", "3")
    monkeypatch.setenv("MAKER_MAX_WAIT", "2")
    assert fees.maker_limit_bps() == 3.0
    assert fees.maker_max_wait() == 2
