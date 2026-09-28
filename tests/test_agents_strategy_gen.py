import pytest

from src.agents import strategy_gen, tools
from src.backtesting.strategies import registry
from src.ingestion.schemas import database_name, get_clickhouse_client

BENIGN = (
    "import pandas as pd\n"
    "DEFAULTS = {'window': 5}\n"
    "def signals(close, window=5):\n"
    "    ma = close.rolling(window).mean()\n"
    "    cond = close > ma\n"
    "    entries = cond & ~cond.shift(1, fill_value=False)\n"
    "    exits = ~cond & cond.shift(1, fill_value=False)\n"
    "    return entries.fillna(False).astype(bool), exits.fillna(False).astype(bool)\n"
)


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.generated_strategies "
              "DELETE WHERE name LIKE 'testgen%'", settings={"mutations_sync": 1})
    for path in registry.GENERATED_DIR.glob("testgen*.py"):
        path.unlink()
    registry.load_generated()


def test_register_benign_and_registry_loads(ch):
    result = strategy_gen.register_generated_strategy(ch, "testgen_ma", BENIGN, "ma cross")
    assert result["registered"] is True
    assert "testgen_ma" in registry.SIGNAL_FUNCS
    assert registry.defaults("testgen_ma")["window"] == 5


def test_register_rejects_dangerous(ch):
    result = strategy_gen.register_generated_strategy(
        ch, "testgen_bad", "import os\ndef signals(c):\n    return c, c\n")
    assert result["registered"] is False and result["errors"]


def test_register_rejects_bad_name(ch):
    assert strategy_gen.register_generated_strategy(ch, "No Spaces", BENIGN)["registered"] is False


def test_register_strategy_tool(ch):
    result = tools.invoke("register_strategy",
                          {"name": "testgen_tool", "code": BENIGN, "description": "d"},
                          ch=ch)
    assert result.ok and result.data["registered"] is True
