import pytest

from src.agents import strategy_synth
from src.backtesting.strategies import registry
from src.ingestion.schemas import database_name, get_clickhouse_client

CODE = (
    "import pandas as pd\n"
    "DEFAULTS = {'window': 10}\n"
    "def signals(close, window=10):\n"
    "    ma = close.rolling(window).mean()\n"
    "    cond = close > ma\n"
    "    e = (cond & ~cond.shift(1, fill_value=False)).fillna(False).astype(bool)\n"
    "    x = (~cond & cond.shift(1, fill_value=False)).fillna(False).astype(bool)\n"
    "    return e, x\n"
)


class _Res:
    model = "mock"

    def __init__(self):
        self.text = CODE


@pytest.fixture
def ch(change_log_guard):
    c = get_clickhouse_client()
    yield c
    c.command(f"ALTER TABLE {database_name()}.generated_strategies "
              "DELETE WHERE name LIKE 'gen%'", settings={"mutations_sync": 1})
    for path in registry.GENERATED_DIR.glob("gen*.py"):
        path.unlink()
    registry.load_generated()


def test_synthesize_batch_registers(ch):
    names = strategy_synth.synthesize_batch(ch, n=1, chat_fn=lambda role, s, u: _Res())
    assert names and names[0].startswith("gen")
    assert names[0] in registry.SIGNAL_FUNCS
