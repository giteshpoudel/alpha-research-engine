from src.agents.strategy_sandbox import smoke_test, validate_source

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


def test_validate_accepts_benign():
    assert validate_source(BENIGN) == []


def test_validate_rejects_dangerous():
    assert any("import not allowed" in e for e in validate_source("import os\n"))
    assert any("while" in e
               for e in validate_source("def signals(c):\n    while True:\n        pass\n"))
    assert any("forbidden name" in e
               for e in validate_source("def signals(c):\n    return eval('1'), c"))
    assert any("forbidden attribute" in e
               for e in validate_source("def signals(c):\n    return c.__class__, c"))
    assert any("signals" in e for e in validate_source("x = 1\n"))


def test_smoke_test_benign_ok():
    result = smoke_test(BENIGN)
    assert result["ok"] is True and result["entries"] >= 0


def test_smoke_test_kills_infinite_loop():
    result = smoke_test("def signals(close):\n    while True:\n        pass\n", timeout=3.0)
    assert result["ok"] is False and "timeout" in result["error"]


def test_smoke_test_reports_exception():
    result = smoke_test("def signals(close):\n    raise ValueError('boom')\n")
    assert result["ok"] is False and "boom" in result["error"]
