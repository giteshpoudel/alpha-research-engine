import pytest

from src.ingestion.schemas import get_clickhouse_client
from src.meta_learning import assignments as A


@pytest.fixture
def ch():
    c = get_clickhouse_client()
    A.ensure_assignments(c)
    yield c
    A.set_assignment(c, "BTC", A.DEFAULT_STRATEGY)  # restore


def test_ensure_and_assign(ch):
    assert A.assigned_strategy(ch, "BTC") in {"mean_reversion", "momentum", "breakout"}
    A.set_assignment(ch, "BTC", "momentum")
    assert A.assigned_strategy(ch, "BTC") == "momentum"
    assert A.list_assignments(ch)["BTC"] == "momentum"


def test_set_assignment_rejects_unknown(ch):
    with pytest.raises(ValueError, match="unknown strategy"):
        A.set_assignment(ch, "BTC", "does_not_exist")


def test_category_priors():
    assert A.default_strategy_for("meme") == "breakout"
    assert A.default_strategy_for("alt") == "momentum"
    assert A.default_strategy_for("major") == "mean_reversion"
    assert A.default_strategy_for(None) == "mean_reversion"


def test_apply_priors_keeps_non_default(ch):
    A.set_assignment(ch, "BTC", "breakout")  # non-default choice
    changed = A.apply_priors(ch, only_default=True)
    assert "BTC" not in changed
    A.set_assignment(ch, "BTC", A.DEFAULT_STRATEGY)  # restore
