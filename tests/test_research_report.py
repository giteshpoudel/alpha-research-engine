from datetime import date
from pathlib import Path

import pytest

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research.graph import run_report

TEST_DATE = date(2026, 1, 15)


@pytest.fixture
def ch_client():
    client = get_clickhouse_client()
    yield client
    client.command(
        f"ALTER TABLE {database_name()}.research_reports DELETE WHERE report_date = {{d:Date}}",
        parameters={"d": TEST_DATE}, settings={"mutations_sync": 1},
    )


def _stub_chat(system: str, user: str, **kwargs) -> str:
    if "risk" in system:
        return "RISK: BTC vol elevated; BTC-ETH correlation 0.9."
    if "macro" in system:
        return "MACRO: sentiment bullish on SOL; funding neutral."
    return "# Daily Research Report\n\nSynthesis of risk and macro."


def test_run_report_stores_sections_and_file(ch_client, tmp_path: Path):
    md = run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=_stub_chat)
    assert "Daily Research Report" in md

    out_file = tmp_path / f"{TEST_DATE.isoformat()}.md"
    assert out_file.exists()
    assert "Synthesis" in out_file.read_text()

    rows = ch_client.query(
        f"SELECT section, model, length(content) > 0 FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date} ORDER BY section",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert [r[0] for r in rows] == ["macro", "report", "risk"]
    assert all(r[1] == "mock" for r in rows)
    assert all(r[2] for r in rows)

    # idempotent: re-run replaces, never duplicates
    run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=_stub_chat)
    count = ch_client.query(
        f"SELECT count() FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date}",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert count[0][0] == 3


def test_run_report_llm_failure_falls_back_to_data(ch_client, tmp_path: Path):
    def boom(system, user, **kwargs):
        raise RuntimeError("LLM request failed after retry: down")

    md = run_report(ch_client, TEST_DATE, out_dir=tmp_path, chat_fn=boom)
    rows = ch_client.query(
        f"SELECT DISTINCT model FROM {database_name()}.research_reports FINAL "
        "WHERE report_date = {d:Date}",
        parameters={"d": TEST_DATE},
    ).result_rows
    assert [r[0] for r in rows] == ["unavailable"]
    assert md  # deterministic data summary, never empty
