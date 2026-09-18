"""LangGraph wiring for the daily research report."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.ingestion.schemas import database_name, get_clickhouse_client
from src.research import agents, collectors
from src.research.llm import active_model, chat

_SECTION_COLUMNS = ("report_date", "section", "content", "model", "created_at")


class ReportState(TypedDict, total=False):
    report_date: date
    risk_data: dict
    macro_data: dict
    portfolio_data: dict
    signal_data: dict
    risk_section: str
    macro_section: str
    report_md: str


def build_graph(chat_fn=chat) -> object:
    graph = StateGraph(ReportState)

    def collect_risk(state: ReportState) -> dict:
        return {"risk_data": collectors.collect_risk_data(get_clickhouse_client())}

    def collect_macro(state: ReportState) -> dict:
        return {"macro_data": collectors.collect_macro_data(get_clickhouse_client())}

    graph.add_node("collect_risk", collect_risk)
    graph.add_node("collect_macro", collect_macro)
    graph.add_node("risk_agent", lambda s: agents.risk_agent(s, chat_fn))
    graph.add_node("macro_agent", lambda s: agents.macro_agent(s, chat_fn))
    graph.add_node("report_agent", lambda s: agents.report_agent(s, chat_fn))

    graph.add_edge(START, "collect_risk")
    graph.add_edge(START, "collect_macro")
    graph.add_edge("collect_risk", "risk_agent")
    graph.add_edge("collect_macro", "macro_agent")
    graph.add_edge("risk_agent", "report_agent")
    graph.add_edge("macro_agent", "report_agent")
    graph.add_edge("report_agent", END)
    return graph.compile()


def _portfolio_md(portfolio: dict) -> str:
    lines = ["## Paper Trading (strategy-implied book)", ""]
    sleeves = portfolio.get("sleeves") or []
    if not sleeves:
        lines.append("No paper data yet.")
        return "\n".join(lines)
    lines.append(f"As of {portfolio['as_of']}; equal-weight return "
                 f"{portfolio['equal_weight_return']:+.3f}.")
    halted = portfolio.get("halted") or []
    lines.append(f"Halts (trailing-30d < 0): {', '.join(halted) if halted else 'none'}.")
    lines.append("")
    lines.append("| Symbol | Equity | Total | Trailing 30d | Status |")
    lines.append("|---|---:|---:|---:|---|")
    for s in sleeves:
        lines.append(f"| {s['symbol']} | {s['equity']:.3f} | {s['total_return']:+.3f} | "
                     f"{s['trailing_return']:+.3f} | {'halted' if not s['enabled'] else 'active'} |")
    versions = sorted(set((portfolio.get("param_valid_from") or {}).values()))
    if versions:
        lines.append("")
        lines.append(f"Param version in effect: {', '.join(versions)}.")
    return "\n".join(lines)


def _signal_md(signal: dict) -> str:
    lines = ["## Sentiment Signal", ""]
    top = signal.get("top_ic") or []
    if not top:
        lines.append("No statistically significant predictive IC at current sample sizes.")
        return "\n".join(lines)
    lines.append("Strongest cross-sectional ICs (block-bootstrap significant):")
    lines.append("")
    lines.append("| Bucket | Feature | Horizon | IC | n | t |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in top:
        lines.append(f"| {r['bucket']} | {r['feature']} | {r['horizon']} | "
                     f"{r['ic']:+.3f} | {r['n']} | {r['t']:+.2f} |")
    return "\n".join(lines)


def _data_summary(state: ReportState) -> str:
    lines = ["# Daily Research Report (data summary — LLM unavailable)", ""]
    lines.append("## Signal states (strategy-implied book)")
    for s in state["risk_data"]["signal_states"]:
        lines.append(f"- {s['symbol']}: {s['state']} (z={s['z']})")
    lines.append("")
    lines.append("## Sentiment")
    for item in state["macro_data"]["sentiment"][:5]:
        lines.append(f"- {item['ticker']}: score {item['weighted_score']}, "
                     f"velocity {item['velocity']}, flags {item['flags']}")
    lines.append("")
    lines.append(_portfolio_md(state.get("portfolio_data", {})))
    lines.append("")
    lines.append(_signal_md(state.get("signal_data", {})))
    lines.append("")
    lines.append("Portfolio figures are strategy-implied (research/backtest "
                 "scope), not real holdings. Not financial advice.")
    return "\n".join(lines)


def run_report(ch_client, report_date: date, out_dir: Path | None = None,
               chat_fn=chat) -> str:
    """Run the full report graph, store sections, write markdown. Returns markdown."""
    out_dir = out_dir or Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic portfolio/signal data feeds the editor and the appended
    # sections; collected up front so the LLM graph stays a simple risk/macro
    # diamond.
    state: ReportState = {
        "report_date": report_date,
        "portfolio_data": collectors.collect_portfolio_data(ch_client),
        "signal_data": collectors.collect_signal_data(ch_client),
    }
    # Identity check: tests inject a stub chat_fn; production uses llm.chat.
    model = "mock" if chat_fn is not chat else active_model()
    try:
        result = build_graph(chat_fn).invoke(state)
        state.update(result)
        risk_section = state["risk_section"]
        macro_section = state["macro_section"]
        report_md = state["report_md"]
    except RuntimeError:
        model = "unavailable"
        # LLM failed: collectors may not have run yet in this process — collect directly.
        for key, fn in (("risk_data", collectors.collect_risk_data),
                        ("macro_data", collectors.collect_macro_data),
                        ("portfolio_data", collectors.collect_portfolio_data),
                        ("signal_data", collectors.collect_signal_data)):
            if key not in state:
                state[key] = fn(ch_client)
        risk_section = _data_summary(state)
        macro_section = _data_summary(state)
        report_md = _data_summary(state)

    # Deterministic sections appended unconditionally so the numbers always
    # appear, regardless of what the LLM editor chose to include.
    portfolio_section = _portfolio_md(state.get("portfolio_data", {}))
    signal_section = _signal_md(state.get("signal_data", {}))
    for section in (portfolio_section, signal_section):
        heading = section.splitlines()[0]
        if heading and heading not in report_md:
            report_md += "\n\n" + section
    # The disclaimer is unconditional — don't rely on LLM compliance.
    if "Not financial advice." not in report_md:
        report_md += ("\n\nPortfolio figures are strategy-implied "
                      "(research/backtest scope), not real holdings. "
                      "Not financial advice.\n")

    now = datetime.now(timezone.utc)
    rows = [
        [report_date, "risk", risk_section, model, now],
        [report_date, "macro", macro_section, model, now],
        [report_date, "portfolio", portfolio_section, model, now],
        [report_date, "signal", signal_section, model, now],
        [report_date, "report", report_md, model, now],
    ]
    ch_client.insert(
        f"{database_name()}.research_reports", rows,
        column_names=list(_SECTION_COLUMNS),
    )
    (out_dir / f"{report_date.isoformat()}.md").write_text(report_md)
    return report_md
