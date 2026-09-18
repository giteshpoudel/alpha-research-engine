"""LLM agent nodes: thin prompt-in/text-out wrappers over llm.chat."""

from __future__ import annotations

import json

from src.research.llm import chat

_RISK_SYSTEM = (
    "You are a portfolio risk analyst for a crypto research desk. Analyze the "
    "provided data: signal states (strategy-implied book, not real holdings), "
    "30d correlations, realized volatility, and strategy drawdowns (max "
    "drawdown over the current OOS window). Write a risk section (max 300 "
    "words): exposure summary, concentration/correlation risks, volatility "
    "outliers, and what to watch. Be specific with numbers."
)

_MACRO_SYSTEM = (
    "You are a macro & sentiment analyst for a crypto research desk. Analyze "
    "the provided data: sentiment metrics with spike flags, 24h price changes, "
    "funding rate extremes, and top social posts. Write a macro section (max "
    "300 words): market trend, sentiment divergences (price vs sentiment), and "
    "likely catalysts. Be specific with numbers."
)

_REPORT_SYSTEM = (
    "You are the editor of a daily crypto research report. Synthesize the Risk "
    "and Macro sections plus the raw data into a markdown daily report with: "
    "title with date, a one-paragraph executive summary, the Risk section, the "
    "Macro section, and a closing note. Include this disclaimer verbatim at "
    "the end: 'Portfolio figures are strategy-implied (research/backtest "
    "scope), not real holdings. Not financial advice.'"
)


def risk_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps(state["risk_data"], default=str)
    return {"risk_section": chat_fn(_RISK_SYSTEM, user)}


def macro_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps(state["macro_data"], default=str)
    return {"macro_section": chat_fn(_MACRO_SYSTEM, user)}


def report_agent(state: dict, chat_fn=chat) -> dict:
    user = json.dumps({
        "date": str(state["report_date"]),
        "risk_section": state["risk_section"],
        "macro_section": state["macro_section"],
        "risk_data": state["risk_data"],
        "macro_data": state["macro_data"],
        "portfolio_data": state.get("portfolio_data", {}),
        "signal_data": state.get("signal_data", {}),
    }, default=str)
    return {"report_md": chat_fn(_REPORT_SYSTEM, user)}
