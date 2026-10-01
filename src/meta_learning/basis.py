"""Perp basis research: (perp mid/mark − spot close) / spot.

The delta-neutral carry trades Binance.US spot against a Hyperliquid perp, so
the basis is measured across exactly those venues: Hyperliquid perp mid/mark
against Binance.US spot close, aligned on the hour. A persistently positive
basis means the perp is rich (short-perp/long-spot also earns the premium);
large basis is a signal to enter and an input to carry risk.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src.backtesting.data import load_ohlcv
from src.ingestion.schemas import database_name, get_clickhouse_client
from src.ingestion.universe import universe

_MARK_COLUMNS = ["ts", "mark_price", "oracle_price", "mid_price", "open_interest"]


def load_perp_marks(client, symbol: str, start: datetime | None = None,
                    end: datetime | None = None) -> pd.DataFrame:
    clauses, params = ["symbol = {s:String}"], {"s": symbol}
    if start is not None:
        clauses.append("ts >= {start:DateTime64(3)}")
        params["start"] = start
    if end is not None:
        clauses.append("ts < {end:DateTime64(3)}")
        params["end"] = end
    rows = client.query(
        f"SELECT {', '.join(_MARK_COLUMNS)} FROM {database_name()}.perp_marks FINAL "
        f"WHERE {' AND '.join(clauses)} ORDER BY ts",
        parameters=params,
    ).result_rows
    df = pd.DataFrame(rows, columns=_MARK_COLUMNS)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def basis_series(client, symbol: str, interval: str = "1h") -> pd.Series:
    """Hourly basis (perp − spot) / spot, aligned across venues."""
    perp = load_perp_marks(client, symbol)
    spot = load_ohlcv(client, symbol, interval=interval)["close"]
    if perp.empty:
        return pd.Series(dtype=float)
    perp_price = perp["mid_price"].where(perp["mid_price"] > 0, perp["mark_price"])
    perp_h = perp_price.resample(interval).last()
    aligned = pd.concat({"perp": perp_h, "spot": spot}, axis=1).dropna()
    return (aligned["perp"] - aligned["spot"]) / aligned["spot"]


def basis_stats(basis: pd.Series) -> dict:
    if basis.empty:
        return {"n": 0}
    return {"n": int(len(basis)), "mean": float(basis.mean()),
            "std": float(basis.std()), "p05": float(basis.quantile(0.05)),
            "p95": float(basis.quantile(0.95)), "current": float(basis.iloc[-1])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Perp basis (Hyperliquid vs Binance.US spot)")
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--interval", default="1h")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    symbols = universe(ch) if args.symbols == "all" else tuple(args.symbols.split(","))
    print(f"{'symbol':6}{'n':>6}{'current':>11}{'mean':>10}{'p05':>10}{'p95':>10}  (bps)")
    for symbol in symbols:
        try:
            stats = basis_stats(basis_series(ch, symbol, interval=args.interval))
            if stats["n"] == 0:
                print(f"{symbol:6}{0:>6}                 (no aligned basis yet)")
                continue
            f = lambda x: x * 1e4
            print(f"{symbol:6}{stats['n']:>6}{f(stats['current']):>11.2f}"
                  f"{f(stats['mean']):>10.2f}{f(stats['p05']):>10.2f}{f(stats['p95']):>10.2f}")
        except Exception as exc:
            print(f"{symbol:6} FAILED: {exc}")


if __name__ == "__main__":
    main()
