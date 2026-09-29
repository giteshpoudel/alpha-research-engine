"""Order-flow-imbalance research on the collected trade tape.

The tape is forward-collected (no historical endpoint), so this module defines
the measurement and lets it run as data accumulates: bucket trades, compute
signed-volume imbalance (OBI), and measure whether OBI predicts the next-bucket
return. The same OBI is the core input to market making (toxicity / fair value).
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src.ingestion.microstructure import order_flow_imbalance
from src.ingestion.schemas import database_name, get_clickhouse_client

TRADE_COLUMNS = ["ts", "price", "qty", "is_buyer_maker"]


def load_trades(client, symbol: str, start: datetime | None = None,
                end: datetime | None = None) -> pd.DataFrame:
    """Public trades for one symbol, oldest-first."""
    clauses, params = ["symbol = {s:String}"], {"s": symbol}
    if start is not None:
        clauses.append("ts >= {start:DateTime64(3)}")
        params["start"] = start
    if end is not None:
        clauses.append("ts < {end:DateTime64(3)}")
        params["end"] = end
    rows = client.query(
        f"SELECT ts, price, qty, is_buyer_maker FROM {database_name()}.market_trades FINAL "
        f"WHERE {' AND '.join(clauses)} ORDER BY ts",
        parameters=params,
    ).result_rows
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def order_flow_predictiveness(trades: pd.DataFrame, horizon: int = 1,
                              bucket: str = "1min") -> dict:
    """Correlation between bucket OBI(t) and the next-``horizon`` bucket return."""
    if trades.empty:
        return {"n_buckets": 0, "corr": None}
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    signed = df["qty"] * (1 - 2 * df["is_buyer_maker"])
    grouped = (df.assign(signed=signed)
                 .resample(bucket)
                 .agg(net=("signed", "sum"), vol=("qty", "sum"), price=("price", "last"))
                 .dropna())
    if len(grouped) < horizon + 2:
        return {"n_buckets": int(len(grouped)), "corr": None}
    grouped["obi"] = grouped["net"] / grouped["vol"]
    grouped["fwd"] = grouped["price"].shift(-horizon) / grouped["price"] - 1.0
    valid = grouped.dropna()
    if len(valid) < 3:
        return {"n_buckets": int(len(valid)), "corr": None}
    return {"n_buckets": int(len(valid)),
            "corr": float(valid["obi"].corr(valid["fwd"]))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Order-flow imbalance research")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--bucket", default="1min")
    args = parser.parse_args(argv)

    ch = get_clickhouse_client()
    trades = load_trades(ch, args.symbol)
    print(f"{args.symbol}: {len(trades)} trades")
    if trades.empty:
        print("  no tape yet — the forward collector needs time to accumulate")
        return
    print(f"  current OBI (last 500 trades): {order_flow_imbalance(trades, window=500):+.3f}")
    stats = order_flow_predictiveness(trades, horizon=args.horizon, bucket=args.bucket)
    print(f"  OBI->return corr (bucket={args.bucket}, horizon={args.horizon}): {stats}")


if __name__ == "__main__":
    main()
