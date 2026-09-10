"""Bounded Polars replay for funded issuer/instrument selection experiments.

The existing framework adapters require Pandas. This small cash/position loop
implements the declared long-only experiment without a dataframe bridge; it is
not a general execution engine. Oracle policy has one position per issuer;
HITS policy uses a shared top-k book. Neither has rank-driven exits.
"""

from datetime import datetime, timedelta
import hashlib
import math
from pathlib import Path

import polars as pl

from quant_orchestrator.artifact_contracts import StrategyArtifactBundle, write_strategy_artifacts


def replay_multirate(
    root: Path,
    predictions: Path,
    output: Path,
    *,
    start: str,
    end: str,
    initial_cash=100_000.0,
    fee_bps=5.0,
    slippage_bps=5.0,
    baseline=False,
    policy="oracle",
    top_k=5,
    entry_threshold=0.5,
    exit_threshold=0.5,
    warehouse=None,
):
    from quant_warehouse import Warehouse
    from quant_warehouse.platforms.data_providers.thetadata.options import (
        read_thetadata_eod_option_chain,
    )

    if policy not in {"oracle", "hits"} or top_k < 1:
        raise ValueError("policy must be oracle or hits and top_k must be positive")
    if baseline and policy != "oracle":
        raise ValueError("baseline cannot be combined with HITS policy")
    if not all(math.isfinite(v) for v in (entry_threshold, exit_threshold)):
        raise ValueError("HITS thresholds must be finite")
    warehouse = warehouse or Warehouse()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    taxonomy = pl.read_csv(root / "taxonomy.csv")
    metadata = {r["symbol"]: r for r in taxonomy.iter_rows(named=True)}
    issuers = sorted(taxonomy["issuer"].unique())
    first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    quote_paths, distribution_paths, underlying_paths = [], [], []
    # Adjusted equity units are synthetic total-return units; do not add
    # dividends or split adjustments a second time.
    for row in taxonomy.filter(pl.col("asset_class") != "option").iter_rows(named=True):
        symbol = row["symbol"]
        frame = warehouse.read_prices(
            symbol, provider="fmp", start=start, end=end, adjustment="splits_and_dividends"
        )
        if frame.is_empty():
            raise ValueError(f"Missing adjusted execution prices for {symbol}")
        frame = frame.select(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.lit(symbol).alias("symbol"),
            pl.col("close").alias("bid"),
            pl.col("close").alias("ask"),
        )
        path = inputs / f"prices_{symbol}.parquet"
        frame.write_parquet(path)
        quote_paths.append(path)
    for issuer in sorted(
        taxonomy.filter(pl.col("asset_class") == "option")["underlying_symbol"].unique()
    ):
        raw = warehouse.read_prices(issuer, provider="fmp", start=start, end=end, adjustment="unadjusted")
        path = inputs / f"underlying_raw_{issuer}.parquet"
        raw.select(pl.col("date").cast(pl.Datetime("ns")), pl.lit(issuer).alias("symbol"), "close").write_parquet(path)
        underlying_paths.append(path)
        selected = taxonomy.filter(
            (pl.col("asset_class") == "option") & (pl.col("underlying_symbol") == issuer)
        )["symbol"].to_list()
        day = first
        while day <= last:
            stop = min(day + timedelta(days=6), last)
            quotes = read_thetadata_eod_option_chain(
                issuer,
                start_date=day,
                end_date=stop,
                columns=["snapshot_date", "contract_symbol", "bid", "ask"],
                backend=warehouse.backend,
            )
            quotes = quotes.filter(
                pl.col("contract_symbol").is_in(selected)
                & (pl.col("bid") >= 0)
                & (pl.col("ask") >= pl.col("bid"))
                & pl.col("bid").is_finite()
                & pl.col("ask").is_finite()
            )
            if not quotes.is_empty():
                quotes = quotes.select(
                    pl.col("snapshot_date").cast(pl.Datetime("ns")).alias("date"),
                    pl.col("contract_symbol").alias("symbol"),
                    "bid",
                    "ask",
                )
                path = inputs / f"options_{issuer}_{day:%Y%m%d}.parquet"
                quotes.write_parquet(path)
                quote_paths.append(path)
            day = stop + timedelta(days=1)
    quotes = pl.scan_parquet(quote_paths)
    underlying_quotes = pl.scan_parquet(underlying_paths) if underlying_paths else None
    distributions = (
        pl.scan_parquet(distribution_paths)
        if distribution_paths
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "date": pl.Datetime("ns"),
                "payment_date": pl.Datetime("ns"),
                "amount": pl.Float64,
            }
        ).lazy()
    )
    scores = (
        pl.scan_csv(predictions, try_parse_dates=True)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .filter(pl.col("date").is_between(first, last))
    )
    dates = (
        quotes.select("date").unique().sort("date").collect(engine="streaming")["date"].to_list()
    )
    cash = float(initial_cash)
    positions = {}
    marks = {}
    previous = {}
    receivables = []
    actions = []
    trades = []
    curve = []
    stale = 0
    trade_id = 0
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    def hits_column(symbol, kind):
        tax = metadata[symbol]
        side = "short" if tax["asset_class"] == "option" and str(tax.get("option_type", "")).lower().startswith("p") else "long"
        return f"hits_{side}_return_{kind}"

    def sell(symbol, date, price, reason):
        nonlocal cash
        pos = positions.pop(symbol)
        proceeds = pos["units"] * price
        charge = proceeds * fee
        cash += proceeds - charge
        actions.append(
            dict(
                date=date,
                symbol=symbol,
                action="exit_long",
                price=price,
                quantity=pos["units"],
                reason=reason,
                fee=charge,
            )
        )
        trades.append(
            dict(
                trade_id=pos["trade_id"],
                symbol=symbol,
                side="long",
                entry_date=pos["entry_date"],
                exit_date=date,
                equity_entry_notional=pos["cost"],
                entry_price=pos["entry_price"],
                exit_price=price,
                pnl=proceeds - charge - pos["cost"] + pos["distributions"],
                asset_class=metadata[symbol]["asset_class"],
                reason=reason,
            )
        )

    for date in dates:
        day = {
            r["symbol"]: r
            for r in quotes.filter(pl.col("date") == date)
            .collect(engine="streaming")
            .iter_rows(named=True)
        }
        # Entitlement is determined before ex-date executions; cash arrives on
        # payment date, including after a position has been sold.
        for row in (
            distributions.filter(pl.col("date") == date)
            .collect(engine="streaming")
            .iter_rows(named=True)
        ):
            if row["symbol"] in positions:
                value = positions[row["symbol"]]["units"] * row["amount"]
                positions[row["symbol"]]["distributions"] += value
                receivables.append((row["payment_date"], value))
        cash += sum(value for pay, value in receivables if pay <= date)
        receivables = [(pay, value) for pay, value in receivables if pay > date]
        for symbol, pos in list(positions.items()):
            tax = metadata[symbol]
            quote = day.get(symbol)
            signal = previous.get(symbol)
            if quote:
                marks[symbol] = quote["bid"]
            else:
                stale += 1
            expiry = datetime.fromisoformat(tax["expiration"]) if tax.get("expiration") else None
            if expiry and date >= expiry:
                underlying = underlying_quotes.filter((pl.col("symbol") == tax["underlying_symbol"]) & (pl.col("date") == date)).collect(engine="streaming")
                if underlying.is_empty():
                    raise ValueError(f"Missing expiry underlying valuation for {symbol}")
                sign = 1 if str(tax["option_type"]).lower().startswith("c") else -1
                intrinsic = max(0.0, sign * (underlying["close"][0] - tax["strike"]))
                sell(symbol, date, intrinsic, "expiration_intrinsic")
            elif (
                not baseline
                and signal
                and quote
                and (
                    signal[hits_column(symbol, "authority")] >= exit_threshold
                    if policy == "hits" else
                    (signal["oracle_is_buy"] <= signal["oracle_is_short"]
                     or signal["oracle_is_sell"] >= 0.5)
                )
            ):
                sell(symbol, date, quote["bid"] * (1 - slip), "hits_authority_exit" if policy == "hits" else "oracle_exit")
        occupied = {metadata[s]["issuer"] for s in positions}
        nav = (
            cash
            + sum(p["units"] * marks[s] for s, p in positions.items())
            + sum(v for _, v in receivables)
        )
        # Free slots are filled from prior-session scores. Existing
        # holdings are never rotated simply because another score is higher.
        candidates = []
        for symbol, signal in previous.items():
            tax = metadata[symbol]
            if (policy != "hits" and tax["issuer"] in occupied) or symbol not in day or symbol in positions:
                continue
            if baseline:
                eligible = tax["asset_class"] == "equity" and symbol == tax["underlying_symbol"]
            elif policy == "hits":
                eligible = signal[hits_column(symbol, "hub")] >= entry_threshold
            else:
                eligible = (
                    signal["oracle_is_buy"] >= 0.5
                    and signal["oracle_is_buy"] > signal["oracle_is_short"]
                )
            expiry = datetime.fromisoformat(tax["expiration"]) if tax.get("expiration") else None
            if date < dates[-1] and eligible and (expiry is None or expiry > date):
                candidates.append((signal[hits_column(symbol, "hub")] if policy == "hits" else signal["hits_long_return_hub"], symbol))
        for rank, symbol in sorted(candidates, key=lambda item: (-item[0], item[1])):
            tax = metadata[symbol]
            if policy == "hits" and len(positions) >= top_k:
                break
            if policy != "hits" and tax["issuer"] in occupied:
                continue
            price = day[symbol]["ask"] * (1 + slip)
            if not math.isfinite(price) or price <= 0:
                continue
            lot = 100 if tax["asset_class"] == "option" else 1
            units = math.floor(min(cash, nav / (top_k if policy == "hits" else len(issuers))) / (price * (1 + fee) * lot)) * lot
            if units <= 0:
                continue
            cost = units * price * (1 + fee)
            cash -= cost
            trade_id += 1
            positions[symbol] = dict(
                units=units,
                cost=cost,
                entry_date=date,
                entry_price=price,
                distributions=0.0,
                trade_id=trade_id,
            )
            marks[symbol] = day[symbol]["bid"]
            occupied.add(tax["issuer"])
            actions.append(
                dict(
                    date=date,
                    symbol=symbol,
                    action="enter_long",
                    price=price,
                    quantity=units,
                    reason="hits_top_k" if policy == "hits" else "issuer_slot",
                    fee=units * price * fee,
                )
            )
        if date == dates[-1]:
            for symbol in list(positions):
                if symbol not in day:
                    raise ValueError(f"Missing terminal liquidation quote for {symbol}")
                sell(symbol, date, day[symbol]["bid"] * (1 - slip), "fold_end")
        nav = (
            cash
            + sum(p["units"] * marks[s] for s, p in positions.items())
            + sum(v for _, v in receivables)
        )
        curve.append(
            dict(
                date=date,
                equity=nav,
                cash=cash,
                receivables=sum(v for _, v in receivables),
                positions=len(positions),
            )
        )
        previous = {
            r["symbol"]: r
            for r in scores.filter(pl.col("date") == date)
            .collect(engine="streaming")
            .iter_rows(named=True)
        }
    equity = pl.DataFrame(curve).with_columns(
        pl.col("equity").pct_change().fill_null(0.0).alias("return")
    )
    equity.write_parquet(output / "equity_curve.parquet")
    drawdown = equity.select((pl.col("equity") / pl.col("equity").cum_max() - 1).min()).item()
    summary = dict(
        initial_cash=initial_cash,
        final_equity=equity["equity"][-1],
        total_return=equity["equity"][-1] / initial_cash - 1,
        max_drawdown=drawdown,
        trades=len(trades),
        stale_position_marks=stale,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        policy=(
            "funded issuer common-equity buy-and-hold"
            if baseline
            else ("funded long-only shared top-k, previous-session HITS hub entry and authority exit"
                  if policy == "hits" else "funded long-only, one instrument per issuer, previous-session Oracle direction, HITS entry ordering")
        ),
        hits_configuration=({"top_k": top_k, "entry_threshold": entry_threshold,
                             "exit_threshold": exit_threshold,
                             "channels": "long for equities/calls; short for puts (prior DTE policy)",
                             "issuer_cap": False} if policy == "hits" else None),
        cashflows="split-and-dividend-adjusted equity prices; no separate corporate-action cashflows",
        equity_price_adjustment="splits_and_dividends",
        options="100-share lots, bid/ask execution, expiry intrinsic cash-equivalent valuation, no rolls",
        limitations=[
            "Equity quantities are synthetic adjusted-price units, not historical share counts",
            "No short borrowing or naked-option margin simulation",
            "Early exercise, assignment and taxes are not modeled",
            "HITS scores are per-instrument graph scores, not calibrated cross-asset returns",
            "Universe is a small retrospective validation roster, not a survivorship-free investment universe",
        ],
    )
    trade_frame = (
        pl.DataFrame(trades, infer_schema_length=None)
        if trades
        else pl.DataFrame(
            schema={
                "trade_id": pl.Int64,
                "symbol": pl.String,
                "side": pl.String,
                "entry_date": pl.Datetime,
                "exit_date": pl.Datetime,
            }
        )
    )
    action_frame = (
        pl.DataFrame(actions, infer_schema_length=None)
        if actions
        else pl.DataFrame(
            schema={
                "date": pl.Datetime,
                "symbol": pl.String,
                "action": pl.String,
                "price": pl.Float64,
            }
        )
    )
    scored = scores.join(
        quotes.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("close")),
        on=["symbol", "date"],
    )
    scored.sink_parquet(output / "scored_panel.parquet")
    summary["input_sha256"] = {}
    for path in [*quote_paths, *distribution_paths, *underlying_paths, predictions, root / "taxonomy.csv"]:
        with path.open("rb") as handle:
            summary["input_sha256"][str(path.resolve())] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    write_strategy_artifacts(
        StrategyArtifactBundle(
            action_tape=action_frame,
            trade_list=trade_frame,
            summary=summary,
            strategy_name="issuer_equity_hold" if baseline else ("multirate_hits_top_k" if policy == "hits" else "multirate_oracle_hits"),
        ),
        output,
        extra_paths={
            "scored_panel": (output / "scored_panel.parquet").resolve(),
            "equity_curve": (output / "equity_curve.parquet").resolve(),
        },
    )
    return summary
