"""Online issuer-by-issuer option training and bid/ask backtest smoke path.

Option documents are generated from the raw warehouse for one issuer, trained,
and discarded. No option corpus or option Parquet is created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTaskSpec, MultiRateTransformer, MultiRateTransformerConfig,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import HITS_SUPERVISED_TASK_NAMES, ORACLE_SUPERVISED_TASK_NAMES
from scripts.backtest_dte_event_driven import _build_daily_quotes
from scripts.build_annual_option_documents import _load_raw_first_day, _select
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec, LabelBuildSpec, build_oracle_labels, build_return_and_speed_hits_labels,
)

ANNUAL_WINDOW = QUARTERLY_WINDOW = DAILY_WINDOW = 252
TASKS = (*ORACLE_SUPERVISED_TASK_NAMES, *HITS_SUPERVISED_TASK_NAMES)


def repair_mask(values: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Avoid all-masked attention rows for short issuer histories."""
    values = values.clone(); padding = padding.clone().bool()
    empty = padding.all(dim=1)
    if empty.any(): values[empty, -1] = 0.0; padding[empty, -1] = False
    leading = padding[:, 0]
    if leading.any(): values[leading, 0] = 0.0; padding[leading, 0] = False
    return values, padding


def norm_date(values):
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def make_groups(symbol: str, start_year: int, end_year: int, dte: int) -> pd.DataFrame:
    raw = _load_raw_first_day({symbol}, start_year=start_year, end_year=end_year)
    if raw.empty:
        return pd.DataFrame()
    selected = _select(raw, max_contracts=0, group_by_dte=True)
    selected = selected.loc[pd.to_numeric(selected["dte"], errors="coerce").eq(dte)].copy()
    if selected.empty:
        return selected
    selected["underlying_symbol"] = symbol
    selected["document_symbol"] = [f"OPT_{symbol}_{int(y)}_{str(t)[:1].upper()}_DTE_{dte}" for y, t in zip(selected.year, selected.option_type)]
    return selected


def make_targets(groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes = _build_daily_quotes(groups, set(groups.underlying_symbol))
    rows: list[pd.DataFrame] = []
    for symbol, frame in quotes.groupby("symbol", sort=False):
        prices = frame.sort_values("date").drop_duplicates("date")
        prices = pd.DataFrame({"date": prices.date, "high": prices.ask, "low": prices.bid, "adj_high": prices.ask, "adj_low": prices.bid, "close": prices.bid, "adj_close": prices.bid}).dropna()
        if len(prices) < 2:
            continue
        hits = build_return_and_speed_hits_labels({symbol: prices}, spec=HitsLabelSpec(max_hold=120, iterations=50))
        if not hits.empty:
            channels = ["long_hub", "long_authority", "short_hub", "short_authority", "speed_long_hub", "speed_long_authority", "speed_short_hub", "speed_short_authority"]
            out = pd.DataFrame({"symbol": symbol, "date": norm_date(hits.date)})
            for i, channel in enumerate(channels): out[TASKS[4 + i]] = np.nan_to_num(hits[channel].to_numpy("float32"), nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(out)
        oracle = build_oracle_labels([symbol], spec=LabelBuildSpec(k_params={"YE": [1]}, min_profit_pct=0.01, buy_execution="adj_high", sell_execution="adj_low", short_execution="adj_low", cover_execution="adj_high"), price_frames={symbol: prices})
        if oracle.label_rows:
            labels = pd.DataFrame(oracle.label_rows); labels["date"] = norm_date(labels["date"])
            out = pd.DataFrame({"symbol": symbol, "date": labels.date.drop_duplicates()})
            for i, label in enumerate(("is_oracle_buy", "is_oracle_sell", "is_oracle_short", "is_oracle_cover")):
                out[TASKS[i]] = out.date.isin(labels.loc[labels.label.eq(label), "date"]).astype("float32")
            rows.append(out)
    targets = pd.concat(rows, ignore_index=True).groupby(["symbol", "date"], as_index=False).max() if rows else pd.DataFrame(columns=["symbol", "date", *TASKS])
    if not targets.empty:
        targets.loc[:, list(TASKS)] = targets[list(TASKS)].fillna(0.0)
    return quotes, targets


def window(table, symbol, date, columns, length):
    rows = table.get(symbol, table.get("__empty__", pd.DataFrame()))
    if "date" in rows:
        rows = rows.loc[rows.date.le(date)].tail(length)
    values = np.full((length, len(columns)), np.nan, dtype="float32")
    padding = np.ones(length, dtype=bool)
    dates = np.empty(0, dtype="datetime64[ns]")
    if len(rows):
        values[-len(rows):] = rows[columns].to_numpy("float32")
        padding[-len(rows):] = False
        dates = norm_date(rows.date).to_numpy()
    # Inputs are standardized globally; zero is the feature mean for missing
    # numeric observations. Padding remains controlled by the mask.
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, padding, dates


def train_issuer(model, optimizer, issuer, groups, tables, device, dte, batch_size, *, fit: bool = True):
    quotes, targets = make_targets(groups)
    if quotes.empty or targets.empty:
        return pd.DataFrame()
    daily_by = {s: g for s, g in tables["daily"].groupby("symbol", sort=False)}
    annual_by = {s: g for s, g in tables["annual"].groupby("symbol", sort=False)}
    quarterly_by = {s: g for s, g in tables["quarterly"].groupby("symbol", sort=False)}
    samples = []
    for row in quotes.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        a, ap, _ = window(annual_by, issuer, date, tables["annual_columns"], ANNUAL_WINDOW)
        q, qp, _ = window(quarterly_by, issuer, date, tables["quarterly_columns"], QUARTERLY_WINDOW)
        d, dp, dd = window(daily_by, issuer, date, tables["daily_columns"], DAILY_WINDOW)
        y = np.zeros((DAILY_WINDOW, len(TASKS)), dtype="float32"); valid = np.zeros_like(y, dtype=bool)
        matching = targets.loc[targets.symbol.eq(row.symbol)].set_index("date")
        for i, value in enumerate(pd.to_datetime(dd).normalize()):
            if value in matching.index:
                y[DAILY_WINDOW - len(dd) + i] = matching.loc[value, list(TASKS)].to_numpy("float32"); valid[DAILY_WINDOW - len(dd) + i] = True
        samples.append((a, ap, q, qp, d, dp, y, valid))
    model.train()
    for start in range(0, len(samples), batch_size) if fit else []:
        batch = samples[start:start + batch_size]
        tensors = [torch.from_numpy(np.stack([x[i] for x in batch])).to(device) for i in range(8)]
        a, ap, q, qp, d, dp, y, valid = tensors
        a, ap = repair_mask(a, ap); q, qp = repair_mask(q, qp); d, dp = repair_mask(d, dp)
        sparse = torch.zeros((len(batch), 16, 17 * 8), device=device)
        sm = torch.ones((len(batch), 16), dtype=torch.bool, device=device); sm[:, -1] = False
        sparse, sm = repair_mask(sparse, sm)
        out = model(d, a, q, sparse, daily_padding_mask=dp.bool(), annual_padding_mask=ap.bool(), quarterly_padding_mask=qp.bool(), sparse_padding_mask=sm, compute_document_outputs=False)
        pred = torch.stack([out["token_outputs"][name].squeeze(-1) for name in TASKS], dim=-1)
        mask = valid.bool() & ~dp.bool().unsqueeze(-1)
        losses = []
        for i, name in enumerate(TASKS):
            if mask[..., i].any():
                losses.append(nn.functional.binary_cross_entropy_with_logits(pred[..., i][mask[..., i]], y[..., i][mask[..., i]]))
        if losses:
            optimizer.zero_grad(set_to_none=True); sum(losses).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval(); prediction_rows = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            tensors = [torch.from_numpy(np.stack([x[i] for x in batch])).to(device) for i in range(8)]
            a, ap, q, qp, d, dp, y, valid = tensors
            a, ap = repair_mask(a, ap); q, qp = repair_mask(q, qp); d, dp = repair_mask(d, dp)
            sparse = torch.zeros((len(batch), 16, 17 * 8), device=device); sm = torch.ones((len(batch), 16), dtype=torch.bool, device=device); sm[:, -1] = False
            sparse, sm = repair_mask(sparse, sm)
            out = model(d, a, q, sparse, daily_padding_mask=dp.bool(), annual_padding_mask=ap.bool(), quarterly_padding_mask=qp.bool(), sparse_padding_mask=sm, compute_document_outputs=False)
            pred = torch.stack([torch.sigmoid(out["token_outputs"][name].squeeze(-1))[:, -1] for name in TASKS], dim=-1).cpu().numpy()
            for index, values in enumerate(pred):
                prediction_rows.append({"symbol": quotes.iloc[start + index].symbol, "date": quotes.iloc[start + index].date, **{name: float(values[i]) for i, name in enumerate(TASKS)}})
    result = pd.DataFrame(prediction_rows).merge(quotes[["symbol", "date", "bid", "ask", "option_type"]], on=["symbol", "date"], how="inner")
    return result


def backtest(scores: pd.DataFrame, output_dir: Path) -> dict[str, float | int]:
    if scores.empty:
        return {"days": 0, "entries": 0, "exits": 0, "final_equity": 1.0, "total_return": 0.0}
    scores = scores.copy(); scores["date"] = norm_date(scores.date); scores = scores.sort_values(["date", "symbol"])
    by_key = scores.set_index(["date", "symbol"]); positions = {}; cash = 1.0; events = []; equity_rows = []
    last_quote_date = scores.groupby("symbol")["date"].max().to_dict()
    last_bid = {}
    for date in pd.date_range(scores.date.min(), scores.date.max(), freq="B"):
        for symbol, position in list(positions.items()):
            key = (date, symbol)
            if key in by_key.index:
                row = by_key.loc[key]
                if np.isfinite(row.bid) and float(row.bid) > 0: last_bid[symbol] = float(row.bid)
                exit_name = "hits_long_return_authority" if position["option_type"] == "call" else "hits_short_return_authority"
                if float(row[exit_name]) >= 0.5 and symbol in last_bid:
                    cash += position["units"] * last_bid[symbol] * 0.9995; events.append({"date": date, "symbol": symbol, "action": "exit"}); del positions[symbol]
            elif date > last_quote_date.get(symbol, date) and symbol in last_bid:
                cash += position["units"] * last_bid[symbol] * 0.9995; events.append({"date": date, "symbol": symbol, "action": "forced_exit", "reason": "quote_history_ended"}); del positions[symbol]
        candidates = []
        for (candidate_date, symbol), row in by_key.iterrows():
            if candidate_date != date or symbol in positions: continue
            entry_name = "hits_long_return_hub" if str(row.option_type) == "call" else "hits_short_return_hub"
            if float(row[entry_name]) >= 0.5 and np.isfinite(row.ask) and float(row.ask) > 0:
                candidates.append((float(row[entry_name]), symbol, str(row.option_type), float(row.ask)))
        candidates.sort(reverse=True)
        for signal, symbol, option_type, ask in candidates[:max(0, 5 - len(positions))]:
            allocation = min(cash, 1.0 / 5); units = allocation * 0.9995 / ask; cash -= allocation; positions[symbol] = {"units": units, "option_type": option_type}; last_bid[symbol] = float(by_key.loc[(date, symbol)].bid); events.append({"date": date, "symbol": symbol, "action": "enter"})
        value = cash + sum(position["units"] * last_bid.get(symbol, 0.0) for symbol, position in positions.items())
        equity_rows.append({"date": date, "equity": value, "positions": len(positions)})
    equity = pd.DataFrame(equity_rows); output_dir.mkdir(parents=True, exist_ok=True); equity.to_csv(output_dir / "daily.csv", index=False); pd.DataFrame(events).to_csv(output_dir / "events.csv", index=False)
    summary = {"days": len(equity), "entries": sum(x["action"] == "enter" for x in events), "exits": sum(x["action"] == "exit" for x in events), "final_equity": float(equity.equity.iloc[-1]), "total_return": float(equity.equity.iloc[-1] - 1.0), "max_drawdown": float((equity.equity / equity.equity.cummax() - 1).min())}
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False); return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--equity-corpus", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--symbols", required=True); p.add_argument("--test-symbols", default=""); p.add_argument("--dte", type=int, default=105); p.add_argument("--start-year", type=int, default=2025); p.add_argument("--end-year", type=int, default=2026); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--checkpoint", type=Path, help="Existing online model.pt to score without training"); p.add_argument("--inference-only", action="store_true", help="Disable all optimizer steps and backtest the loaded/current model"); args=p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    root=args.equity_corpus; manifest=json.loads((root/"manifest.json").read_text()); families=manifest["feature_families"]; cols=[f"value__{x}" for x in families]; rates={n:pd.read_parquet(root/f"{n}.parquet",columns=["symbol","date",*cols]) for n in ("annual","quarterly","daily")};
    for t in (rates["annual"], rates["quarterly"], rates["daily"]): t.symbol=t.symbol.astype(str).str.upper(); t.date=norm_date(t.date)
    means={}; scales={}
    for n,t in rates.items():
        v=t[cols].to_numpy("float64"); means[n]=np.nan_to_num(np.nanmean(v,axis=0)); scales[n]=np.where(np.nanstd(v,axis=0)>1e-6,np.nanstd(v,axis=0),1.0); t.loc[:,cols]=((v-means[n])/scales[n]).astype("float32")
    rates["annual_columns"]=cols; rates["quarterly_columns"]=cols; rates["daily_columns"]=cols
    tasks=tuple(MultiRateTaskSpec(name,level="token",output_dim=1,source="daily") for name in TASKS); config=MultiRateTransformerConfig(backbone="encoder_decoder",d_model=128,num_heads=8,layers=2,document_pool="mean",cacheable_rate_states=True)
    model=MultiRateTransformer({"annual":len(cols),"quarterly":len(cols),"daily":len(cols),"sparse":17*8},config=config,tasks=tasks,prediction_tasks=()).to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4) if not args.inference_only else None
    symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()]; test={x.strip().upper() for x in args.test_symbols.split(",") if x.strip()}; predictions=[]
    for epoch in range(args.epochs):
        for issuer in symbols:
            groups=make_groups(issuer,args.start_year,args.end_year,args.dte)
            if groups.empty: continue
            result=train_issuer(model,optimizer,issuer,groups,rates,torch.device(args.device),args.dte,args.batch_size,fit=(not args.inference_only and issuer not in test))
            if (args.inference_only or issuer in test) and epoch == args.epochs - 1: predictions.append(result)
            print(f"epoch={epoch+1} issuer={issuer} documents={len(groups)} rows={len(result)}",flush=True)
    pred=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(); pred.to_csv(args.output_dir/"online_predictions.csv",index=False); summary=backtest(pred,args.output_dir/"backtest"); print(json.dumps({"test_rows":len(pred),"test_symbols":sorted(test),"dte":args.dte,"checkpoint":str(args.checkpoint) if args.checkpoint else None,"inference_only":args.inference_only,"backtest":summary}))

if __name__ == "__main__": main()
