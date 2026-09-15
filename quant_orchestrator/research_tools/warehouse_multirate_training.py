"""Train and evaluate the multi-rate model while consuming warehouse documents."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import polars as pl
import torch

from .warehouse_multirate import WarehouseAnnualStream, SPARSE_FAMILIES
from .sampled_options import SELECTION_POLICY
from .annual_memory import AnnualMemory, ANNUAL_CONTRACT
from .multirate_batch import BatchTensors
from .multirate_training_step import make_training_step
from .multirate_objectives import family_channels
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTransformer, MultiRateTransformerConfig, Task, add_subtoken_temporal_tasks,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    DOCUMENT_TASK_NAMES, SUPERVISED_TARGET_TASK_NAMES, PREDICTION_TASK_NAMES,
    HITS_SUPERVISED_TASK_NAMES,
)


def document_batches(documents, batch_size):
    """Never put two years of one instrument in the same recurrent step."""
    batch, symbols = [], set()
    for item in documents:
        if batch and (len(batch) == batch_size or item['symbol'] in symbols):
            yield batch
            batch, symbols = [], set()
        batch.append(item)
        symbols.add(item['symbol'])
    if batch:
        yield batch


def prefetch_batches(batches):
    """One CPU batch overlaps the current GPU update; no corpus materialization."""
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, iterator, None)
        while True:
            batch = future.result()
            if batch is None:
                break
            future = pool.submit(next, iterator, None)
            yield batch


class EpochProgress:
    """Bound streaming progress output without counting/materializing the corpus."""
    def __init__(self, upper_bound, limit):
        if not 1 <= limit <= 10:
            raise ValueError('--progress-updates-per-epoch must be between 1 and 10')
        self.upper_bound = max(1, upper_bound)
        self.limit = limit
        self.bucket = 0
        self.finished = False

    def due(self, processed, *, complete=False):
        if self.finished:
            return False
        if complete:
            self.finished = True
            return True
        bucket = min(self.limit-1, processed*self.limit//self.upper_bound)
        if bucket > self.bucket:
            self.bucket = bucket
            return True
        return False


def predict_batch(model, batch, memory, layout):
    device = next(model.parameters()).device
    stack = BatchTensors(batch, device)
    widths = {**{r:tuple(layout.values()) for r in ('annual','quarterly','daily')}, 'sparse':(8,)*len(SPARSE_FAMILIES)}
    values, padding = {}, {}
    for r in widths:
        values[r], padding[r] = stack(r).clone(), stack(r+'_padding').clone()
        empty = padding[r].all(1)
        values[r][empty,-1] = 0
        padding[r][empty,-1] = False
        values[r][:,0] = 0
        padding[r][:,0] = False
    issuer = {}
    for r in ('daily','sparse'):
        v,p = stack('issuer_'+r).clone(), stack('issuer_'+r+'_padding').clone()
        v[:,0]=0;p[:,0]=False
        issuer[r]=dict(values=v,padding=p,dates=stack('issuer_'+r+'_timestamps'))
    result = model(values['daily'],values['annual'],values['quarterly'],values['sparse'],
        **{r+'_padding_mask':padding[r] for r in widths},
        **{r+'_dates':stack(r+'_timestamps') for r in widths},
        **{r+'_family_presence':family_channels(torch.isfinite(stack(r)) & ~stack(r+'_padding').unsqueeze(-1), widths[r]).any(-1) for r in widths},
        daily_modality_ids=torch.tensor([0 if b['asset_class']=='equity' else 1 for b in batch],device=device)[:,None].expand(-1,values['daily'].shape[1]),
        issuer_streams=issuer,compute_document_outputs=False,annual_memory=memory.inputs(batch,model))
    memory.update(batch,result)
    predictions={name:(value.sigmoid() if name not in HITS_SUPERVISED_TASK_NAMES else value).squeeze(-1).cpu()
                 for name,value in result['token_outputs'].items()}
    rows=[]
    for i,item in enumerate(batch):
        for j,(date,valid) in enumerate(zip(item['daily_dates'],item['daily_score_valid']),1):
            if valid:
                rows.append(dict(symbol=item['symbol'],underlying_symbol=item['underlying_symbol'],asset_class=item['asset_class'],date=date,
                    **{name:float(value[i,j]) for name,value in predictions.items()},
                    **({k:item[k] for k in ('option_type','expiration','settlement')} if item['asset_class']=='option' else {})))
    return rows


def evaluate_epoch(model, stream, args, epoch):
    output=Path(args.output_dir)/'epoch_validation'/f'epoch_{epoch:04d}'
    output.mkdir(parents=True,exist_ok=False)
    start,end=map(datetime.fromisoformat,(args.prediction_start_date,args.prediction_end_date))
    memory=AnnualMemory();memory.begin_epoch(epoch)
    model.eval();rows=[];prices=[];began=perf_counter()
    with torch.inference_mode():
        for step,batch in enumerate(prefetch_batches(document_batches(stream.documents(batch_size=args.batch_size,training=False,start=start,end=end,include_options=False),args.batch_size)),1):
            rows.extend(predict_batch(model,batch,memory,stream.layout))
            for item in batch:
                prices.append(item['prices'].select('date','open','high','low','close','volume').with_columns(
                    pl.lit(item['symbol']).alias('symbol'),pl.lit(item['asset_class']).alias('asset_class')))
            print(f'[warehouse-inference] epoch={epoch} batch={step} scores={len(rows)} seconds={perf_counter()-began:.1f}',flush=True)
    scores=pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col('date').cast(pl.Datetime('ns')))
    quotes=pl.concat(prices,how='diagonal_relaxed').with_columns(pl.col('date').cast(pl.Datetime('ns')))
    if scores.select(pl.struct('symbol','date').is_duplicated().any()).item():
        raise ValueError('Duplicate streamed predictions')
    expected=quotes.filter(pl.col('close').is_finite()).select('symbol','date').unique()
    if expected.join(scores.select('symbol','date'),on=['symbol','date'],how='anti').height:
        raise ValueError('Streamed inference omitted priced dates')
    if scores.select(pl.any_horizontal([~pl.col(c).is_finite() for c in SUPERVISED_TARGET_TASK_NAMES]).any()).item():
        raise ValueError('Nonfinite streamed predictions')
    scores.write_parquet(output/'predictions.parquet');quotes.write_parquet(output/'prices.parquet')
    inference_seconds=perf_counter()-began
    from quant_orchestrator.platforms.backtesting_frameworks.existing_multirate_backtest import run_existing_multirate_backtest
    from quant_orchestrator.platforms.backtesting_frameworks.equity_option_trade_backtest import run_equity_option_trade_backtest
    reports=[];began=perf_counter()
    from collections import OrderedDict
    from .warehouse_multirate import inference_day
    query_documents=OrderedDict()
    def rank_candidates(symbol, year, entry, members, paths):
        batch=[]
        for identity in members['document_symbol']:
            key=(identity,year)
            if key not in query_documents:
                query_documents[key]=stream.sample(symbol,year,option_symbol=identity,members=members,
                    prices=paths.filter(pl.col('symbol')==identity),training=False,
                    start=datetime(year,1,1),end=min(end,datetime(year,12,31)))
                while len(query_documents)>16:
                    query_documents.popitem(last=False)
            query_documents.move_to_end(key)
            batch.append(inference_day(query_documents[key],entry))
        # Cache raw features only. Every queried day gets fresh option predictions
        # with empty memory; no preceding/future rows enter the model query.
        with torch.inference_mode():
            return predict_batch(model,batch,AnnualMemory(),stream.layout)
    for year in range(start.year,end.year+1):
        s=scores.filter(pl.col('date').dt.year()==year)
        p=quotes.filter(pl.col('date').dt.year()==year)
        if s.is_empty() or p.is_empty():
            raise ValueError(f'Missing equity backtest coverage for {year}')
        equity_path=output/str(year)/'equity'
        result=run_existing_multirate_backtest(s.lazy(),p.select('symbol','date','close').lazy(),equity_path,
            next_session_execution=True)
        for report in result:
            report=dict(year=year,asset_class='equity',**report);reports.append(report)
            print('[warehouse-backtest-book] '+json.dumps(dict(epoch=epoch,**report)),flush=True)
        trades=pl.read_parquet(equity_path/'trade_windows.parquet').to_pandas()
        result=run_equity_option_trade_backtest(trades,stream,rank_candidates,output/str(year)/'option',year=year,
            dates=sorted(p['date'].unique().to_list()),capacity=min(20,p['symbol'].n_unique()))
        for report in result:
            report=dict(year=year,asset_class='option',**report);reports.append(report)
            print('[warehouse-backtest-book] '+json.dumps(dict(epoch=epoch,**report)),flush=True)
    (output/'results.json').write_text(json.dumps(reports,indent=2))
    timing=dict(inference_seconds=inference_seconds,backtest_seconds=perf_counter()-began,predictions=scores.height,
                inference_initialization='empty_memory_no_warmup', inference_assets=['equity'], training_option_selection=SELECTION_POLICY, supervised_context_order=['annual','quarterly','daily','sparse','instrument'])
    (output/'timing.json').write_text(json.dumps(timing,indent=2))
    print('[warehouse-backtest] '+json.dumps(dict(epoch=epoch,reports=reports,**timing)),flush=True)
    return reports


def run_warehouse_training(args):
    started=perf_counter()
    EpochProgress(1,args.progress_updates_per_epoch)  # Validate before opening the warehouse.
    if not args.min_market_cap>0 or args.epochs<1 or args.batch_size<1:
        raise ValueError('Positive market cap, epochs and batch size are required')
    if not args.train_end_date or not args.prediction_start_date or not args.prediction_end_date:
        raise ValueError('Warehouse training requires explicit training cutoff and prediction start/end dates')
    first,cutoff,start,end=map(datetime.fromisoformat,(args.warehouse_start_date,args.train_end_date,args.prediction_start_date,args.prediction_end_date))
    if not first<cutoff<=start<=end or (cutoff.month,cutoff.day)!=(1,1):
        raise ValueError('Require warehouse start < January 1 training cutoff <= prediction start <= prediction end')
    if args.warehouse_option_start_date:
        option_start = datetime.fromisoformat(args.warehouse_option_start_date)
        if (option_start.month, option_start.day) != (1, 1) or max(first, option_start) >= cutoff:
            raise ValueError('--warehouse-option-start-date must be January 1 before the training cutoff')
    if args.checkpoint or args.resume_training or args.inference_only:
        raise ValueError('Warehouse streaming starts a fresh run; checkpoint continuation is not implemented for this loader')
    if args.sequence_mode not in (None,'annual_memory'):
        raise ValueError('Warehouse streaming uses annual_memory documents')
    if args.mixed_precision or args.fp8 or args.compile_model or args.max_samples or args.validation_fraction:
        raise ValueError('This warehouse streaming path requires full-universe FP32 training')
    if args.issuer_context!='full' or args.legacy_rate_fusion or args.attention_backend!='pytorch':
        raise ValueError('Warehouse annual streaming requires full issuer context and the PyTorch date-aware model')
    if args.optimizer!='adamw' or args.grad_accumulation_steps!=1 or args.training_sequence_stride:
        raise ValueError('Warehouse streaming uses AdamW, one optimizer step per batch, and annual documents')
    if args.mrl_dimensions.strip() or not args.disable_document_tasks:
        raise ValueError('Warehouse streaming requires --mrl-dimensions "" --disable-document-tasks')
    if args.skip_predictions:
        raise ValueError('Warehouse validation includes equity and option backtests; omit --skip-predictions')
    args.sequence_mode='annual_memory';args.disable_document_tasks=True;args.issuer_context='full'
    args.output_dir.mkdir(parents=True,exist_ok=False)
    args._warehouse_run_started=True
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(dataset_mode='warehouse_on_demand',normalization='signed_log1p_div10',document_contract=ANNUAL_CONTRACT,
        options='annual hindsight-filtered sample; score equities first, then model-rank surviving options for triggered equity trades and use held-option Oracle exits',
        option_selection=SELECTION_POLICY, supervised_context_order=['annual','quarterly','daily','sparse','instrument'])
    (args.output_dir/'configuration.json').write_text(json.dumps(config,indent=2))
    stream=WarehouseAnnualStream(min_market_cap=args.min_market_cap,start=args.warehouse_start_date,
        end=args.prediction_end_date,cutoff=args.train_end_date,output=args.output_dir,option_start=args.warehouse_option_start_date)
    print(f'[warehouse-stream] metadata_ready_seconds={perf_counter()-started:.2f} equities={len(stream.prices)} option_underlyings={len(stream.expected_option_symbols)} corpus_built=false',flush=True)
    device=torch.device(args.device);torch.manual_seed(args.seed)
    widths={**{r:tuple(stream.layout.values()) for r in ('annual','quarterly','daily')},'sparse':(8,)*len(SPARSE_FAMILIES)}
    family_names=[*stream.layout,*SPARSE_FAMILIES]
    bundle=add_subtoken_temporal_tasks([],family_names,{n:['unused'] for n in DOCUMENT_TASK_NAMES[1:]},feature_dimensions=widths)
    model=MultiRateTransformer({**{r:len(stream.features) for r in ('annual','quarterly','daily')},'sparse':8*len(SPARSE_FAMILIES)},
        config=MultiRateTransformerConfig(backbone='encoder_decoder',d_model=args.d_model,num_heads=args.num_heads,layers=args.layers,
            document_pool='mean',max_position=512,cacheable_rate_states=True,learned_aggregation_gate=args.learned_aggregation_gate),
        feature_families={**{r:stream.layout for r in ('annual','quarterly','daily')},'sparse':dict.fromkeys(SPARSE_FAMILIES,8)},
        modalities=['equity','option'],tasks=bundle.supervised_tasks,prediction_tasks=bundle.prediction_tasks).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    prediction_names={n for n in PREDICTION_TASK_NAMES if args.self_supervision=='both' or n.startswith(args.self_supervision+'_')}
    tasks=tuple(Task(t.name,t.spec,args.reconstruction_weight if t.name in prediction_names else t.loss_weight)
        for t in bundle.tasks if t.name in SUPERVISED_TARGET_TASK_NAMES or t.name in prediction_names)
    memory=AnnualMemory();clock=SimpleNamespace(current_epoch=0,current_step=0)
    observations=Counter();family_observations=Counter();loss_sums=Counter()
    training_step=make_training_step(args=args,trainer=clock,device=device,annual_state=memory,
        reconstruction_widths=widths,feature_family_dimensions=stream.layout,sparse_input_families=list(SPARSE_FAMILIES),raw_sparse_columns=list(range(8)),
        family_names=family_names,asset_class_ids={'equity':0,'option':1},enabled_document_tasks=(),prediction_names=prediction_names,
        expected_task_names=SUPERVISED_TARGET_TASK_NAMES+PREDICTION_TASK_NAMES,mrl_dimensions=(),task_observations=observations,
        task_family_observations=family_observations,task_loss_sums=loss_sums,epoch_clocks={},alignment_loss=None)
    # Only metadata already loaded for the selected universe is needed. Options
    # may have fewer survivors; this is explicitly an upper bound, not a prepass.
    document_upper_bound = sum(
        frame.filter(pl.col('date') < stream.cutoff)['date'].dt.year().n_unique()
        for frame in stream.prices.values()) + sum(
        sum(year < stream.cutoff.year for year in years) * 2 * SELECTION_POLICY['contracts_per_side']
        for years in stream.option_years.values())
    first_update=None
    def checkpoint(epoch,batch,complete,loss):
        payload=dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),configuration=config,
            feature_schema=stream.features,normalization='signed_log1p_div10',asset_classes=['equity','option'],
            annual_memory_state=memory.state_dict(),task_observations=dict(observations),task_family_observations=dict(family_observations),
            task_loss_sums=dict(loss_sums),metrics=dict(epoch=epoch,batch=batch,epoch_complete=complete,loss=loss),
            torch_rng_state=torch.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        temporary=args.output_dir/'checkpoint.tmp.pt';torch.save(payload,temporary);temporary.replace(args.output_dir/'checkpoint_latest.pt')
        if complete:
            torch.save(payload,args.output_dir/f'epoch_{epoch:04d}.pt')
    for epoch in range(1,args.epochs+1):
        clock.current_epoch=epoch;model.train();total=0.;counts=Counter();seen_options=set();began=perf_counter()
        progress=EpochProgress(document_upper_bound,args.progress_updates_per_epoch)
        for batch_index,batch in enumerate(prefetch_batches(document_batches(stream.documents(batch_size=args.batch_size,seed=args.seed+epoch),args.batch_size)),1):
            clock.current_step=batch_index;optimizer.zero_grad(set_to_none=True)
            losses=training_step(model,batch,tasks)
            loss=sum(t.loss_weight*losses[t.name] for t in tasks)
            if not torch.isfinite(loss):raise ValueError('Nonfinite streaming training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())
            if first_update is None:
                first_update=perf_counter()-started
                (args.output_dir/'startup_timing.json').write_text(json.dumps(dict(first_optimizer_update_seconds=first_update,corpus_built=False),indent=2))
            for item in batch:
                counts[item['asset_class']]+=1
                if item['asset_class']=='option':
                    seen_options.add(item['underlying_symbol'])
                    key=item['underlying_symbol'];record=stream.observed.setdefault(key,dict(documents=0,price_observations=0,temporal_documents=0))
                    record['documents']+=1;record['price_observations']+=item['prices'].height
                    record['temporal_documents']+=int(item['prices'].height>1)
            status=dict(stage='training',epoch=epoch,batch=batch_index,loss=total/batch_index,documents=dict(counts),
                option_underlyings_seen=len(seen_options),first_optimizer_update_seconds=first_update,elapsed_seconds=perf_counter()-began)
            (args.output_dir/'status.json').write_text(json.dumps(status,indent=2))
            status.update(document_upper_bound=document_upper_bound,epoch_training_complete=False)
            if progress.due(sum(counts.values())):
                print('[warehouse-training] '+json.dumps(status),flush=True)
            if args.checkpoint_every_batches and batch_index%args.checkpoint_every_batches==0:
                checkpoint(epoch,batch_index,False,total/batch_index);stream.write_coverage()
        eligible=set()
        for symbol, coverage in stream.coverage.items():
            for record in coverage['cohorts']:
                if record['year'] < stream.cutoff.year and record.get('selected_contracts',0)>0:
                    eligible.add(symbol)
        missing=eligible-seen_options
        if missing:raise ValueError(f'Warehouse option underlyings omitted from training: {sorted(missing)}')
        snapshots_only={s for s in seen_options if not stream.observed[s]['temporal_documents']}
        if snapshots_only:raise ValueError(f'Options lack historical price-series documents: {sorted(snapshots_only)}')
        if not counts['equity'] or not counts['option']:raise ValueError('Both equity and option price documents must train')
        checkpoint(epoch,batch_index,True,total/batch_index);stream.write_coverage()
        if progress.due(sum(counts.values()),complete=True):
            status.update(epoch_training_complete=True,elapsed_seconds=perf_counter()-began)
            print('[warehouse-training] '+json.dumps(status),flush=True)
        if device.type=='cuda':torch.cuda.empty_cache()
        reports=evaluate_epoch(model,stream,args,epoch)
        (args.output_dir/'status.json').write_text(json.dumps(dict(stage='epoch_complete',epoch=epoch,loss=total/batch_index,documents=dict(counts),backtest_reports=len(reports)),indent=2))
    status=json.loads((args.output_dir/'status.json').read_text())
    status.update(stage='complete',epochs_completed=args.epochs,first_optimizer_update_seconds=first_update,total_seconds=perf_counter()-started)
    (args.output_dir/'status.json').write_text(json.dumps(status,indent=2))
    print('status: complete\noutput: '+json.dumps(str(args.output_dir)),flush=True)
