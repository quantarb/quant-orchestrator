from collections import Counter
import torch
from quant_orchestrator.research_tools.multirate_batch import BatchTensors, supervision_counts


def test_staged_raw_targets_survive_context_padding_and_corruption():
    raw = torch.tensor([[float('nan'), float('nan')], [1., 2.]])
    padding = torch.tensor([True, False])
    fields = BatchTensors([{'daily': raw, 'padding': padding}], 'cpu')
    targets = fields('daily')
    context = fields('daily').clone()
    context_padding = fields('padding').bool().clone()
    context[0,0] = 0.
    context[0,1,0] = float('nan')
    context_padding[0,0] = False
    assert fields('daily') is targets
    torch.testing.assert_close(targets[0], raw, equal_nan=True)
    assert fields('padding')[0,0]
    assert fields('daily')[0,1,0] == 1.


def test_staging_caches_per_batch_not_across_batches():
    first=BatchTensors([{'x':[1.,2.]}], 'cpu')
    second=BatchTensors([{'x':[3.,4.]}], 'cpu')
    assert first('x') is first('x')
    assert first('x').tolist()==[[1.,2.]]
    assert second('x').tolist()==[[3.,4.]]


def test_batched_diagnostics_match_individual_counts_for_all_asset_classes():
    batch=[{'asset_class':asset} for asset in ('equity','option','equity','note_bond','preferred')]
    valid=torch.tensor([[True,False,True],[False,False,False],[True,True,True],[False,True,False],[True,False,False]])
    expected=Counter()
    for item,row in zip(batch,valid):
        expected[item['asset_class']] += int(row.sum())
    total,by_asset=supervision_counts(valid,batch)
    assert total==int(valid.sum()) and by_asset==expected
    assert 'option' in by_asset and by_asset['option']==0
