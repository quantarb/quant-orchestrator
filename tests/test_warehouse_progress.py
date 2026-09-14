import pytest
from quant_orchestrator.research_tools.warehouse_multirate_training import EpochProgress


@pytest.mark.parametrize('limit', [1, 3, 10])
@pytest.mark.parametrize('bound,total,batch_size', [(10000,10000,1), (10000,2300,17), (10,10,100), (1,50,1)])
def test_streaming_updates_are_capped_and_always_finish(limit,bound,total,batch_size):
    progress=EpochProgress(bound,limit)
    updates=[n for n in range(batch_size,total+1,batch_size) if progress.due(n)]
    assert progress.due(total,complete=True)
    assert len(updates)+1<=limit
    assert not progress.due(total,complete=True)


def test_ten_milestones_include_completion():
    progress=EpochProgress(100,10)
    assert [n for n in range(1,101) if progress.due(n)]==list(range(10,100,10))
    assert progress.due(100,complete=True)


@pytest.mark.parametrize('limit',[0,11,-1])
def test_invalid_update_limit_rejected(limit):
    with pytest.raises(ValueError,match='between 1 and 10'):
        EpochProgress(100,limit)
