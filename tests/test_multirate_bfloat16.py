import pytest
import torch
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTransformer, MultiRateTransformerConfig, MultiRateTaskSpec, MultiRatePredictionTaskSpec,
)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason='BF16 CUDA hardware required')
def test_bfloat16_forward_backward_keeps_all_rate_and_time_gradients_finite():
    torch.manual_seed(4)
    rates=('annual','quarterly','daily','sparse')
    model=MultiRateTransformer({rate:3 for rate in rates},
        config=MultiRateTransformerConfig(d_model=8,num_heads=2,layers=1,dropout=0.),
        feature_families={rate:{'a':2,'b':1} for rate in rates},
        tasks=[MultiRateTaskSpec('return','token',source='daily')],
        prediction_tasks=[MultiRatePredictionTaskSpec('next','next_token',level='subtoken',source='daily',output_dim=2)],
    ).cuda().train()
    data={f'{rate}_values':torch.randn(2,4,3,device='cuda') for rate in rates}
    data.update({f'{rate}_dates':torch.tensor([[1,3,8,15],[1,4,9,15]],device='cuda')*86400000000000 for rate in rates})
    data['annual_values'][:,0]=float('nan')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(**data)
        loss=output['token_outputs']['return'].float().square().mean()+output['prediction_outputs']['next'].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for prefix in ('annual_encoder.','quarterly_encoder.','encoders.daily.','encoders.sparse.','information_age.','auto_feature_engineer.elapsed_time.'):
        grads=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),prefix
        assert sum(float(g.abs().sum()) for g in grads)>0,prefix
