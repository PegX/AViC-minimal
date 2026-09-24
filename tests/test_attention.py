import torch
from avic_gatn.models.gatn_min import MultiHeadSelfAttention
from avic_gatn.core.circuits import CircuitID

def test_attention_ablation_and_restore():
    torch.manual_seed(7)
    model = MultiHeadSelfAttention(4, 2).eval()
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        baseline = model(x).clone()
        assert torch.allclose(model.last_attn.sum(-1), torch.ones(1, 2, 3))
        model.set_ablation(0)
        model(x)
        assert torch.count_nonzero(model.last_attn[:, 0]) == 0
        model.set_ablation(None)
        assert torch.allclose(model(x), baseline)
    assert CircuitID(0, 'attn1', 0).to_dict()['key'] == 'layer0.attn1.head0'
