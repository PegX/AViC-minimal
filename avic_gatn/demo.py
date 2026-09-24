"""Measure attention-head ablation on synthetic node features."""
import argparse
import json
from pathlib import Path
import torch
from .models.gatn_min import MultiHeadSelfAttention
from .core.circuits import CircuitID

def run():
    torch.manual_seed(7)
    model=MultiHeadSelfAttention(4,2).eval()
    features=torch.randn(1,3,4)
    with torch.no_grad():
        clean=model(features).clone();attention=model.last_attn.clone();rows=[]
        for head in range(2):
            model.set_ablation(head)
            try:
                changed=model(features)
                rows.append({"circuit":CircuitID(0,"attn1",head).to_dict(),"output_l2_change":float(torch.linalg.vector_norm(changed-clean)),"ablated_attention_mass":float(model.last_attn[:,head].sum())})
            finally:model.set_ablation(None)
        restored=bool(torch.allclose(model(features),clean))
    return {"project":"AViC","scope":"attention component demo; no images or benchmark accuracy", "input_shape":list(features.shape),"attention_shape":list(attention.shape),"heads":rows,"restored":restored}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path);a=p.parse_args();text=json.dumps(run(),indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+"\n")
    else:print(text)
if __name__ == "__main__":main()
