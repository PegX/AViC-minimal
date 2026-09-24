from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class CircuitID:
    layer: int
    attn: str
    head: int
    family: str = "gatn"
    unit: str = "attention_head"
    cache_key: str = ""
    display_name: str = ""
    ablation_mode: str = "zero"
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def key(self) -> str:
        if self.display_name:
            return self.display_name
        if self.cache_key:
            return f"{self.cache_key}#{self.head}"
        return f"layer{self.layer}.{self.attn}.head{self.head}"

    @property
    def index(self) -> int:
        return int(self.head)

    def tensor_key(self, default_suffix: str = "attn") -> str:
        if self.cache_key:
            return self.cache_key
        return f"layer{self.layer}.{self.attn}.{default_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "key": self.key(),
            "family": self.family,
            "unit": self.unit,
            "layer": int(self.layer),
            "group": self.attn,
            "index": int(self.index),
            "ablation_mode": self.ablation_mode,
        }
        if self.cache_key:
            data["cache_key"] = self.cache_key
        if self.display_name:
            data["display_name"] = self.display_name
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


def circuit_to_dict(circuit: Any) -> Dict[str, Any]:
    if hasattr(circuit, "to_dict"):
        return circuit.to_dict()
    out: Dict[str, Any] = {"key": str(circuit)}
    for attr in ("family", "unit", "layer", "attn", "head", "cache_key", "display_name", "ablation_mode"):
        if hasattr(circuit, attr):
            out[attr] = getattr(circuit, attr)
    if "attn" in out and "group" not in out:
        out["group"] = out.pop("attn")
    if "head" in out and "index" not in out:
        out["index"] = out.pop("head")
    return out
