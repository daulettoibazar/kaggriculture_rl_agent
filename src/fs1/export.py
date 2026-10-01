"""Export an fs1 checkpoint to the flat float16 array dict the numpy policy loads.

The numpy policy (`submission/fs1_numpy_policy.py`) has no torch dependency, so the export is
a plain rename of the state dict into short keys plus the model config as integers.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import Fs1Config, Fs1Policy


def policy_arrays(model: Fs1Policy, dtype=np.float16) -> dict:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    cfg = model.cfg
    out = {
        "cfg_d_model": np.array(cfg.d_model), "cfg_n_layers": np.array(cfg.n_layers),
        "cfg_n_heads": np.array(cfg.n_heads), "cfg_mlp_ratio": np.array(cfg.mlp_ratio),
        "cfg_d_target": np.array(cfg.d_target), "cfg_hist_k": np.array(cfg.hist_k),
    }

    def put(name, t):
        out[name] = t.numpy().astype(dtype)

    put("stem.w", sd["stem.weight"]); put("stem.b", sd["stem.bias"]); put("pos", sd["pos"])
    for i in range(cfg.n_layers):
        pre = f"blocks.{i}."
        for src, dst in (("ln1.weight", "ln1.g"), ("ln1.bias", "ln1.b"), ("qkv.weight", "qkv.w"),
                         ("qkv.bias", "qkv.b"), ("proj.weight", "proj.w"), ("proj.bias", "proj.b"),
                         ("ln2.weight", "ln2.g"), ("ln2.bias", "ln2.b"), ("fc1.weight", "fc1.w"),
                         ("fc1.bias", "fc1.b"), ("fc2.weight", "fc2.w"), ("fc2.bias", "fc2.b")):
            put(f"b{i}.{dst}", sd[pre + src])
    put("lnf.g", sd["ln_f.weight"]); put("lnf.b", sd["ln_f.bias"])
    put("verb.w1", sd["verb_head.0.weight"]); put("verb.b1", sd["verb_head.0.bias"])
    put("verb.w2", sd["verb_head.2.weight"]); put("verb.b2", sd["verb_head.2.bias"])
    put("verb_emb", sd["verb_emb.weight"])
    put("tq.w", sd["tq.weight"]); put("tq.b", sd["tq.bias"]); put("tk.w", sd["tk.weight"]); put("tk.b", sd["tk.bias"])
    put("market.w1", sd["market_head.0.weight"]); put("market.b1", sd["market_head.0.bias"])
    put("market.w2", sd["market_head.2.weight"]); put("market.b2", sd["market_head.2.bias"])
    if getattr(cfg, "residual", False):            # rl_last Option B: FOLLOW/override gates
        out["cfg_n_extra_in"] = np.array(cfg.n_extra_in)
        out["cfg_sell_only"] = np.array(int(getattr(cfg, "open_gates", "all") == "sell"))
        put("ugate.w", sd["unit_gate.weight"]); put("ugate.b", sd["unit_gate.bias"])
        put("mgate.w", sd["market_gate.weight"]); put("mgate.b", sd["market_gate.bias"])
    return out


def load_checkpoint_model(path, device="cpu") -> Fs1Policy:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Fs1Config.from_dict(ck["model_cfg"])
    m = Fs1Policy(cfg)
    m.load_state_dict(ck["model"])
    return m.eval()
