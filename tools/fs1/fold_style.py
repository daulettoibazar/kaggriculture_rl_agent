#!/usr/bin/env python3
"""Bake one team style of a replay-BC checkpoint into a plain fs1 checkpoint (rl_last clone -> self-play).

    python3 tools/fs1/fold_style.py --checkpoint runs/learning/rl_last_bc_r1c/bc_epoch3.pt --style 3 \
        --out runs/learning/rl_last_bc_r1c/clone_s3.pt

`Fs1Policy.fold_style` adds the style vector to pos[TOK_GLOBAL] and drops the embedding, so the
result has n_styles = 0 and the trainer's default model config: `tools/fs1/train.py --init-from`
(with --init-no-optimizer --init-no-league --init-no-ret-scale) and `--ref-policy` accept it, and
the numpy export needs no style support.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.fs1.model import Fs1Config, Fs1Policy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    m = Fs1Policy(Fs1Config.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    f = m.fold_style(a.style)
    out = {"model": f.state_dict(), "model_cfg": f.cfg.to_dict(), "update": 0, "upd_base": 0, "elapsed": 0.0,
           "ppo_cfg": ck.get("ppo_cfg", {}), "args": ck.get("args", {}), "ret_scale": {"m2": 1.0, "n": 0},
           "league": {"last_best": None, "pool": []},
           "fold": {"source": a.checkpoint, "style": a.style, "bc": ck.get("bc", {}).get("val")}}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"wrote {a.out} (style {a.style} of {a.checkpoint})")


if __name__ == "__main__":
    main()
