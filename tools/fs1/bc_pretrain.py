#!/usr/bin/env python3
"""Behaviour-cloning pretraining of the fs1 policy on the offline demonstration set.

    python3 tools/fs1/bc_pretrain.py --demos data/fs1_demos/replays --init <rl_checkpoint>.pt \
        --out runs/learning/fs1_bc --epochs 3 --device cuda:1 [--lr 1e-4 --batch 1024]

Streams the replay-demo shards written by `tools/fs1/make_replay_demos.py` (obs f16, bit-packed masks, label rows),
trains `Fs1Policy.imitation_nll` (per-component NLL over labelled unit heads and all 21 market
heads) with AdamW, holds out `--val-shards` shards for validation, logs to <out>/bc_events.jsonl
and writes <out>/bc_epochN.pt plus <out>/latest.pt as checkpoints the trainer's `--init-from`
accepts (model, model_cfg, ppo_cfg, args, ret_scale/league/upd_base copied from `--init`,
optimizer state omitted -> use --init-no-optimizer on the resume). Initialising from the RL
checkpoint keeps the value heads sensible; `--from-scratch` starts from random weights.

rl_last replay BC (research/rl_last/DESIGN.md): `--demos` takes several sets as
`path[:weight[:style]]` (style `meta` = meta[:, 0] from `make_replay_demos.py`, an int = that
constant style for every row, e.g. 0 for the program demos); `--n-styles` gives the model a
per-team style embedding; `--top-rank`/`--top-weight` up-weight rows of seats ranked <= top-rank
(meta[:, 5]); `--relax-masks` applies the F1 mask fix (`src/fs1/masks.py`). Validation shards
come from the first set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.fs1 import constants as C  # noqa: E402
from src.fs1.masks import relax_unit_mask  # noqa: E402
from src.fs1.model import Fs1Config, Fs1Policy  # noqa: E402


def parse_sets(spec):
    """'a:1:meta,b:0.3:0' -> [(path, weight, style)]; style None = meta column 0."""
    out = []
    for part in spec.split(","):
        f = part.split(":")
        style = None if len(f) < 3 or f[2] == "meta" else int(f[2])
        out.append((f[0], float(f[1]) if len(f) > 1 else 1.0, style))
    return out


def load_shard(path, device, relax=False, style=None, weight=1.0, top_rank=0, top_weight=1.0):
    """-> (obs, um, tm, row, ulab, mlab, style [n] long, w [n] float)."""
    z = np.load(path)
    n = z["obs"].shape[0]
    obs = torch.from_numpy(z["obs"]).to(device)
    um = torch.from_numpy(np.unpackbits(z["um"], axis=1)[:, :C.MAX_UNITS * C.N_VERBS].reshape(n, C.MAX_UNITS, C.N_VERBS).astype(bool)).to(device)
    tm = torch.from_numpy(np.unpackbits(z["tm"], axis=1)[:, :C.N_TARGETS * C.N_VERBS].reshape(n, C.N_TARGETS, C.N_VERBS).astype(bool)).to(device)
    row = torch.from_numpy(z["row"].astype(np.int64)).to(device)
    ulab = torch.from_numpy(z["ulab"]).to(device)
    mlab = torch.ones(n, C.N_MARKET_HEADS, dtype=torch.bool, device=device)
    if relax:
        um = relax_unit_mask(um, tm)
    meta = z["meta"] if "meta" in z.files else np.zeros((n, 1), np.int64)
    sty = torch.from_numpy(meta[:, 0].astype(np.int64) if style is None else np.full(n, style, np.int64)).to(device)
    w = np.full(n, weight, np.float32)
    if top_rank and style is None and meta.shape[1] > 5:
        w = np.where((meta[:, 5] > 0) & (meta[:, 5] <= top_rank), w * top_weight, w).astype(np.float32)
    return obs, um, tm, row, ulab, mlab, sty, torch.from_numpy(w).to(device)


def run_epoch(model, shards, device, batch, opt=None, autocast=True, log=None, epoch=0, tag="train", lk=None):
    tot = torch.zeros(2, dtype=torch.float64)
    cnt = torch.zeros(2, dtype=torch.float64)
    seen = 0
    t0 = time.time()
    order = list(shards)
    if opt is not None:
        np.random.shuffle(order)
    lk = lk or {}
    for si, path in enumerate(order):
        obs, um, tm, row, ulab, mlab, sty, w = load_shard(path, device, **lk.get(path, {}))
        n = obs.shape[0]
        perm = torch.randperm(n, device=device) if opt is not None else torch.arange(n, device=device)
        for j in range(0, n, batch):
            idx = perm[j:j + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
                nll, k = model.imitation_nll(obs[idx], um[idx], tm[idx], row[idx], ulab[idx], mlab[idx],
                                             style=sty[idx])
            nll, k = nll.float(), k.float()
            wi = w[idx][:, None]
            loss = (nll * wi).sum() / (k * wi).sum().clamp_min(1.0)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += nll.sum(0).detach().cpu().double()
            cnt += k.sum(0).detach().cpu().double()
            seen += int(idx.numel())
        if log is not None and opt is not None and (si + 1) % 10 == 0:
            rec = {"epoch": epoch, "shard": si + 1, "rows": seen, "nll_unit": float(tot[0] / cnt[0].clamp(min=1)),
                   "nll_market": float(tot[1] / cnt[1].clamp(min=1)), "elapsed": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
    return {"tag": tag, "epoch": epoch, "rows": seen, "nll_unit": float(tot[0] / cnt[0].clamp(min=1)),
            "nll_market": float(tot[1] / cnt[1].clamp(min=1)),
            "nll": float(tot.sum() / cnt.sum().clamp(min=1)), "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True, help="path[:weight[:style]], comma-separated (see module doc)")
    ap.add_argument("--n-styles", type=int, default=0, help="> 0: add a per-team style embedding (replay BC)")
    ap.add_argument("--relax-masks", action="store_true", help="F1 mask fix on the stored unit masks")
    ap.add_argument("--top-rank", type=int, default=0)
    ap.add_argument("--top-weight", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="fs1 checkpoint to start from (weights + cfg); required unless --from-scratch")
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--val-shards", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(a.device)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sets = parse_sets(a.demos)
    lk, per_set = {}, []
    for path, weight, style in sets:
        sh = sorted(glob.glob(str(pathlib.Path(path) / "shard_*.npz")))
        if a.max_shards:
            sh = sh[:a.max_shards]
        per_set.append(sh)
        for s_ in sh:
            lk[s_] = {"relax": a.relax_masks, "style": style, "weight": weight,
                      "top_rank": a.top_rank, "top_weight": a.top_weight}
    assert len(per_set[0]) > a.val_shards, (len(per_set[0]), a.val_shards)
    rng = np.random.default_rng(a.seed)
    val = sorted(rng.choice(per_set[0], size=a.val_shards, replace=False).tolist())
    train = [s for sh in per_set for s in sh if s not in set(val)]
    src = None
    if a.init and not a.from_scratch:
        src = torch.load(a.init, map_location="cpu", weights_only=False)
        cfg = Fs1Config.from_dict({**src["model_cfg"], "n_styles": a.n_styles or src["model_cfg"].get("n_styles", 0)})
    else:
        cfg = Fs1Config(n_styles=a.n_styles)
    model = Fs1Policy(cfg).to(device)
    if src is not None:
        missing, unexpected = model.load_state_dict(src["model"], strict=False)
        assert not unexpected and all(k.startswith("style_emb.") for k in missing), (missing, unexpected)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    if src is not None:                                   # --init-from expects the league snapshots beside the checkpoint
        import shutil
        lg = src.get("league", {})
        names = ([lg["last_best"]] if lg.get("last_best") else []) + list(dict.fromkeys(lg.get("pool", [])))
        (out / "league").mkdir(exist_ok=True)
        for nm in names:
            snap = pathlib.Path(a.init).resolve().parent / "league" / f"{nm}.pt"
            if snap.exists() and not (out / "league" / f"{nm}.pt").exists():
                shutil.copyfile(snap, out / "league" / f"{nm}.pt")
    log = open(out / "bc_events.jsonl", "a")
    (out / "bc_args.json").write_text(json.dumps({"args": vars(a), "train_shards": len(train), "val_shards": val}, indent=1))
    model.eval()
    with torch.no_grad():
        v0 = run_epoch(model, val, device, a.batch, None, epoch=0, tag="val", lk=lk)
    print(json.dumps(v0), flush=True); log.write(json.dumps(v0) + "\n")
    for ep in range(1, a.epochs + 1):
        model.train()
        tr = run_epoch(model, train, device, a.batch, opt, log=log, epoch=ep, tag="train", lk=lk)
        model.eval()
        with torch.no_grad():
            va = run_epoch(model, val, device, a.batch, None, epoch=ep, tag="val", lk=lk)
        print(json.dumps(tr), flush=True); print(json.dumps(va), flush=True)
        log.write(json.dumps(tr) + "\n"); log.write(json.dumps(va) + "\n"); log.flush()
        ck = {"model": model.state_dict(), "update": int(src["update"]) if src else 0, "elapsed": 0.0,
              "upd_base": int(src.get("upd_base", 0)) + (int(src["update"]) + 1 if src else 0) if src else 0,
              "model_cfg": cfg.to_dict(), "ppo_cfg": src["ppo_cfg"] if src else {}, "args": src["args"] if src else {},
              "ret_scale": src["ret_scale"] if src else {"m2": 1.0, "n": 0},
              "league": src["league"] if src else {"last_best": None, "pool": []},
              "bc": {"epoch": ep, "train": tr, "val": va, "init": a.init, "demos": a.demos}}
        torch.save(ck, out / f"bc_epoch{ep}.pt")
        torch.save(ck, out / "latest.pt")
    log.close()


if __name__ == "__main__":
    main()
