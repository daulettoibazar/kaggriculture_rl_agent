#!/usr/bin/env python3
"""Demonstration set from recorded leaderboard episodes (rl_last replay-BC init).

    python3 tools/fs1/make_replay_demos.py --catalog data/replays_2026-09-26/catalog.json \
        --out data/fs1_demos/replays_0926 --max-rank 60 --workers 4 [--envs-per-worker 16]

Each episode is re-simulated through `Fs1VecEnv` from its `info.seed` with BOTH seats' recorded
turns (`PyVecEnv.step(actions, exact_actions)`); the re-simulation is exact (final money to the
dollar, research/rl_last/replay_parity_test.py) and an episode whose final money does not match
its recorded rewards is dropped and counted. Every step of a kept seat (lb_rank <= --max-rank) is
converted with `imitation_labels.LabelBuffer` and stored as shards (obs f16, bit-packed masks, label rows), so
`tools/fs1/bc_pretrain.py` reads it unchanged. The team's style id rides in meta:

    meta = [style_self, style_opp, seed, t, seat, lb_rank_self, episode_id]

Style ids: lb_rank r <= --max-rank gets id r; any other team gets --max-rank + 1; 0 is reserved
for "no style" (programs, deployment default). The id table is written to <out>/styles.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STEPS = 720


def norm(action):
    action = action or {}
    return {"farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in (action.get("hands") or [])],
            "market": [list(m) for m in (action.get("market") or [])]}


def style_of(lb, max_rank):
    if not lb:
        return max_rank + 1, 0
    r = int(lb["lb_rank"])
    return (r if r <= max_rank else max_rank + 1), r


def load_episode(e, max_rank):
    d = json.loads(pathlib.Path(e["path"]).read_text())
    steps = d["steps"]
    if len(steps) < STEPS:
        return None
    styles = [style_of(lb, max_rank) for lb in e["leaderboard"]]
    return {"seed": int(d["info"]["seed"]), "rewards": [float(x) for x in d["rewards"]],
            "rows": [[norm(steps[t + 1][s].get("action")) for t in range(STEPS - 1)] for s in range(2)],
            "styles": [s for s, _ in styles], "ranks": [r for _, r in styles],
            "keep": [0 < r <= max_rank for _, r in styles], "ep": int(e["episode_id"])}


def worker(w: int, args: dict):
    import numpy as np
    import torch

    import src.evo  # noqa: F401  (pins rayon before kaggri)
    from src.fs1 import constants as C
    from src.fs1.env import Fs1VecEnv
    from src.fs1.imitation_labels import LabelBuffer

    out = pathlib.Path(args["out"])
    eps = args["episodes"][w::args["workers"]]
    stats = {"episodes": 0, "dropped_short": 0, "dropped_mismatch": 0, "seat_games": 0, "rows": 0}
    shard, shard_k = [], 0

    def flush(force=False):
        nonlocal shard, shard_k
        if not shard or (not force and len(shard) < args["shard_rows"]):
            return
        np.savez(out / f"shard_{w:02d}_{shard_k:04d}.npz",
                 obs=np.stack([r[0] for r in shard]).astype(np.float16),
                 um=np.packbits(np.stack([r[1] for r in shard]).reshape(len(shard), -1), axis=1),
                 tm=np.packbits(np.stack([r[2] for r in shard]).reshape(len(shard), -1), axis=1),
                 row=np.stack([r[3] for r in shard]).astype(np.int16),
                 ulab=np.stack([r[4] for r in shard]),
                 meta=np.stack([r[5] for r in shard]).astype(np.int64))
        shard_k += 1
        stats["rows"] += len(shard)
        shard = []

    t0 = time.time()
    B = args["envs_per_worker"]
    zeros = None
    for b0 in range(0, len(eps), B):
        games = [g for g in (load_episode(e, args["max_rank"]) for e in eps[b0:b0 + B]) if g is not None]
        stats["dropped_short"] += len(eps[b0:b0 + B]) - len(games)
        games = [g for g in games if any(g["keep"])]
        if not games:
            continue
        n = len(games)
        env = Fs1VecEnv(n, [g["seed"] for g in games], "cpu", hist_k=args["hist_k"], stagger=False)
        zeros = torch.zeros(n, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
        bufs = [[LabelBuffer(), LabelBuffer()] for _ in range(n)]
        staged = [[[], []] for _ in range(n)]
        final = np.full((n, 2), np.nan)
        for t in range(STEPS):
            obs, um, tm = env.flat()
            obs_np, um_np, tm_np = obs.numpy(), um.numpy(), tm.numpy()
            rows = []
            for i, g in enumerate(games):
                r = [g["rows"][s][t] if t < STEPS - 1 else None for s in range(2)]
                if np.isnan(final[i, 0]):
                    st = env.env.state_dict(i)
                    for s in range(2):
                        if g["keep"][s] and r[s] is not None:
                            bufs[i][s].observe(t, st, s, r[s])
                            staged[i][s].append((t, obs_np[2 * i + s].copy(), um_np[2 * i + s].copy(),
                                                 tm_np[2 * i + s].copy()))
                rows.append(r)
            last = env.money.numpy().copy()
            done, fm, _ = env.step(zeros, rows)
            for i in np.flatnonzero(done.numpy()):
                if np.isnan(final[i, 0]):
                    final[i] = fm[i].numpy() if fm[i].any() else last[i]
            if not np.isnan(final[:, 0]).any():
                break
        for i, g in enumerate(games):
            stats["episodes"] += 1
            if not all(abs(final[i, s] - g["rewards"][s]) < 1.0 for s in range(2)):
                stats["dropped_mismatch"] += 1
                continue
            for s in range(2):
                if not g["keep"][s]:
                    continue
                labels = {t: (row, ul) for (t, row, ul, ml) in bufs[i][s].drain()}
                for (t, o, u, tmask) in staged[i][s]:
                    if t in labels:
                        row, ul = labels[t]
                        shard.append((o, u, tmask, row, ul, np.array(
                            [g["styles"][s], g["styles"][1 - s], g["seed"], t, s, g["ranks"][s], g["ep"]])))
                stats["seat_games"] += 1
        flush()
    flush(force=True)
    stats["seconds"] = time.time() - t0
    (out / f"worker_{w:02d}.json").write_text(json.dumps(stats))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="data/replays_2026-09-26/catalog.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rank", type=int, default=60, help="seats of teams ranked <= this are kept")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--hist-k", type=int, default=4)
    ap.add_argument("--shard-rows", type=int, default=16384)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    eps = json.loads(pathlib.Path(a.catalog).read_text())["episodes"]
    if a.max_episodes:
        eps = eps[:a.max_episodes]
    styles = {}
    for e in eps:
        for lb in e["leaderboard"]:
            if lb and lb["lb_rank"] <= a.max_rank:
                styles[int(lb["lb_rank"])] = lb["members"]
    (out / "styles.json").write_text(json.dumps(
        {"max_rank": a.max_rank, "n_styles": a.max_rank + 2, "other": a.max_rank + 1, "none": 0,
         "by_id": {str(k): v for k, v in sorted(styles.items())}}, indent=1, ensure_ascii=False))
    args = {"out": str(out), "episodes": eps, "workers": a.workers, "envs_per_worker": a.envs_per_worker,
            "max_rank": a.max_rank, "hist_k": a.hist_k, "shard_rows": a.shard_rows}
    (out / "args.json").write_text(json.dumps({k: v for k, v in args.items() if k != "episodes"} |
                                              {"catalog": a.catalog, "n_episodes": len(eps)}, indent=1))
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        stats = pool.starmap(worker, [(w, args) for w in range(a.workers)])
    tot = {k: sum(s[k] for s in stats) for k in stats[0] if k != "seconds"}
    tot["seconds"] = max(s["seconds"] for s in stats)
    print(json.dumps(tot))
    (out / "summary.json").write_text(json.dumps({"total": tot, "workers": stats}, indent=1))


if __name__ == "__main__":
    main()
