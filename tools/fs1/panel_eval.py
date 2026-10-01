#!/usr/bin/env python3
"""fs1 panel: play a checkpoint against the public field in the exact simulator and report
win rate, LCB95, margin and median money per opponent, lockday-style, with an incumbent row.

    python3 tools/fs1/panel_eval.py --checkpoint runs/learning/fs1_X/latest.pt \
        [--opponents mengine_v4,goodpjw,...] [--seeds 32] [--incumbent dist/tape_v21_cha22x/main.py] \
        [--out research/fs1/panel/uNNNN]

Design: every cell (one opponent) runs its 2 x seeds games in lockstep inside this process. The
candidate acts through `kaggri.Encoder` on the Kaggle-format observation (exactly the bundle's
path) with the torch policy batched on the GPU (greedy); the opponents are `KernelPolicy`
instances (guards, per-turn timeout, counted fallbacks), one per game, reset per game. The
optional incumbent row is any bundle `main.py`, also through `KernelPolicy`. Names resolve
through `src.rl3.opponents` (the opponent registry). The public
programs never see anything but their own observation. Runs from a scratch CWD because public
agents write files into the CWD.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import src.evo  # noqa: E402,F401  (pins rayon before kaggri)
import kaggri  # noqa: E402
from src.evo import kernel_opponents as K  # noqa: E402
from src.fs1 import constants as C  # noqa: E402
from src.fs1.export import load_checkpoint_model  # noqa: E402
from src.fs1.masks import relax_unit_mask  # noqa: E402
from src.rl3 import opponents as O  # noqa: E402

DEFAULT_OPPONENTS = ("mengine_v4", "yummers_v52", "abo_v50", "abo_v49", "goodpjw", "farm2945")
STEPS = C.EPISODE_STEPS


# ------------------------------------------------------------------ candidates
class TorchCandidate:
    """The fs1 checkpoint, acting greedily on a batch of games through the Encoder path."""

    name = "fs1"

    def __init__(self, checkpoint: pathlib.Path, device, temperature: float = 0.0, relax: bool = False):
        self.model = load_checkpoint_model(checkpoint).to(device).eval()
        self.device = torch.device(device)
        self.k = self.model.cfg.hist_k
        self.temperature = float(temperature)      # 0 = greedy (argmax); > 0 = sampled at that temperature
        self.relax = bool(relax)                   # rl_last F1 mask relaxation (src/fs1/masks.py)
        self.enc = None
        self.fallbacks = 0

    def start_games(self, n: int, cfg: dict):
        self.enc = kaggri.Encoder(cfg, obs_dtype="float32")
        self.hist = torch.zeros(n, len(C.PRODUCTS), self.k, device=self.device)
        self.prev_rel = None

    @torch.no_grad()
    def act_batch(self, kobs_list, cfg):
        n = len(kobs_list)
        obs = np.empty((n, C.OBS_DIM), np.float32)
        um = np.empty((n, C.MASK_DIM), np.float32)
        tm = np.empty((n, C.TILE_MASK_DIM), np.uint8)
        for i, kobs in enumerate(kobs_list):
            o, m, t = self.enc.encode_full(kobs)
            obs[i], um[i], tm[i] = np.asarray(o), np.asarray(m), np.asarray(t)
        o = torch.from_numpy(obs).to(self.device).view(n, C.N_TOKENS, C.N_FEATURES)
        rel = o[:, C.TOK_PRODUCT0:C.TOK_PRODUCT1, C.HDR + C.P_PRICE_REL]
        delta = torch.zeros_like(rel) if self.prev_rel is None else rel - self.prev_rel
        if self.k:
            self.hist = torch.roll(self.hist, 1, -1)
            self.hist[..., 0] = delta
        self.prev_rel = rel.clone()
        aug = torch.zeros(n, C.N_TOKENS, self.k, device=self.device)
        if self.k:
            aug[:, C.TOK_PRODUCT0:C.TOK_PRODUCT1] = self.hist
        obs_aug = torch.cat([o, aug], -1).to(torch.float16)
        umask = torch.from_numpy(um).to(self.device).view(n, C.MAX_UNITS, C.N_VERBS) > 0.5
        tmask = torch.from_numpy(tm).to(self.device).view(n, C.N_TARGETS, C.N_VERBS) > 0
        if self.relax:
            umask = relax_unit_mask(umask, tmask)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            rows, *_ = self.model.act(obs_aug, umask, tmask, None, greedy=self.temperature <= 0,
                                      temperature=max(self.temperature, 1e-6))
        rows = rows.cpu().numpy()
        out = []
        for i, kobs in enumerate(kobs_list):
            try:
                out.append(self.enc.action(kobs, rows[i]))
            except Exception:
                self.fallbacks += 1
                out.append(dict(K.PASS_TURN))
        return out


class KernelRow:
    """A bundle/notebook `main.py` as the candidate (the incumbent row), one instance per game."""

    def __init__(self, path: pathlib.Path, name: str, struct: bool = True):
        self.path, self.name, self.struct = pathlib.Path(path), name, struct
        self.policies = []

    def start_games(self, n: int, cfg: dict):
        self.policies = [K.KernelPolicy(self.path, timeout=10.0).reset() for _ in range(n)]

    def act_batch(self, kobs_list, cfg):
        return [p.act(kobs) for p, kobs in zip(self.policies, kobs_list)]

    @property
    def fallbacks(self):
        return sum(p.stats.fallbacks for p in self.policies)


# ------------------------------------------------------------------- one cell
def play_cell(candidate, opp_name: str, seeds, log=print):
    """2 x len(seeds) games (both seats) in lockstep. Returns per-game dicts."""
    games, seats, opps, opp_struct = [], [], [], True
    if opp_name != "pass":
        opp_path = O.path_of(opp_name)
        opp_struct = O.struct_of(opp_name)
    jobs = [(s, seat) for s in seeds for seat in (0, 1)]
    for s, seat in jobs:
        g = kaggri.PyGame({"episodeSteps": STEPS, "seed": int(s)})
        games.append(g)
        seats.append(seat)
        opps.append(None if opp_name == "pass" else K.KernelPolicy(opp_path, timeout=10.0).reset())
    cfg = games[0].config()
    candidate.start_games(len(games), cfg)
    cand_struct = getattr(candidate, "struct", False)
    t0 = time.time()
    for t in range(STEPS):
        states = [g.state_dict() for g in games]
        cand_obs = []
        for st, seat in zip(states, seats):
            ko = K.observation_of(st, seat, struct=cand_struct)
            if not cand_struct:
                ko["step"] = t
            cand_obs.append(ko)
        cand_actions = candidate.act_batch(cand_obs, cfg)
        for i, g in enumerate(games):
            if opps[i] is None:
                theirs = dict(K.PASS_TURN)
            else:
                theirs = opps[i].act(K.observation_of(states[i], 1 - seats[i], struct=opp_struct))
            a = [None, None]
            a[seats[i]], a[1 - seats[i]] = cand_actions[i], theirs
            g.step(a[0], a[1])
        if all(g.done for g in games):
            break
    rows = []
    for g, seat, s, opp, (seed, _) in zip(games, seats, seeds * 2, opps, jobs):
        money = g.money
        ours, theirs = float(money[seat]), float(money[1 - seat])
        rows.append({"seed": int(seed), "seat": seat, "our_money": ours, "their_money": theirs,
                     "margin": ours - theirs, "win": ours > theirs, "draw": ours == theirs,
                     "opp_fallbacks": 0 if opp is None else opp.stats.fallbacks})
    log(f"    {opp_name:<12} {len(games)} games in {time.time() - t0:5.0f}s  cand_fallbacks={candidate.fallbacks}")
    return rows


def summarise(rows):
    w = np.array([1.0 if r["win"] else (0.5 if r["draw"] else 0.0) for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    m = np.array([r["margin"] for r in rows])
    ours = np.array([r["our_money"] for r in rows])
    try:
        se = float(K.cluster_se(w, seeds))
    except Exception:
        se = float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0.0
    return {"n": len(rows), "score": float(w.mean()), "lcb95": max(0.0, float(w.mean()) - 1.645 * se),
            "margin_mean": float(m.mean()), "margin_median": float(np.median(m)),
            "money_median": float(np.median(ours)), "money_mean": float(ours.mean()),
            "draws": int(sum(r["draw"] for r in rows)),
            "opp_fallbacks": int(sum(r["opp_fallbacks"] for r in rows))}


def render(table: dict, floor: float) -> str:
    rows_names = list(table.keys())
    opps = list(next(iter(table.values())).keys())
    out = []
    head = f"  {'opponent':<12}" + "".join(f"{n:>34}" for n in rows_names)
    out.append(head)
    for o in opps:
        line = f"  {o:<12}"
        for rn in rows_names:
            c = table[rn][o]
            line += f"  {c['score']:.3f} (L{c['lcb95']:.3f}) {c['margin_mean']:>+8,.0f} med${c['money_median']:>7,.0f}"
        out.append(line)
    for agg, fn in (("WORST", min), ("MEAN", lambda xs: sum(xs) / len(xs))):
        line = f"  {agg:<12}"
        for rn in rows_names:
            v = fn([table[rn][o]["score"] for o in opps])
            line += f"  {v:.3f}" + " " * 28
        out.append(line)
    for rn in rows_names:
        bad = [o for o in opps if table[rn][o]["score"] < floor]
        out.append(f"  floor {floor:.2f}: {rn} {'CLEAR' if not bad else 'FAILS on ' + ', '.join(bad)}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS), help="registry names, or 'pass'")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=77_000_000)
    ap.add_argument("--incumbent", default=None, help="a bundle main.py to play as an extra row")
    ap.add_argument("--incumbent-name", default="incumbent")
    ap.add_argument("--floor", type=float, default=0.40)
    ap.add_argument("--out", default=None, help="write <out>.json and <out>.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; e.g. 1.0 = sample as in training")
    ap.add_argument("--relax-masks", action="store_true", help="F1 mask relaxation, for policies trained with it")
    a = ap.parse_args(argv)

    # resolve every path before leaving the repo CWD (public agents write into the CWD)
    ck = pathlib.Path(a.checkpoint).resolve()
    inc = pathlib.Path(a.incumbent).resolve() if a.incumbent else None
    out = pathlib.Path(a.out).resolve() if a.out else None
    scratch = tempfile.mkdtemp(prefix="fs1_panel_")
    os.chdir(scratch)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    opps = [o.strip() for o in a.opponents.split(",") if o.strip()]
    cand = TorchCandidate(ck, a.device, a.temperature, relax=a.relax_masks)
    rows = [(f"fs1_t{a.temperature:g}", cand)]
    if inc is not None:
        rows.append((a.incumbent_name, KernelRow(inc, a.incumbent_name)))

    table, raw = {}, {}
    for rn, policy in rows:
        print(f"row {rn}:", flush=True)
        table[rn], raw[rn] = {}, {}
        for o in opps:
            games = play_cell(policy, o, seeds)
            table[rn][o] = summarise(games)
            raw[rn][o] = games
    text = render(table, a.floor)
    print(text, flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"checkpoint": str(ck), "seeds": a.seeds, "seed0": a.seed0, "opponents": opps,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out.with_suffix(".json")).write_text(json.dumps({"meta": meta, "table": table, "games": raw}, indent=1))
        (out.with_suffix(".md")).write_text(f"# fs1 panel {meta['ts']}\n\ncheckpoint {ck}\n\n```\n{text}\n```\n")
    return table


if __name__ == "__main__":
    main()
