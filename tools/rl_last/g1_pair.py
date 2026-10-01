"""rl_last G1 mini panel: a cached cha22x control, and the paired join of a candidate against it.

    # once: cha22x vs the G1 columns on block 130M (16 workers, CPU only)
    python3 tools/rl_last/g1_pair.py control --out <control.json>

    # every 2 h: pair a candidate panel (tools/fs1/panel_eval.py JSON output) against the control, same seeds, seats and opponents
    python3 tools/rl_last/g1_pair.py pair --control <control.json> \
        --candidate <panel.json> [--row fs1_t0] [--b0 <previous pair json>] [--expect-zero] [--out <pair>]

D = candidate margin - cha22x margin in the identical world (seed, seat, opponent). Cell means
and the pooled mean (equal weight per cell) carry an LCB95 clustered on the seed
(`kernel_opponents.cluster_se`); every pooled game of one seed shares a cluster. --expect-zero
is the G0 parity check for Option B at FOLLOW: it fails unless every D is exactly $0.
Run from a scratch CWD (public agents write into the CWD).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
from src.evo import kernel_opponents as K  # noqa: E402

G1_COLS = ("cha22x_bundle", "abo_v57", "yummers_v52", "goodpjw", "haideptry_2965", "dmitrii_herd", "pass")
CHA22X = R + "/dist/tape_v21_cha22x/main.py"
Z = 1.96
_POL = {}


def _policy(path):
    if path not in _POL:
        _POL[path] = K.KernelPolicy(path, timeout=10.0)
    return _POL[path].reset()


def _play(job):
    """One game: cha22x in `seat` vs `opp`, the same loop and observation types as panel_eval."""
    import kaggri
    from src.rl3 import opponents as O
    seed, seat, opp = job
    ours = _policy(CHA22X)
    theirs = None if opp == "pass" else _policy(str(O.path_of(opp)))
    st_their = True if opp == "pass" else O.struct_of(opp)
    game = kaggri.PyGame(dict(K.DEFAULT_CFG, seed=int(seed)))
    them = 1 - seat
    with K._quiet():
        for _ in range(K.STEPS_PER_GAME):
            st = game.state_dict()
            a = ours.act(K.observation_of(st, seat, struct=True))
            b = {} if theirs is None else theirs.act(K.observation_of(st, them, struct=st_their))
            acts = [None, None]; acts[seat], acts[them] = a, b
            game.step(acts[0], acts[1])
            if game.done:
                break
    m = game.money
    return {"opp": opp, "seed": int(seed), "seat": seat, "margin": float(m[seat] - m[them]),
            "our_money": float(m[seat]), "their_money": float(m[them]),
            "fallbacks": ours.stats.fallbacks + (theirs.stats.fallbacks if theirs else 0)}


def control(a):
    cols = a.cols.split(",")
    jobs = [(a.seed0 + i, seat, c) for c in cols for i in range(a.seeds) for seat in (0, 1)]
    t0 = time.time()
    with Pool(a.workers) as p:
        rows = p.map(_play, jobs, chunksize=1)
    games = {c: [r for r in rows if r["opp"] == c] for c in cols}
    meta = {"row": "cha22x", "path": CHA22X, "seed0": a.seed0, "seeds": a.seeds, "cols": cols,
            "secs": round(time.time() - t0, 1), "fallbacks": sum(r["fallbacks"] for r in rows)}
    json.dump({"meta": meta, "games": {"cha22x": games}}, open(a.out, "w"), indent=1)
    print(f"[g1] control {len(rows)} games in {meta['secs']} s, fallbacks {meta['fallbacks']} -> {a.out}")


def _load_games(path, row):
    """{opp: {(seed, seat): (margin, our_money)}} from a panel_eval JSON (or a control JSON)."""
    d = json.load(open(path))
    out = {}
    G = d["games"]
    row = row if row in G else next(iter(G))
    for c, gs in G[row].items():
        out[c] = {(g["seed"], g["seat"]): (g["margin"], g["our_money"]) for g in gs}
    return out


def pair(a):
    ctl = _load_games(a.control, "cha22x")
    cand = _load_games(a.candidate, a.row)
    cells, all_d, all_s = {}, [], []
    for c in ctl:
        if c not in cand:
            continue
        keys = sorted(k for k in cand[c] if k in ctl[c])
        if not keys:
            continue
        d = np.array([cand[c][k][0] - ctl[c][k][0] for k in keys])
        s = np.array([k[0] for k in keys])
        m = np.array([cand[c][k][0] for k in keys])
        se = K.cluster_se(d, s) if len(set(s)) > 1 else 0.0
        cells[c] = {"n": len(keys), "d_mean": float(d.mean()), "d_lcb": float(d.mean() - Z * se),
                    "exact": float(((m > 0) + 0.5 * (m == 0)).mean()),
                    "exact_ctl": float(np.mean([(ctl[c][k][0] > 0) + 0.5 * (ctl[c][k][0] == 0) for k in keys])),
                    "money_med": float(np.median([cand[c][k][1] for k in keys])),
                    "money_med_ctl": float(np.median([ctl[c][k][1] for k in keys])),
                    "nonzero": int((d != 0).sum())}
        if c != "pass":  # PASS is the productivity check (S5), not part of the pooled margin
            all_d.append(d); all_s.append(s)
    if not cells:
        sys.exit("[g1] no overlapping (opponent, seed, seat) games between candidate and control")
    # equal weight per cell: per seed, the mean over cells of that seed's mean D in the cell
    per_seed = {}
    for d, s in zip(all_d, all_s):
        for seed in set(s.tolist()):
            per_seed.setdefault(seed, []).append(float(d[s == seed].mean()))
    ps = np.array([np.mean(v) for v in per_seed.values()])
    pooled = {"d_mean": float(ps.mean()), "d_lcb": float(ps.mean() - Z * ps.std(ddof=1) / np.sqrt(len(ps))),
              "seeds": len(ps)}
    margin_cells = {c: v for c, v in cells.items() if c != "pass"}
    worst = min(margin_cells, key=lambda c: margin_cells[c]["d_lcb"]) if margin_cells else None
    res = {"candidate": a.candidate, "row": a.row, "control": a.control, "cells": cells, "pooled": pooled,
           "worst_cell": worst, "worst_d_lcb": margin_cells[worst]["d_lcb"] if worst else None}
    lines = [f"G1 paired vs cha22x  ({a.candidate})",
             f"  {'opponent':16s} {'n':>4s} {'D mean':>9s} {'D LCB95':>9s} {'exact':>6s} {'ctl':>6s} "
             f"{'med$':>9s} {'ctl med$':>9s} {'D!=0':>5s}"]
    for c, v in cells.items():
        lines.append(f"  {c:16s} {v['n']:4d} {v['d_mean']:+9,.0f} {v['d_lcb']:+9,.0f} {v['exact']:6.3f} "
                     f"{v['exact_ctl']:6.3f} {v['money_med']:9,.0f} {v['money_med_ctl']:9,.0f} {v['nonzero']:5d}")
    lines.append(f"  POOLED (equal cell weight, {pooled['seeds']} seed clusters, PASS excluded): "
                 f"D {pooled['d_mean']:+,.0f}  LCB95 {pooled['d_lcb']:+,.0f};  worst cell {worst} "
                 f"LCB95 {res['worst_d_lcb']:+,.0f}")
    ok = True
    if a.b0:
        b0 = json.load(open(a.b0))["pooled"]["d_mean"]
        res["b0"] = b0
        lines.append(f"  vs B0 {b0:+,.0f}: delta {pooled['d_mean'] - b0:+,.0f}")
    if a.expect_zero:
        nz = sum(v["nonzero"] for v in cells.values())
        ok = nz == 0
        lines.append(f"  G0 exact parity: {'PASS' if ok else 'FAIL'} ({nz} games with D != 0)")
    print("\n".join(lines))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if ok else 1


def main():
    # Some public programs iterate sets/dicts of strings, so a game depends on the hash seed
    # (env-architect, 2026-09-27: seed 130000009 flips by $6). Pin it; forked workers inherit it.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("control")
    c.add_argument("--cols", default=",".join(G1_COLS))
    c.add_argument("--seeds", type=int, default=16)
    c.add_argument("--seed0", type=int, default=130_000_000)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--control", required=True, help="control JSON written by the `control` mode")
    p.add_argument("--candidate", required=True)
    p.add_argument("--row", default="fs1_t0")
    p.add_argument("--b0")
    p.add_argument("--expect-zero", action="store_true")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "control":
        control(a)
        return 0
    return pair(a)


if __name__ == "__main__":
    sys.exit(main())
