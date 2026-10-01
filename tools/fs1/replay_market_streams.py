#!/usr/bin/env python3
"""Extract per-step market order streams of top-ranked seats from LB replays, for the phantom seat.

`kaggri.PyVecEnv.set_phantom_seats` turns a seat into a market-pressure stream: its SELL / BUY_PRODUCT
orders move `market.inventory` and nothing else. Feeding a recorded top team's realised order stream to a
phantom seat exposes the training seat to top-of-field market pressure that no public program produces.

Output (JSON): {"since", "min_rank", "streams": [{"ep", "seat", "team", "lb_rank", "final", "market": [[orders]*T]}]}
where `market[t]` is the recorded seat's `action.market` at step t filtered to SELL / BUY_PRODUCT rows.

    python3 tools/fs1/replay_market_streams.py --catalog data/replays_2026-09-26/catalog.json --max-rank 20 \
        --out data/fs1_demos/phantom_streams_top20.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

KEEP = ("SELL", "BUY_PRODUCT")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/replays_2026-09-26/catalog.json")
    ap.add_argument("--max-rank", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cat = json.load(open(a.catalog))
    streams, n_ep, skipped = [], 0, 0
    for e in cat["episodes"]:
        lb = e.get("leaderboard") or []
        seats = [s for s, row in enumerate(lb) if row and row.get("lb_rank") and int(row["lb_rank"]) <= a.max_rank]
        if not seats:
            continue
        try:
            d = json.load(open(e["path"]))
        except Exception:
            skipped += 1
            continue
        steps = d["steps"]
        n_ep += 1
        for s in seats:
            market, money = [], []
            for st in steps:
                act = (st[s].get("action") or {}) if s < len(st) else {}
                mk = act.get("market") or []
                market.append([[str(o[0]), str(o[1]), int(o[2])] for o in mk
                               if isinstance(o, list) and len(o) >= 3 and o[0] in KEEP])
                # the recorded seat's money BEFORE this step resolves (the virtual opponent's potential)
                try:
                    money.append(float(st[s]["observation"]["farms"][s]["money"]))
                except (KeyError, IndexError, TypeError):
                    money.append(money[-1] if money else 3000.0)
            streams.append({"ep": int(e["episode_id"]), "seat": s, "team": lb[s].get("team_name") or lb[s].get("team"),
                            "lb_rank": int(lb[s]["lb_rank"]), "final": float(e["rewards"][s]), "market": market,
                            "money": money})
    out = {"since": cat.get("since"), "min_rank": a.max_rank, "n_episodes": n_ep, "streams": streams}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    n_sell = sum(sum(1 for t in s["market"] for o in t if o[0] == "SELL") for s in streams)
    print(f"{len(streams)} seat-streams from {n_ep} episodes (skipped {skipped}); {n_sell} SELL orders "
          f"({n_sell / max(1, len(streams)):.0f}/game); wrote {a.out}")


if __name__ == "__main__":
    main()
