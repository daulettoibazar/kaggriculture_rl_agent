#!/usr/bin/env python3
"""Mirror an fs1 run's `events.jsonl` into ONE Weights & Biases run.

    python3 tools/fs1/wandb_sync.py --run-dir runs/<run> [--name <run>] [--once]

The trainer's rank 0 is the only writer of `events.jsonl`, and every number in it is already
reduced across all ranks (money statistics, win rates, achievement counts), so one sync process
gives one run for the whole multi-node training. The sync keeps a byte offset and the wandb run
id in the run directory, so it can be stopped and restarted without duplicating or forking the
run. Nested dicts (`orders_per_game`, `market_per_game`, `achievements_per_game`) are flattened to
`group/key`; gate events are logged as `gate/score` and `gate/promoted` at the gate's update.

Runs as a small daemon (nohup) on the local box; PID in `<run-dir>/wandb_sync.pid`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

FLAT_GROUPS = ("orders_per_game", "market_per_game", "achievements_per_game", "win_vs_public", "win_vs_public_ema",
               "public_games", "public_fallbacks")
# teacher_* and win_vs_* scalars flatten as plain numbers
SKIP = {"league", "ts"}


def flatten(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in SKIP:
            continue
        if k in FLAT_GROUPS and isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    out[f"{k}/{kk}"] = vv
        elif isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = v
    lg = rec.get("league")
    if isinstance(lg, dict):
        if lg.get("last_best"):
            out["league/last_best_winrate"] = lg["last_best"].get("winrate", float("nan"))
        out["league/pool_size"] = len(lg.get("pool", []))
    return out


def open_run(run_dir: pathlib.Path, name: str, project: str, entity: str):
    import wandb

    id_file = run_dir / "wandb_sync.id"
    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}
    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    run = wandb.init(project=project, entity=entity, name=name, id=run_id, resume="allow", config=cfg,
                     settings=wandb.Settings(console="off", disable_code=True, disable_git=True))
    id_file.write_text(run.id)
    run.define_metric("update")
    run.define_metric("*", step_metric="update")
    return run


def sync_once(run, events: pathlib.Path, offset_file: pathlib.Path) -> int:
    offset = int(offset_file.read_text()) if offset_file.exists() else 0
    n = 0
    with open(events, "rb") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break                                   # partial line still being written
            offset = f.tell()
            try:
                rec = json.loads(line)
            except Exception:
                continue
            upd = rec.get("update")
            if upd is None:
                continue
            if rec.get("event") == "gate":
                sc = rec.get("score")
                run.log({"update": upd, "gate/score": (float("nan") if sc is None else sc),
                         "gate/promoted": int(bool(rec.get("promoted")))}, commit=True)
            else:
                flat = flatten(rec)
                flat["update"] = upd
                run.log(flat, commit=True)
            n += 1
    offset_file.write_text(str(offset))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "kaggriculture"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    run_dir = pathlib.Path(a.run_dir).resolve()
    events = run_dir / "events.jsonl"
    offset_file = run_dir / "wandb_sync.offset"
    (run_dir / "wandb_sync.pid").write_text(str(os.getpid()))
    run = open_run(run_dir, a.name or run_dir.name, a.project, a.entity)
    print(f"wandb run: {run.url}", flush=True)
    while True:
        if events.exists():
            n = sync_once(run, events, offset_file)
            if n:
                print(f"{time.strftime('%H:%M:%S')} synced {n} records", flush=True)
        if a.once:
            break
        time.sleep(a.interval)
    run.finish()


if __name__ == "__main__":
    main()
