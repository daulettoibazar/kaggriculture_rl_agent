#!/usr/bin/env python3
"""Export an fs1 checkpoint to a Kaggle bundle directory and (optionally) verify it.

    python3 tools/fs1/export_bundle.py --checkpoint runs/learning/fs1_X/ckpt_001000.pt \
        --out dist/fs1_u1000 [--verify] [--games 2]

Members: main.py, fs1_numpy_policy.py, fs1_policy.npz (float16), kaggri/ (the abi3 extension
package), manifest.json. `--verify` loads the bundle's `agent` from the out directory, plays
full games against PASS in the exact simulator through Kaggle-format observations, checks that
its action rows equal the torch policy's greedy rows on the same states, and prints per-turn
timing (single BLAS thread, the closest local stand-in for Kaggle's 1.6 vCPU).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.fs1 import constants as C  # noqa: E402
from src.fs1.export import load_checkpoint_model, policy_arrays  # noqa: E402

MEMBERS = ("main.py", "fs1_numpy_policy.py", "fs1_policy.npz", "manifest.json", "kaggri/__init__.py")
NOTICE = """This bundle contains a neural policy trained by the submitting team (rl_last line, 2026-09).
Its training used, as opponents and as imitation data, public Kaggle notebooks for the
Kaggriculture competition (Apache-2.0; notably abhinav0370/cha22-agent and the notebooks it
credits: thomastschinkel, yhay81, destbreso, aurax7, tetsutani, prvsiyan, Dmitrii Gluzdov,
Ahmed Berat Ozer) and public leaderboard episode replays. No third-party code is included:
main.py, fs1_numpy_policy.py and the kaggri simulator bindings are the team's own.
"""


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kaggri_package_dir() -> pathlib.Path:
    import kaggri
    return pathlib.Path(kaggri.__file__).resolve().parent


def export(checkpoint: pathlib.Path, out: pathlib.Path, temperature: float = 0.0, relax_masks: bool = False) -> dict:
    model = load_checkpoint_model(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "fs1_policy.npz", **policy_arrays(model, dtype=np.float16))
    shutil.copy(REPO / "submission" / "fs1_main.py", out / "main.py")
    shutil.copy(REPO / "submission" / "fs1_numpy_policy.py", out / "fs1_numpy_policy.py")
    pkg = _kaggri_package_dir()
    dst = out / "kaggri"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for f in pkg.iterdir():
        if f.suffix in (".py", ".so"):
            shutil.copy(f, dst / f.name)
    extra = []
    if temperature > 0 or relax_masks:                  # rl_last: sampled play + the F1 mask relaxation
        (out / "fs1_bundle.json").write_text(json.dumps({"temperature": float(temperature),
                                                          "relax_masks": bool(relax_masks)}) + "\n")
        shutil.copy(REPO / "dist" / "tape_v21_cha22x" / "LICENSE.txt", out / "LICENSE.txt")
        (out / "NOTICE.txt").write_text(NOTICE)
        extra = ["fs1_bundle.json", "LICENSE.txt", "NOTICE.txt"]
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        git = "unknown"
    manifest = {
        "line": "fs1", "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
        "update": int(ck.get("update", -1)), "model_cfg": ck.get("model_cfg"), "n_params": model.n_params(),
        "git": git, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kaggri_so": [f.name for f in dst.iterdir() if f.suffix == ".so"],
        "members": {},
    }
    kaggri_members = [f"kaggri/{f.name}" for f in sorted(dst.iterdir()) if f.suffix in (".py", ".so")]
    for m in ["main.py", "fs1_numpy_policy.py", "fs1_policy.npz"] + extra + kaggri_members:
        manifest["members"][m] = {"sha256": _sha(out / m), "bytes": (out / m).stat().st_size}
    manifest["play"] = {"temperature": float(temperature), "relax_masks": bool(relax_masks)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_bundle_agent(out: pathlib.Path):
    """Import the bundle's main.py as an isolated module (fresh state)."""
    # `kaggri` is NOT evicted: the bundle copy is the same build as the installed package and a
    # second extension copy cannot be loaded into one process. `bundle_imports_alone` checks the
    # bundle's own kaggri/ in a fresh interpreter instead.
    for name in ("fs1_bundle_main", "fs1_numpy_policy"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("fs1_bundle_main", out / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bundle_imports_alone(out: pathlib.Path) -> str:
    """Import the bundle's own kaggri/ and main.py in a fresh interpreter with the user site and
    the repo off the path; returns the resolved kaggri file path (raises on failure)."""
    code = ("import sys, os; sys.path.insert(0, %r); import kaggri, main; "
            "print(kaggri.__file__)" % str(out))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONNOUSERSITE"] = "1"
    res = subprocess.run([sys.executable, "-c", code], cwd=str(out), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"bundle does not import alone:\n{res.stderr[-2000:]}")
    path = res.stdout.strip().splitlines()[-1]
    if not path.startswith(str(out)):
        raise RuntimeError(f"bundle import resolved kaggri outside the bundle: {path}")
    return path


def verify(checkpoint: pathlib.Path, out: pathlib.Path, games: int = 2, seed0: int = 101, sampled_games: int = 4) -> dict:
    import src.evo  # noqa: F401  (pins rayon threads before kaggri)
    import kaggri
    from src.evo.kernel_opponents import observation_of
    from src.fs1.model import Fs1Policy  # noqa: F401

    mod = load_bundle_agent(out)
    model = load_checkpoint_model(checkpoint)
    bundle_temp = mod.TEMPERATURE
    mod.TEMPERATURE = 0.0                      # the parity check is greedy; sampling is checked below
    hist_k = model.cfg.hist_k
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    mism, turns, times, finals = 0, 0, [], []
    for g in range(games):
        game = kaggri.PyGame({"seed": seed0 + g})
        game.reset()
        cfg = game.config()
        hist = torch.zeros(1, len(C.PRODUCTS), hist_k)
        prev_rel = None
        while not game.done:
            sd = game.state_dict()
            kobs = observation_of(sd, 0, struct=False)
            kobs["step"] = game.step_no
            t0 = time.perf_counter()
            action = mod.agent(kobs, cfg)
            times.append((time.perf_counter() - t0) * 1e3)
            # torch greedy on the same state (through the same Encoder planes)
            enc = kaggri.Encoder(cfg, obs_dtype="float32")
            o, m, tm = enc.encode_full(kobs)
            o = torch.from_numpy(np.asarray(o, dtype=np.float32)).view(1, C.N_TOKENS, C.N_FEATURES)
            rel = o[:, C.TOK_PRODUCT0:C.TOK_PRODUCT1, C.HDR + C.P_PRICE_REL]
            delta = torch.zeros_like(rel) if prev_rel is None else rel - prev_rel
            if hist_k:
                hist = torch.roll(hist, 1, -1)
                hist[..., 0] = delta
            prev_rel = rel.clone()
            aug = torch.zeros(1, C.N_TOKENS, hist_k)
            aug[:, C.TOK_PRODUCT0:C.TOK_PRODUCT1] = hist
            obs_aug = torch.cat([o, aug], -1)
            um = torch.from_numpy(np.asarray(m)).view(1, C.MAX_UNITS, C.N_VERBS) > 0.5
            tmask = torch.from_numpy(np.asarray(tm)).view(1, C.N_TARGETS, C.N_VERBS) > 0
            if mod.RELAX_MASKS:
                from src.fs1.masks import relax_unit_mask
                um = relax_unit_mask(um, tmask)
            row_t, *_ = model.act(obs_aug, um, tmask, None, greedy=True)
            want = enc.action(kobs, row_t[0].numpy())
            if want != action:
                mism += 1
            turns += 1
            game.step(action, pass_action)
        finals.append(game.money[0])
    sampled = []
    if bundle_temp > 0:                        # the shipped play mode: sampled, vs PASS
        mod.TEMPERATURE = bundle_temp
        for g in range(sampled_games):
            mod._STATE["hist"] = None
            game = kaggri.PyGame({"seed": seed0 + 1000 + g})
            game.reset()
            cfg = game.config()
            while not game.done:
                kobs = observation_of(game.state_dict(), 0, struct=False)
                kobs["step"] = game.step_no
                t0 = time.perf_counter()
                action = mod.agent(kobs, cfg)
                times.append((time.perf_counter() - t0) * 1e3)
                game.step(action, pass_action)
            sampled.append(round(float(game.money[0])))
    t = np.array(times)
    rep = {"play": {"temperature": bundle_temp, "relax_masks": mod.RELAX_MASKS},
           "sampled_money_vs_pass": sampled,"games": games, "turns": turns, "action_mismatch_vs_torch": mism,
           "kaggri_in_bundle": bundle_imports_alone(out),
           "fallbacks": mod._STATE["fallbacks"], "import_ms": round(mod.IMPORT_MS, 2),
           "turn_ms_mean": round(float(t.mean()), 2), "turn_ms_p99": round(float(np.percentile(t, 99)), 2),
           "turn_ms_max": round(float(t.max()), 2), "final_money_vs_pass": [round(float(x)) for x in finals]}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: the bundle samples at this temperature")
    ap.add_argument("--relax-masks", action="store_true", help="the bundle applies the F1 mask relaxation")
    ap.add_argument("--sampled-games", type=int, default=4)
    a = ap.parse_args(argv)
    ck, out = pathlib.Path(a.checkpoint), pathlib.Path(a.out)
    man = export(ck, out, a.temperature, a.relax_masks)
    print(json.dumps({k: man[k] for k in ("update", "n_params", "git", "kaggri_so")}))
    if a.verify:
        rep = verify(ck, out, a.games, sampled_games=a.sampled_games)
        print(json.dumps(rep))
        if rep["fallbacks"] or rep["action_mismatch_vs_torch"] > 0.02 * rep["turns"]:
            raise SystemExit("bundle verification FAILED")


if __name__ == "__main__":
    main()
