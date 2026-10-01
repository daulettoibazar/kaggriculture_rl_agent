#!/usr/bin/env python3
"""fs1 trainer: self-play PPO from scratch, one process per GPU, DistributedDataParallel.

Launch (one rank):     python3 tools/fs1/train.py --run-dir runs/learning/fs1_smoke --n-envs 64 ...
Launch (local):        scripts/rl.sh (single process, or torchrun with one rank per GPU), with
                       RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT in the env.

Every rank owns `--n-envs` phase-staggered environments (2 seats each). Envs [0, n_selfplay)
are latest-vs-latest and both seats train; envs [n_selfplay, n) are league envs whose seat 1 is
played by a frozen opponent (last-best or a PFSP pool member) and does not train. Rollouts of
`--horizon` steps feed a clipped PPO update; gradients all-reduce through DDP; the league gate
and pool decisions are computed identically on every rank (all-reduced win fractions, identical
weights), so nothing else is communicated.

Thread pinning happens before `torch` / `kaggri` are imported (see `_pin_threads`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import signal
import sys
import time


def _pin_threads():
    """kaggri's rayon pool and BLAS must not oversubscribe the rank's CPU share."""
    cpus = os.environ.get("FS1_RAYON_THREADS") or "8"
    os.environ.setdefault("RAYON_NUM_THREADS", str(cpus))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")


_pin_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.fs1 import constants as C  # noqa: E402
from src.fs1.env import Fs1VecEnv  # noqa: E402
from src.fs1.league import League  # noqa: E402
from src.fs1.model import Fs1Config, Fs1Policy  # noqa: E402
from src.fs1.opener import opener_actions, prefix_length  # noqa: E402
from src.fs1 import achievements as A  # noqa: E402
from src.fs1.public_opponents import PublicOpponents  # noqa: E402
from src.fs1.teacher_prefix import TeacherPrefix  # noqa: E402
from src.fs1.ppo import PPOConfig, Rollout, RunningScale, _global_mean_std, compute_gae, ppo_update  # noqa: E402
from src.fs1.residual_ppo import compute_gae_day  # noqa: E402
from src.fs1.reward import alpha_from_progress, shaped_reward, ramp_with_hold  # noqa: E402


# ----------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file whose keys override the defaults below")
    p.add_argument("--run-dir", required=False)
    p.add_argument("--n-envs", type=int, default=2048, help="environments per rank (x2 seats)")
    p.add_argument("--league-frac", type=float, default=0.3)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--deadline-utc", default="",
                   help="absolute stop, e.g. 2026-09-29T23:30:00: checkpoint and exit 0 once past it")
    p.add_argument("--episode-steps", type=int, default=C.EPISODE_STEPS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--chunk-envs", type=int, default=8)
    # model
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--d-target", type=int, default=64)
    p.add_argument("--hist-k", type=int, default=4)
    p.add_argument("--market-noop-bias", type=float, default=8.0, help="no-op logit prior on the 12 spend heads")
    p.add_argument("--market-sell-noop-bias", type=float, default=4.0, help="no-op logit prior on the 9 sell heads")
    p.add_argument("--no-centralised-critic", action="store_true")
    # ppo
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--vf-win-coef", type=float, default=0.25)
    p.add_argument("--ent-start", type=float, default=1e-3)
    p.add_argument("--ent-end", type=float, default=1e-4)
    p.add_argument("--kl-lb-coef", type=float, default=0.0, help="KL(last-best || policy); costs an extra forward per step")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # reward
    p.add_argument("--alpha-ramp-frac", type=float, default=0.3, help="alpha reaches 1 at this progress")
    p.add_argument("--alpha-start", type=float, default=0.0, help="alpha at progress 0 (warm start for a resume)")
    p.add_argument("--alpha-hold", type=float, default=0.0, help="alpha stays at --alpha-start until this progress")
    p.add_argument("--w-win", type=float, default=1.0)
    p.add_argument("--win-ramp-frac", type=float, default=0.3, help="win term reaches w-win at this progress")
    p.add_argument("--w-win-start", type=float, default=0.0, help="win term multiplier at progress 0 (warm start)")
    p.add_argument("--w-win-hold", type=float, default=0.0, help="win term stays at --w-win-start x w-win until this progress")
    # curriculum
    p.add_argument("--opener-max-len", type=int, default=240, help="longest scripted-opener prefix (steps) at progress 0")
    p.add_argument("--opener-ramp-frac", type=float, default=0.3, help="prefix length reaches 0 at this progress")
    p.add_argument("--lock-expensive-until", type=float, default=0.15,
                   help="force BUY_LAND / BUY_ANIMAL heads to no-op until this progress fraction")
    p.add_argument("--ach-bonus", type=float, default=0.03,
                   help="one-time reward per first-reached milestone per episode (reward units; 0.03 = $300)")
    p.add_argument("--ach-ramp-frac", type=float, default=0.5, help="achievement bonus reaches 0 at this progress")
    # league
    p.add_argument("--gate-every", type=int, default=50)
    p.add_argument("--gate-games", type=int, default=512, help="total across ranks")
    p.add_argument("--gate-threshold", type=float, default=0.6)
    p.add_argument("--gate-temperature", type=float, default=1.0,
                   help="0 = greedy gate; 1.0 = both sides sample as in training (pilot 2: greedy play is a weaker, different player)")
    p.add_argument("--pool-every", type=int, default=200)
    p.add_argument("--pool-max", type=int, default=16)
    p.add_argument("--last-best-share", type=float, default=0.5)
    # io
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--keep-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--public-opponents", default="",
                   help="comma-separated src.rl3.opponents names played as frozen seat-1 opponents (Kaggle programs)")
    p.add_argument("--public-envs", type=int, default=0, help="league envs per rank whose seat 1 is a public program")
    p.add_argument("--public-timeout", type=float, default=1.0, help="per-turn deadline for a public program (s)")
    p.add_argument("--public-workers", type=int, default=0,
                   help="> 0: run the public programs in this many worker processes per rank "
                        "(src/fs1/public_pool.py, shadow games, parity-checked every step); 0 = serial")
    p.add_argument("--teacher-envs", type=int, default=0,
                   help="self-play envs per rank whose episodes open with BOTH seats played by a public program")
    p.add_argument("--teacher-opponents", default="", help="registry names of the teacher programs (default: public-opponents)")
    p.add_argument("--teacher-max-len", type=int, default=0, help="longest program-played prefix (steps) at progress 0")
    p.add_argument("--teacher-ramp-frac", type=float, default=0.5, help="teacher prefix reaches 0 at this progress")
    p.add_argument("--bc-coef", type=float, default=0.0,
                   help="imitation weight on teacher-labelled rows (per-component NLL); needs --teacher-envs")
    p.add_argument("--bc-minibatch", type=int, default=512)
    p.add_argument("--ref-policy", default=None,
                   help="frozen reference policy checkpoint (e.g. the BC clone); with --kl-lb-coef > 0 the KL term anchors the "
                        "policy to it instead of to the league last-best")
    # rl_last (research/rl_last/DESIGN.md): clone -> self-play
    p.add_argument("--relax-masks", action="store_true", help="F1 unit-mask relaxation (src/fs1/masks.py)")
    p.add_argument("--kl-lb-end", type=float, default=None,
                   help="KL anchor coefficient at progress 1 (linear from --kl-lb-coef); default constant")
    p.add_argument("--day-lambda", type=float, default=0.0,
                   help="> 0: two advantage streams, the 12 spend heads on a day-level GAE with this lambda")
    p.add_argument("--critic-warmup", type=int, default=0,
                   help="first N updates of this run train only the value head (policy lr 0)")
    p.add_argument("--train-temperature", type=float, default=1.0,
                   help="!= 1: every policy (learner, reference, league, gate) is softmax(logits / T) in act AND evaluate, "
                        "so PPO trains the tempered policy the bundle plays; export with --temperature T")
    p.add_argument("--win-adv-coef", type=float, default=0.0,
                   help="> 0: the win term leaves the main reward and gets its own advantage from the win value head "
                        "(GAE on the +-w_win terminal stream), normalised separately and added with this weight to both "
                        "streams' normalised advantages; use with --w-win 1")
    p.add_argument("--mb-over-valid", action="store_true",
                   help="minibatches per update = ceil(max over ranks of the VALID rows / minibatch) instead of "
                        "ceil(T*N / minibatch), which re-uses 1/valid_frac - 1 of the rows (60%% with 3/8 frozen seats)")
    p.add_argument("--q4-intervene-frac", type=float, default=0.0,
                   help="> 0: in this share of episodes force seat 0's 4th-quadrant BUY_LAND on the first legal step "
                        "from a random day in [--q4-day-lo, --q4-day-hi] (src/fs1/q4.py); the forced step is not trained on")
    p.add_argument("--q4-day-lo", type=int, default=8)
    p.add_argument("--q4-day-hi", type=int, default=14)
    p.add_argument("--debug-checks", action="store_true",
                   help="crash forensics (a run once died on a CUDA index assert with no Python stack)"
                        "every thread's stack on the watchdog's abort, and host-side range checks with clear messages on "
                        "the league rows, the packed actions, the observations and the PPO minibatch indices")
    # rl_last review 2026-09-28 (research/rl_last/REVIEW_2026-09-28.md), both default off
    p.add_argument("--ref-refresh-every", type=int, default=0,
                   help="> 0: every N updates copy the live weights into the --ref-policy anchor (a trailing trust "
                        "region instead of a leash to the init policy, which the events show binding: kl_lb flat at "
                        "0.055-0.062 in an early run)")
    p.add_argument("--phantom-envs", type=int, default=0,
                   help="> 0: the LAST K public envs per rank get a phantom seat 1 (kaggri set_phantom_seats) that "
                        "replays a recorded top-team SELL/BUY_PRODUCT stream from --phantom-streams; the public "
                        "program block shrinks by K")
    p.add_argument("--phantom-streams", default="",
                   help="JSON from tools/fs1/replay_market_streams.py (per-step market orders of top-ranked seats)")
    p.add_argument("--ach-only", default="",
                   help="comma-separated milestone names (src/fs1/achievements.NAMES) that keep their --ach-bonus "
                        "weight; every other milestone pays 0 (review Q4 test: quadrant4,planted_70)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="checkpoint to initialise a NEW run from (weights, optimizer, ret_scale, league); "
                        "schedules restart at progress 0; ignored when the run dir already has latest.pt")
    p.add_argument("--split-param-groups", action="store_true",
                   help="keep the value/policy optimizer groups without a critic warm-up (to --init-from an optimizer "
                        "saved by a run that had one)")
    p.add_argument("--init-no-optimizer", action="store_true", help="with --init-from: fresh optimizer state")
    p.add_argument("--init-no-league", action="store_true", help="with --init-from: start with an empty league")
    p.add_argument("--init-no-ret-scale", action="store_true",
                   help="with --init-from: fresh return scale (use when the reward mix changes, e.g. a new alpha)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cpu", action="store_true", help="run on CPU (tests)")
    a = p.parse_args(argv)
    if a.config:
        # precedence: explicit command-line flags > config file > the defaults above
        with open(a.config) as f:
            over = json.load(f)
        defaults = {}
        for k, v in over.items():
            if k.startswith("_"):
                continue
            key = k.replace("-", "_")
            if not hasattr(a, key):
                raise SystemExit(f"unknown config key {k}")
            defaults[key] = v
        p.set_defaults(**defaults)
        a = p.parse_args(argv)
    if not a.run_dir:
        raise SystemExit("--run-dir is required")
    return a


# ------------------------------------------------------------------------ helpers
def ddp_setup(cpu: bool = False):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        backend = os.environ.get("FS1_DDP_BACKEND") or ("gloo" if cpu else "nccl")
        if backend == "nccl":
            dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local))
        else:
            dist.init_process_group(backend, rank=rank, world_size=world)
    return world, rank, local


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def progress_at(update: int, elapsed: float, a) -> float:
    """Run progress in [0, 1]: by updates, or by wall clock when --max-seconds is the binding stop."""
    frac = update / max(1, a.updates)
    if a.max_seconds and a.max_seconds > 0:
        frac = max(frac, elapsed / a.max_seconds)
    return float(min(1.0, frac))


def lr_at(update: int, progress: float, a) -> float:
    if update < a.warmup:
        return a.lr * (update + 1) / a.warmup
    w = a.warmup / max(1, a.updates)
    frac = min(1.0, max(0.0, (progress - w) / max(1e-9, 1.0 - w)))
    return a.lr * (a.lr_min_frac + (1 - a.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def ent_at(progress: float, a) -> float:
    return a.ent_start * (a.ent_end / a.ent_start) ** min(1.0, progress)


def make_optimizer(model, a):
    """AdamW with decay / no-decay groups; with --critic-warmup each group is split again into
    value-head and policy parameters (`policy` flag) so the warm-up can zero the policy lr."""
    split = getattr(a, "critic_warmup", 0) > 0 or getattr(a, "split_param_groups", False)
    groups: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = p.ndim >= 2 and "pos" not in n and "emb" not in n
        policy = split and not n.startswith("value_head")
        groups.setdefault((decay, policy), []).append(p)
    return torch.optim.AdamW(
        [{"params": ps, "weight_decay": a.weight_decay if decay else 0.0, "policy": policy}
         for (decay, policy), ps in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1]))],
        lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
    )


class Log:
    """Rank 0 writes events.jsonl (every number already reduced across ranks) and, when FS1_WANDB=1,
    mirrors each record into ONE wandb run for the whole multi-node training (the run id is kept
    in <run-dir>/wandb_sync.id so a requeue resumes the same run; do not also run
    tools/fs1/wandb_sync.py on such a run)."""

    def __init__(self, run_dir: pathlib.Path, rank: int, config: dict | None = None):
        self.rank = rank
        self.path = run_dir / "events.jsonl"
        self.wb = None
        if rank == 0 and os.environ.get("FS1_WANDB", "0") == "1":
            try:
                import wandb
                sys.path.insert(0, str(REPO / "tools" / "fs1"))
                from wandb_sync import flatten, open_run  # noqa: E402
                self._flatten = flatten
                self.wb = open_run(run_dir, f"fs1_{run_dir.name}", os.environ.get("WANDB_PROJECT", "kaggriculture"),
                                   os.environ.get("WANDB_ENTITY"))
                if config:
                    self.wb.config.update(config, allow_val_change=True)
                print(f"wandb run: {self.wb.url}", flush=True)
            except Exception as e:  # never let logging kill a 5-day run
                print(f"wandb disabled: {e!r}", flush=True)
                self.wb = None

    def write(self, rec: dict):
        if self.rank != 0:
            return
        rec = {"ts": time.time(), **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wb is not None:
            try:
                if rec.get("event") == "gate":
                    sc = rec.get("score")
                    self.wb.log({"update": rec["update"], "gate/score": (float("nan") if sc is None else sc),
                                 "gate/promoted": int(bool(rec.get("promoted")))})
                else:
                    flat = self._flatten(rec)
                    flat["update"] = rec["update"]
                    self.wb.log(flat)
            except Exception as e:
                print(f"wandb log failed: {e!r}", flush=True)
        u = rec.get("update")
        keys = ("money_med", "money_mean", "win_vs_frozen", "score", "ent", "kl_old", "clipfrac", "lr",
                "alpha", "env_steps_s", "t_collect", "t_update")
        parts = [f"{k}={rec[k]:.4g}" for k in keys if k in rec]
        print(f"[{u}] " + " ".join(parts), flush=True)


def _check_actions(actions: torch.Tensor, update: int, t: int, rank: int):
    """--debug-checks: every packed action component inside its range (targets < 100, verbs < 38, market heads
    < their sizes), raised on the host with the offending rows instead of a device-side assert."""
    a = actions.detach().to("cpu", torch.int64)
    hi = torch.tensor([C.N_TARGETS] * C.MAX_UNITS + [C.N_VERBS] * C.MAX_UNITS + list(C.MARKET_HEAD_SIZES))
    bad = (a < 0) | (a >= hi)
    if bool(bad.any()):
        r, c = torch.nonzero(bad)[0].tolist()
        raise RuntimeError(f"rank {rank} update {update} step {t}: action row {r} component {c} = {int(a[r, c])} "
                           f"outside [0, {int(hi[c])}) ({int(bad.any(-1).sum())} bad rows)")


# --------------------------------------------------------------------- the gate
@torch.no_grad()
def gate_eval(model, opponent, a, device, world, rank, n_games_total: int, seed: int) -> float:
    """`model` vs `opponent`, both at `--gate-temperature` (0 = greedy), half the games seat-swapped.

    Returns the model's global score in [0, 1] (win 1, tie 0.5, loss 0).
    """
    n = max(2, n_games_total // world)
    n -= n % 2
    seeds = [seed * 1_000_003 + rank * n + i for i in range(n)]
    env = Fs1VecEnv(n, seeds, device, hist_k=a.hist_k, episode_steps=a.episode_steps, stagger=False,
                    chunk_envs=a.chunk_envs, relax_masks=a.relax_masks)
    pair = Fs1VecEnv.pair_index(n, device)
    swap = torch.zeros(n, dtype=torch.bool, device=device)
    swap[n // 2:] = True                              # model plays seat 1 in the second half
    model_rows = torch.where(swap, torch.arange(n, device=device) * 2 + 1, torch.arange(n, device=device) * 2)
    opp_rows = model_rows ^ 1
    score = torch.zeros((), device=device)
    counted = torch.zeros((), device=device)
    for _ in range(a.episode_steps + 2):
        obs, um, tm = env.flat()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=obs.is_cuda):
            g, temp = a.gate_temperature <= 0, max(a.gate_temperature, 1e-6)
            am, _, _, _, _ = model.act(obs[model_rows], um[model_rows], tm[model_rows], None, greedy=g, temperature=temp)
            ao, _, _, _, _ = opponent.act(obs[opp_rows], um[opp_rows], tm[opp_rows], None, greedy=g, temperature=temp)
        actions = torch.zeros(2 * n, C.PLAYER_ACTION_DIM, dtype=torch.long, device=device)
        actions[model_rows] = am
        actions[opp_rows] = ao
        done, final_money, _ = env.step(actions.view(n, 2, -1))
        if bool(done.any()):
            fm = final_money[done]                      # [k, 2]
            sw = swap[done]
            mine = torch.where(sw, fm[:, 1], fm[:, 0])
            theirs = torch.where(sw, fm[:, 0], fm[:, 1])
            score += (0.5 * (torch.sign(mine - theirs) + 1.0)).sum()
            counted += float(done.sum())
        if float(counted) >= n:
            break
    tot = all_reduce_sum(torch.stack([score, counted]))
    return float(tot[0] / tot[1].clamp_min(1.0))


# ------------------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    if a.debug_checks:
        import faulthandler
        faulthandler.enable(all_threads=True)          # SIGABRT from the NCCL watchdog -> the main thread's Python stack
    world, rank, local = ddp_setup(a.cpu)
    device = torch.device("cpu") if a.cpu else torch.device("cuda", local)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(a.seed * 7919 + rank)
    rng = np.random.default_rng(a.seed * 104729 + rank)
    run_dir = pathlib.Path(a.run_dir).resolve()
    # Resolve path-valued args now: the process chdirs into the run directory below.
    for _k in ("init_from", "ref_policy", "phantom_streams"):
        if getattr(a, _k):
            setattr(a, _k, str(pathlib.Path(getattr(a, _k)).resolve()))
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "league").mkdir(exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    use_public = a.public_envs > 0 and bool(a.public_opponents)
    use_teacher = a.teacher_envs > 0 and a.teacher_max_len > 0 and bool(a.teacher_opponents or a.public_opponents)
    if use_public or use_teacher:
        # public programs write files into the CWD: give every rank its own scratch directory
        cwd = run_dir / "agent_cwd" / f"rank{rank}"
        cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(cwd)
    log = Log(run_dir, rank)

    mcfg = Fs1Config(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, mlp_ratio=a.mlp_ratio,
                     d_target=a.d_target, hist_k=a.hist_k, market_noop_bias=a.market_noop_bias,
                     market_sell_noop_bias=a.market_sell_noop_bias,
                     centralised_critic=not a.no_centralised_critic)
    pcfg = PPOConfig(horizon=a.horizon, epochs=a.epochs, minibatch=a.minibatch, grad_accum=a.grad_accum,
                     gamma=a.gamma,
                     gae_lambda=a.gae_lambda, clip=a.clip, vf_coef=a.vf_coef, vf_win_coef=a.vf_win_coef,
                     kl_lb_coef=a.kl_lb_coef, max_grad_norm=a.max_grad_norm,
                     bc_coef=a.bc_coef, bc_minibatch=a.bc_minibatch, mb_over_valid=a.mb_over_valid,
                     debug_checks=a.debug_checks)

    model = Fs1Policy(mcfg).to(device)
    model.logit_temp = a.train_temperature

    def make_frozen():
        m = Fs1Policy(mcfg).to(device)
        m.logit_temp = a.train_temperature
        return m
    optimizer = make_optimizer(model, a)
    ret_scale = RunningScale(pcfg.ret_scale_ema, device)
    league = League(make_frozen, pool_max=a.pool_max, last_best_share=a.last_best_share)

    # ---- resume
    start_update = 0
    elapsed_prev = 0.0
    upd_base = 0                 # league snapshots are named by upd_base + update: unique across --init-from chains
    latest = run_dir / "latest.pt"
    if latest.exists() and not a.no_resume:
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        ret_scale.load_state_dict(ck["ret_scale"])
        start_update = int(ck["update"]) + 1
        elapsed_prev = float(ck.get("elapsed", 0.0))
        upd_base = int(ck.get("upd_base", 0))
        lg = ck.get("league", {})
        if lg.get("last_best"):
            league.set_last_best(lg["last_best"], torch.load(run_dir / "league" / f"{lg['last_best']}.pt",
                                                             map_location=device))
        for name in dict.fromkeys(lg.get("pool", [])):                 # de-duplicated, order kept
            league.add_to_pool(name, torch.load(run_dir / "league" / f"{name}.pt", map_location=device))
        del ck
        if rank == 0:
            print(f"resumed from {latest} at update {start_update}", flush=True)
    else:
        init_update = None
        if a.init_from:
            src = pathlib.Path(a.init_from)
            ck = torch.load(src, map_location=device, weights_only=False)
            if Fs1Config.from_dict(ck.get("model_cfg") or {}).to_dict() != mcfg.to_dict():   # defaults fill new keys
                raise SystemExit(f"--init-from model config {ck.get('model_cfg')} != this run's {mcfg.to_dict()}")
            model.load_state_dict(ck["model"])
            if not a.init_no_optimizer:
                optimizer.load_state_dict(ck["optimizer"])
            if not a.init_no_ret_scale:
                ret_scale.load_state_dict(ck["ret_scale"])
            init_update = int(ck["update"])
            upd_base = int(ck.get("upd_base", 0)) + init_update + 1      # continue the source chain's numbering
            lg = ck.get("league", {}) if not a.init_no_league else {}
            names = list(dict.fromkeys(([lg["last_best"]] if lg.get("last_best") else []) + list(lg.get("pool", []))))
            if rank == 0:
                for name in names:                       # the frozen snapshots live beside the source checkpoint
                    snap = src.parent / "league" / f"{name}.pt"
                    if not snap.exists():
                        raise SystemExit(f"--init-from: league snapshot {snap} not found (pass the checkpoint inside its "
                                         f"run dir, or --init-no-league)")
                    shutil.copyfile(snap, run_dir / "league" / f"{name}.pt")
            if dist.is_initialized():
                dist.barrier()
            if lg.get("last_best"):
                league.set_last_best(lg["last_best"], torch.load(run_dir / "league" / f"{lg['last_best']}.pt",
                                                                 map_location=device))
            for name in dict.fromkeys(lg.get("pool", [])):
                league.add_to_pool(name, torch.load(run_dir / "league" / f"{name}.pt", map_location=device))
            del ck
            if rank == 0:
                print(f"initialised from {src} (update {init_update}, league {names}, snapshot numbering from "
                      f"{upd_base}); schedules restart at 0", flush=True)
        if rank == 0:
            (run_dir / "config.json").write_text(json.dumps(
                {"args": vars(a), "model": mcfg.to_dict(), "ppo": pcfg.to_dict(), "world_size": world,
                 "n_params": model.n_params(), "init_from": a.init_from, "init_update": init_update}, indent=2))

    ref_model = None
    if a.ref_policy:
        rck = torch.load(pathlib.Path(a.ref_policy).resolve(), map_location=device, weights_only=False)
        if Fs1Config.from_dict(rck.get("model_cfg") or {}).to_dict() != mcfg.to_dict():
            raise SystemExit(f"--ref-policy model config differs: {rck.get('model_cfg')} vs {mcfg.to_dict()}")
        ref_model = make_frozen()
        ref_model.load_state_dict(rck["model"])
        ref_model.eval()
        for prm in ref_model.parameters():
            prm.requires_grad_(False)
        del rck
        if rank == 0:
            print(f"KL anchor: reference policy {a.ref_policy} (kl-lb-coef {a.kl_lb_coef})", flush=True)
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None)
    if a.compile:
        ddp = torch.compile(ddp)

    # ---- envs
    n = a.n_envs
    n_league = int(round(a.league_frac * n))
    n_self = n - n_league
    seeds = [a.seed * 1_000_003 + (rank * n + i) for i in range(n)]
    env = Fs1VecEnv(n, seeds, device, hist_k=a.hist_k, episode_steps=a.episode_steps, stagger=True,
                    stagger_seed=a.seed + rank, chunk_envs=a.chunk_envs, relax_masks=a.relax_masks)
    pair = Fs1VecEnv.pair_index(n, device)
    N = 2 * n
    two_stream = a.day_lambda > 0
    model.split_logp = two_stream              # act / evaluate return (step | strategy) log-probs
    ro = Rollout(a.horizon, N, mcfg.in_dim, mcfg.d_model, device, with_lb=a.kl_lb_coef > 0, split=two_stream)
    frozen_rows = torch.arange(n_self, n, device=device) * 2 + 1          # seat 1 of league envs
    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[frozen_rows] = False
    opp_of_env: list = [None] * n_league                                  # Opponent object per league env
    prefix_len = np.zeros(n, dtype=np.int64)          # scripted-opener prefix per env, sampled at each episode start
    prefix_rng = np.random.default_rng(a.seed * 7 + rank + 1)
    lock_heads = torch.zeros(C.N_MARKET_HEADS, dtype=torch.bool, device=device)
    lock_heads[1] = True                              # buy_land
    lock_heads[7:10] = True                           # buy_animal x3
    n_public = min(int(a.public_envs), n_league) if use_public else 0
    if use_public and n_public < int(a.public_envs) and rank == 0:
        print(f"WARNING: --public-envs {a.public_envs} clamped to the {n_league} league envs per rank", flush=True)
    n_frozen_end = n - n_public                       # league envs [n_self, n_frozen_end) play frozen fs1 snapshots
    pub = None                                        # league envs [n_frozen_end, n) play public programs on seat 1
    q4 = None
    if a.q4_intervene_frac > 0:
        from src.fs1.q4 import LAND_HEAD, Q4Intervention
        q4 = Q4Intervention(n, a.q4_intervene_frac, a.q4_day_lo, a.q4_day_hi, seed=a.seed * 7 + 13 * rank)
        if rank == 0:
            print(f"q4 intervention: frac {a.q4_intervene_frac} days {a.q4_day_lo}-{a.q4_day_hi}", flush=True)
    phantom = None                                    # the last K of them: seat 1 = a recorded top-team order stream
    norm_mask = None                                  # rows defining ret_scale / advantage stats (None = all valid)
    n_phantom = min(int(a.phantom_envs), n_public) if a.phantom_envs > 0 else 0
    if n_phantom > 0:
        from src.fs1.phantom import PhantomStreams
        phantom = PhantomStreams(a.phantom_streams, range(n - n_phantom, n), seed=a.seed * 19 + rank,
                                 episode_steps=a.episode_steps)
        env.env.set_phantom_seats(phantom.seat_array(n))
        n_public -= n_phantom                         # the program pool keeps [n_frozen_end, n - n_phantom)
        # phantom rows train but never define ret_scale or the advantage statistics (their critic bootstrap is
        # out of distribution: the first smoke inflated ret_scale 0.66 -> 1.03 in 6 updates)
        norm_mask = torch.ones(N, dtype=torch.bool, device=device)
        norm_mask[torch.tensor(phantom.envs, device=device) * 2] = False
        if rank == 0:
            print(f"phantom seats: {n_phantom} envs/rank replay {phantom.meta}", flush=True)
    if n_public > 0:
        if a.public_workers > 0:
            from src.fs1.public_pool import ParallelPublicOpponents
            pub = ParallelPublicOpponents(a.public_opponents.split(","), n_public, n_frozen_end,
                                          timeout=a.public_timeout, seed=a.seed * 13 + rank,
                                          workers=a.public_workers, cwd_root=os.getcwd(),
                                          episode_steps=a.episode_steps)
        else:
            pub = PublicOpponents(a.public_opponents.split(","), n_public, n_frozen_end, timeout=a.public_timeout,
                                  seed=a.seed * 13 + rank)
    teach = None                                      # program-taught prefixes on the first self-play envs
    if use_teacher:
        n_teach = min(int(a.teacher_envs), n_self)
        teach = TeacherPrefix((a.teacher_opponents or a.public_opponents).split(","), range(n_teach),
                              timeout=a.public_timeout, seed=a.seed * 17 + rank)
    tracker = A.AchievementTracker(n, device)
    ach_weights = A.weights_tensor(device)                 # per-milestone multiplier of the base bonus
    if a.ach_only:
        keep = [x.strip() for x in a.ach_only.split(",") if x.strip()]
        unknown = [x for x in keep if x not in A.NAMES]
        if unknown:
            raise SystemExit(f"--ach-only: unknown milestones {unknown}; known: {list(A.NAMES)}")
        ach_weights = ach_weights * torch.tensor([1.0 if nm in keep else 0.0 for nm in A.NAMES], device=device)

    # SIGUSR1 (a scheduler's pre-emption warning, or `kill -USR1`): finish the current
    # update, checkpoint, and exit 75 so the batch script requeues. Reduced so all ranks agree.
    usr1 = {"flag": False}
    signal.signal(signal.SIGUSR1, lambda *_: usr1.__setitem__("flag", True))

    def sync_flag(v: bool) -> bool:
        t = torch.tensor([1.0 if v else 0.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    if rank == 0:
        print(f"fs1: world={world} n_envs/rank={n} (league {n_league}) params={model.n_params():,} "
              f"horizon={a.horizon} device={device}", flush=True)

    t_start = time.time()
    deadline_ts = 0.0
    if a.deadline_utc:
        import datetime as _dt
        deadline_ts = _dt.datetime.fromisoformat(a.deadline_utc).replace(tzinfo=_dt.timezone.utc).timestamp()
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")  # noqa: E731

    # wall-clock progress and the stop flag are reduced across ranks so every rank sees the same
    # schedules and stops on the same update
    def sync_clock():
        t = torch.tensor([elapsed_prev + time.time() - t_start], device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t)

    for update in range(start_update, a.updates):
        if (ref_model is not None and a.ref_refresh_every > 0 and update > start_update
                and (update - start_update) % a.ref_refresh_every == 0):
            ref_model.load_state_dict(model.state_dict())      # trailing trust region (every rank, in lockstep)
        elapsed = sync_clock()
        progress = progress_at(update, elapsed, a)
        lr = lr_at(update, progress, a)
        warm = update - start_update < a.critic_warmup
        for g in optimizer.param_groups:
            g["lr"] = 0.0 if (warm and g.get("policy", False)) else lr
        if a.kl_lb_end is not None:
            pcfg.kl_lb_coef = a.kl_lb_coef + (a.kl_lb_end - a.kl_lb_coef) * progress
        ent_coef = ent_at(progress, a)
        alpha = ramp_with_hold(progress, a.alpha_start, a.alpha_hold, a.alpha_ramp_frac)
        w_win = a.w_win * ramp_with_hold(progress, a.w_win_start, a.w_win_hold, a.win_ramp_frac)
        max_prefix = prefix_length(progress, a.opener_max_len, a.opener_ramp_frac)
        max_teacher = prefix_length(progress, a.teacher_max_len, a.teacher_ramp_frac) if teach is not None else 0
        if teach is not None and update == start_update:
            teach.start_all(env.env, max_teacher)
        market_lock = lock_heads if progress < a.lock_expensive_until else None
        ach_bonus = A.bonus_at(progress, a.ach_bonus, a.ach_ramp_frac)
        ach_counts = torch.zeros(A.N, device=device)
        q4_count = 0
        diag_unplaced = torch.zeros((), device=device)
        diag_place_legal = torch.zeros((), device=device)
        diag_steps = 0.0
        diag_rows_total = 0.0
        teacher_finals = []
        model.eval()

        # -------------------------------------------------------------- collect
        t0 = time.time()
        finals, wins_frozen = [], []
        phantom_finals, phantom_margins = [], []          # this rank's phantom envs: seat-0 final, final - recorded final
        verb_counts = torch.zeros(C.N_VERBS, device=device)
        market_nonzero = torch.zeros(C.N_MARKET_HEADS, device=device)
        opps = league.opponents()
        if opps and n_league:
            for i in range(n_league - n_public):
                if opp_of_env[i] is None:
                    opp_of_env[i] = league.sample(rng)
        ro.clear_bc()
        if teach is not None:
            teach.new_rollout()
        for t in range(a.horizon):
            obs, um, tm = env.flat()
            steps_np = np.asarray(env.env_steps())
            in_prefix_np = steps_np < prefix_len
            if teach is not None:
                in_prefix_np = in_prefix_np | (steps_np < teach.prefix_len(n))
            in_prefix = torch.from_numpy(in_prefix_np).to(device)                         # [n] bool
            with autocast():
                actions, logp, ent, value, vtok = model.act(obs, um, tm, pair, market_lock=market_lock)
                opener_rows = torch.from_numpy(steps_np < prefix_len).to(device)
                if bool(opener_rows.any()):
                    rows = torch.nonzero(opener_rows.repeat_interleave(2), as_tuple=False).squeeze(-1)
                    actions[rows] = opener_actions(obs[rows], um[rows], tm[rows])
                if opps and n_league:
                    groups: dict = {}
                    for i, opp in enumerate(opp_of_env):
                        if opp is not None:
                            groups.setdefault(id(opp), (opp, []))[1].append(i)
                    for opp, envs_i in groups.values():
                        if a.debug_checks and not (0 <= min(envs_i) and max(envs_i) < n_league):
                            raise RuntimeError(f"league env index out of range: {min(envs_i)}..{max(envs_i)} vs {n_league}")
                        rows = torch.as_tensor(envs_i, device=device) * 2 + 1 + 2 * n_self
                        ao, _, _, _, _ = opp.model.act(obs[rows], um[rows], tm[rows], None)
                        actions[rows] = ao
                lb_logp = None
                if ro.lb_logp is not None and ref_model is not None:
                    lb_logp, _, _ = ref_model.evaluate(obs, um, tm, actions, None)
                elif ro.lb_logp is not None and league.last_best is not None:
                    lb_logp, _, _ = league.last_best.model.evaluate(obs, um, tm, actions, None)
            ro.add(t, obs, um, tm, actions, logp, value, vtok[pair], lb_logp)
            if two_stream:
                ro.hour0[t] = torch.from_numpy(steps_np % C.TURNS_PER_DAY == 0).to(device).repeat_interleave(2)
            diag_rows = train_mask & ~in_prefix.repeat_interleave(2)           # the network's own decisions only
            tr = actions[diag_rows]
            verb_counts += torch.bincount(tr[:, C.MAX_UNITS:2 * C.MAX_UNITS].reshape(-1), minlength=C.N_VERBS)
            market_nonzero += (tr[:, 2 * C.MAX_UNITS:] > 0).sum(0)
            priv_tr = obs[diag_rows][:, C.TOK_PRIVATE, C.HDR + C.V_ANIMALS0:C.HDR + C.V_ANIMALS0 + 3].float()
            diag_unplaced += priv_tr.sum() * C.V_ANIMAL_SCALE                 # animals owned but not placed
            diag_place_legal += um[diag_rows][:, :, 15:18].any(-1).any(-1).float().sum()
            diag_steps += float(tr.shape[0])
            diag_rows_total += float(tr.shape[0])

            phi_prev = env.phi.clone()
            intervened = None
            if q4 is not None:                                   # BEFORE the public pool resolves seat 0 into an exact turn (which would drop the override)
                quad = (env.obs_aug[:, 0, C.TOK_GLOBAL, C.HDR + C.G_UNLOCKED_QUADRANTS].float() * 4.0).round().long()
                idx = q4.select(steps_np, quad.cpu().numpy(), env.money[:, 0].cpu().numpy())
                if len(idx):
                    idx_t = torch.as_tensor(idx, device=device)
                    actions[2 * idx_t, LAND_HEAD] = 1                    # seat 0 row of env e is 2e; 1 = BUY_LAND
                    intervened = torch.zeros(n, dtype=torch.bool, device=device)
                    intervened[idx_t] = True
                    q4_count += int(len(idx))
            exact = pub.exact_actions(env.env, n, actions.view(n, 2, -1)) if pub is not None else None
            if phantom is not None:
                if exact is None:
                    exact = [[None, None] for _ in range(n)]
                phantom.fill(exact, steps_np)
            if teach is not None:
                if exact is None:
                    exact = [[None, None] for _ in range(n)]
                teach.fill(exact, env.env, steps_np, t)
                if a.bc_coef > 0:
                    for (lt, e, seat, row55, ulab, mlab) in teach.drain_labels():
                        ro.add_bc(lt, 2 * e + seat, row55, ulab, mlab)
            if a.debug_checks:
                _check_actions(actions, update, t, rank)
            done, final_money, valid = env.step(actions.view(n, 2, -1), exact)
            phi_next = env.phi
            if phantom is not None:
                # margin race against the recorded seat's own money curve (never its constant phantom Phi)
                done_ph = done.cpu().numpy()
                phi_prev, phi_next, final_money = phantom.virtual_opponent(phi_prev, env.phi, final_money, done_ph, steps_np)
                dph = done_ph[phantom.envs]
                if dph.any():
                    fm = final_money[phantom.envs][torch.from_numpy(dph).to(device)]
                    phantom_finals.append(fm[:, 0].reshape(-1))
                    phantom_margins.append((fm[:, 0] - fm[:, 1]).reshape(-1))
            if a.debug_checks and not bool(torch.isfinite(env.obs_aug).all()):
                bad = torch.nonzero(~torch.isfinite(env.obs_aug).view(n, -1).all(-1)).view(-1)[:8].tolist()
                raise RuntimeError(f"rank {rank} update {update} step {t}: non-finite observation in envs {bad}")
            if pub is not None and a.public_workers > 0:        # every step: the pool's shadows follow the env
                pub.on_done(done.cpu().numpy(), final_money.cpu().numpy(), env.env)
            if bool(done.any()):
                done_np = done.cpu().numpy()
                if phantom is not None:
                    phantom.on_done(done_np)
                if pub is not None and a.public_workers == 0:
                    pub.on_done(done_np, final_money.cpu().numpy())
                if teach is not None:
                    had_prefix = torch.from_numpy(teach.prefix_len(n) > 0).to(device) & done
                    if bool(had_prefix.any()):
                        teacher_finals.append(final_money[had_prefix].reshape(-1))
                    teach.on_done(done_np, max_teacher)
            valid = valid & ~in_prefix                                                   # scripted steps do not train
            if intervened is not None:
                valid = valid & ~intervened                                              # nor the forced BUY_LAND step
            if bool(done.any()) and max_prefix > 0:
                d_np = done.cpu().numpy()
                prefix_len[d_np] = prefix_rng.integers(0, max_prefix + 1, size=int(d_np.sum()))
            elif bool(done.any()):
                prefix_len[done.cpu().numpy()] = 0
            r = shaped_reward(phi_prev, phi_next, final_money, done, alpha,
                              0.0 if a.win_adv_coef > 0 else w_win)                          # [n, 2]
            new_ach = tracker.step(env.obs_aug[..., :C.N_FEATURES], done)                    # [n, 2, N] bool
            if ach_bonus > 0:
                r = r + ach_bonus * (new_ach.float() * ach_weights).sum(-1)
            ach_valid = valid if intervened is None else (valid | intervened)   # count the milestone the forced step unlocks
            ach_counts += (new_ach & (ach_valid.unsqueeze(-1) & train_mask.view(n, 2))[..., None]).float().sum((0, 1))
            margin = final_money - final_money.flip(-1)
            r_win = torch.where(done.unsqueeze(-1), w_win * torch.sign(margin), torch.zeros_like(margin))
            valid_seat = valid.unsqueeze(-1) & train_mask.view(n, 2)
            ro.add_outcome(t, r.view(-1), r_win.view(-1), done.repeat_interleave(2), valid_seat.view(-1))

            if bool(done.any()):
                d = done & valid
                if bool(d[:n_self].any()):
                    finals.append(final_money[:n_self][d[:n_self]].reshape(-1))
                dl = d[n_self:n_frozen_end]
                if (n_frozen_end - n_self) and bool(dl.any()):
                    fm = final_money[n_self:n_frozen_end][dl]
                    s = 0.5 * (torch.sign(fm[:, 0] - fm[:, 1]) + 1.0)                 # latest is seat 0
                    wins_frozen.append(s)
                    for i, sc in zip(np.flatnonzero(dl.cpu().numpy()), s.tolist()):
                        if opp_of_env[i] is not None:
                            opp_of_env[i].record(sc)
                        opp_of_env[i] = league.sample(rng) if league.has_opponents() else None
        with autocast():
            obs, um, tm = env.flat()
            _, _, _, last_value, _ = model.act(obs, um, tm, pair)
        t_collect = time.time() - t0

        # --------------------------------------------------------------- update
        t1 = time.time()
        scale = ret_scale.scale
        adv, ret = compute_gae(ro.reward, ro.value[..., 0] * scale, ro.done, last_value[:, 0] * scale,
                               pcfg.gamma, pcfg.gae_lambda)
        adv_win, ret_win = compute_gae(ro.reward_win, ro.value[..., 1], ro.done, last_value[:, 1], 1.0, 1.0)
        adv_day = None
        if two_stream:
            adv_day, _ = compute_gae_day(ro.reward, ro.value[..., 0] * scale, ro.done, ro.hour0,
                                         last_value[:, 0] * scale, a.day_lambda)
        if a.win_adv_coef > 0:
            # decoupled win stream: each advantage normalised on its own, the win part weighted by --win-adv-coef
            # (ppo_update renormalises the sum, so only the ratio matters)
            vmask = ro.valid.float()
            m_w, s_w = _global_mean_std(adv_win, vmask)
            adv_win_n = (adv_win - m_w) / s_w
            m_a, s_a = _global_mean_std(adv, vmask)
            adv = (adv - m_a) / s_a + a.win_adv_coef * adv_win_n
            if adv_day is not None:
                m_a, s_a = _global_mean_std(adv_day, vmask)
                adv_day = (adv_day - m_a) / s_a + a.win_adv_coef * adv_win_n
        model.train()
        stats = ppo_update(ddp, optimizer, ro, adv, ret, ret_win.clamp(-1, 1), pcfg, scale, ent_coef,
                           kl_active=(ref_model is not None) or (league.last_best is not None), market_lock=market_lock,
                           adv_day=adv_day, norm_mask=norm_mask)
        ret_scale.update(ret[ro.valid if norm_mask is None else (ro.valid & norm_mask.view(1, -1))])   # unconditional collective
        if a.win_adv_coef > 0:
            stats["adv_win_std"] = float(s_w)
        t_update = time.time() - t1

        # ---------------------------------------------------------------- stats
        fin = torch.cat(finals) if finals else torch.zeros(0, device=device)
        sums = torch.tensor([fin.sum().item(), float(fin.numel()),
                             (torch.cat(wins_frozen).sum().item() if wins_frozen else 0.0),
                             float(sum(w.numel() for w in wins_frozen)),
                             env.money.mean().item(), 1.0], device=device)
        sums = all_reduce_sum(sums)
        # median of final money: gather on rank 0 is expensive; use this rank's median as a proxy
        rec = {
            "update": update, "lr": lr, "alpha": alpha, "ent_coef": ent_coef, "w_win": w_win, "ach_bonus": ach_bonus,
            "max_prefix": max_prefix, "market_lock": market_lock is not None,
            "frac_in_prefix": float(((np.asarray(env.env_steps()) < prefix_len)
                                     | (np.asarray(env.env_steps()) < (teach.prefix_len(n) if teach is not None else 0))).mean()),
            "money_mean": float(sums[0] / max(1.0, float(sums[1]))),
            "money_med": float(fin.median()) if fin.numel() else float("nan"),
            "finals": int(sums[1]),
            "win_vs_frozen": float(sums[2] / sums[3]) if float(sums[3]) > 0 else float("nan"),
            "live_money_mean": float(sums[4] / sums[5]),
            "env_steps_s": a.horizon * n * world / max(1e-6, t_collect + t_update),
            "t_collect": t_collect, "t_update": t_update,
            "ret_scale": float(ret_scale.scale), "progress": progress, "elapsed": elapsed,
            "kl_lb_coef": pcfg.kl_lb_coef, "critic_warmup": bool(warm),
            "gpu_mem_gb": (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0,
            **stats,
        }
        if (update + 1) % a.ckpt_every == 0:
            rec["league"] = league.summary()
        per_game = max(1.0, diag_rows_total) / a.episode_steps        # the network's own seat-games' worth of steps
        vc = verb_counts / max(1.0, per_game)
        rec["orders_per_game"] = {
            "water": float(vc[1]), "harvest": float(vc[2]), "fertilize": float(vc[3]), "dig": float(vc[4]),
            "build": float(vc[5] + vc[6]), "feed_care": float(vc[7] + vc[8]),
            "plant": float(vc[10:15].sum()), "place_animal": float(vc[15:18].sum()),
            "pickup": float(vc[18:28].sum()), "deposit": float(vc[28:37].sum()), "drop": float(vc[37]),
            "idle": float(vc[0]),
        }
        rec["q4_interventions_per_game"] = float(q4_count) / max(1.0, per_game)
        rec["achievements_per_game"] = {name: float(v) / max(1.0, per_game) for name, v in zip(A.NAMES, ach_counts.tolist())}
        mn = market_nonzero / max(1.0, per_game)
        if pub is not None:
            rec.update(pub.summary(all_reduce_sum(pub.totals(device))))
        if teach is not None:
            tfin = torch.cat(teacher_finals) if teacher_finals else torch.zeros(0, device=device)
            tsum = all_reduce_sum(torch.tensor([tfin.sum().item(), float(tfin.numel()), float(teach.episodes),
                                                float(teach.fallbacks), float(teach.budget_hits)], device=device))
            rec.update(teach.summary())
            rec["teacher_episodes"], rec["teacher_fallbacks"], rec["teacher_budget_hits"] = float(tsum[2]), float(tsum[3]), float(tsum[4])
            rec["teacher_money_mean"] = float(tsum[0] / tsum[1]) if float(tsum[1]) > 0 else float("nan")
            rec["teacher_money_med"] = float(tfin.median()) if tfin.numel() else float("nan")
            rec["teacher_finals"] = int(tsum[1])
            rec["max_teacher_prefix"] = max_teacher
            rec["bc_labels"] = int(ro.bc_unit.any(-1).sum()) + 0   # labelled rows in this rollout (rank 0)
        if phantom is not None:                            # rank-local means (a proxy, like money_med)
            pf = torch.cat(phantom_finals) if phantom_finals else torch.zeros(0, device=device)
            pm = torch.cat(phantom_margins) if phantom_margins else torch.zeros(0, device=device)
            rec["phantom_money_mean"] = float(pf.mean()) if pf.numel() else float("nan")
            rec["phantom_margin_mean"] = float(pm.mean()) if pm.numel() else float("nan")
            rec["phantom_finals"] = int(pf.numel())
        rec["animals_unplaced_mean"] = float(diag_unplaced) / max(1.0, diag_steps)    # per training seat-step
        rec["place_animal_legal_frac"] = float(diag_place_legal) / max(1.0, diag_steps)
        rec["market_per_game"] = {"hire": float(mn[0]), "land": float(mn[1]), "seeds": float(mn[2:7].sum()),
                                  "animals": float(mn[7:10].sum()), "buy_product": float(mn[10:12].sum()),
                                  "sell": float(mn[12:21].sum())}
        if update % a.log_every == 0:
            log.write(rec)

        # --------------------------------------------------------------- league
        if (update + 1) % a.pool_every == 0:
            name = f"u{upd_base + update + 1:06d}"
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if rank == 0:
                torch.save(sd, run_dir / "league" / f"{name}.pt")
            league.add_to_pool(name, sd)
        if (update + 1) % a.gate_every == 0:
            name = f"lb{upd_base + update + 1:06d}"
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if league.last_best is None:
                promote, score = True, float("nan")
            else:
                model.eval()
                score = gate_eval(model, league.last_best.model, a, device, world, rank, a.gate_games,
                                  seed=a.seed * 31 + 7)          # fixed seeds: consecutive gates comparable
                promote = score >= a.gate_threshold
            if promote:
                if rank == 0:
                    torch.save(sd, run_dir / "league" / f"{name}.pt")
                league.set_last_best(name, sd)
            log.write({"update": update, "event": "gate", "score": score, "promoted": promote,
                       "last_best": league.last_best.name if league.last_best else None})

        # ----------------------------------------------------------- checkpoint
        requeue = sync_flag(usr1["flag"])
        if rank == 0 and ((update + 1) % a.ckpt_every == 0 or update + 1 == a.updates or requeue):
            ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                  "elapsed": elapsed, "upd_base": upd_base,
                  "model_cfg": mcfg.to_dict(), "ppo_cfg": pcfg.to_dict(), "args": vars(a),
                  "ret_scale": ret_scale.state_dict(),
                  "league": {"last_best": league.last_best.name if league.last_best else None,
                             "pool": [o.name for o in league.pool]}}
            tmp = run_dir / "latest.pt.tmp"
            torch.save(ck, tmp)
            os.replace(tmp, latest)
            if (update + 1) % a.keep_every == 0:
                torch.save(ck, run_dir / f"ckpt_{update + 1:06d}.pt")
        if requeue:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            if rank == 0:
                print("SIGUSR1: checkpointed, exiting 75 for requeue", flush=True)
            sys.exit(75)
        if a.max_seconds and sync_clock() > a.max_seconds:      # reduced clock: all ranks stop together
            if rank == 0:
                print("max_seconds reached, stopping", flush=True)
            break
        if deadline_ts and sync_flag(time.time() >= deadline_ts):
            if rank == 0:
                ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                      "elapsed": elapsed, "upd_base": upd_base, "model_cfg": mcfg.to_dict(), "ppo_cfg": pcfg.to_dict(),
                      "args": vars(a), "ret_scale": ret_scale.state_dict(),
                      "league": {"last_best": league.last_best.name if league.last_best else None,
                                 "pool": [o.name for o in league.pool]}}
                torch.save(ck, run_dir / "latest.pt.tmp")
                os.replace(run_dir / "latest.pt.tmp", latest)
                print(f"deadline {a.deadline_utc} reached, checkpointed and stopping", flush=True)
            break

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
