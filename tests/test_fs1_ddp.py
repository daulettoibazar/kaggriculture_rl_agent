"""Two-rank DDP run of the fs1 trainer on CPU with the gloo backend: the collective count must
be identical on every rank even while the ranks disagree on how many samples are valid (they
have different stagger offsets), the run must finish, and both ranks must stop together on
--max-seconds. Exercises the paths that only exist when WORLD_SIZE > 1."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

_CHA22X = pathlib.Path(__file__).resolve().parents[1] / "dist/tape_v21_cha22x/policy.py"
needs_cha22x = pytest.mark.skipif(not _CHA22X.exists(), reason="cha22 agent not vendored; see README")

pytestmark = [pytest.mark.kaggri, pytest.mark.slow, needs_cha22x]

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(run_dir, extra):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29517",
           str(REPO / "tools" / "fs1" / "train.py"), "--run-dir", str(run_dir), "--cpu",
           "--n-envs", "6", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--mlp-ratio", "2",
           "--d-target", "16", "--horizon", "6", "--minibatch", "16", "--grad-accum", "2",
           "--episode-steps", "30", "--gate-every", "2", "--gate-games", "4", "--pool-every", "2",
           "--ckpt-every", "1", "--warmup", "1", "--chunk-envs", "1", "--league-frac", "0.34",
           "--opener-max-len", "8", "--public-envs", "1", "--public-opponents", "cha22x_bundle",
           "--teacher-envs", "1", "--teacher-max-len", "10", "--bc-coef", "0.5", "--bc-minibatch", "8"] + extra
    env = dict(os.environ, FS1_DDP_BACKEND="gloo", CUDA_VISIBLE_DEVICES="", FS1_RAYON_THREADS="1",
               OMP_NUM_THREADS="1")
    return subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)


def test_two_rank_gloo_run_completes_and_resumes(tmp_path):
    run = tmp_path / "ddp"
    # 10 updates x 6 steps = 60 steps: the 30-step episodes restart twice, so the teacher block draws
    # prefixes (staggered first episodes get none) and imitation labels appear
    res = _run(run, ["--updates", "10"])
    assert res.returncode == 0, res.stderr[-3000:]
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    ups = [e for e in events if "env_steps_s" in e]
    assert [e["update"] for e in ups] == list(range(10))
    # both ranks ran the same, constant number of minibatches per update
    assert len({e["n_minibatches"] for e in ups}) == 1
    assert any(e.get("event") == "gate" for e in events)
    assert "cha22x_bundle" in ups[-1]["win_vs_public"] and ups[-1]["public_fallbacks"]["cha22x_bundle"] == 0
    assert "teacher_episodes" in ups[-1] and ups[-1]["teacher_fallbacks"] == 0
    assert "teacher_money_med" in ups[-1] and "teacher_finals" in ups[-1]
    assert "bc_nll" in ups[-1] and "n_bc" in ups[-1] and "bc_labels" in ups[-1]
    assert any(e["n_bc"] > 0 and e["bc_nll"] > 0 for e in ups), [(e["n_bc"], e["bc_nll"]) for e in ups]
    assert not list((run / "agent_cwd").glob("rank*/*.py")), "public program files must stay in agent_cwd"
    # resume with a wall-clock stop: both ranks must exit cleanly together
    res2 = _run(run, ["--updates", "50", "--max-seconds", "5"])
    assert res2.returncode == 0, res2.stderr[-3000:]
    assert "max_seconds reached" in res2.stdout
    # a NEW run initialised from that checkpoint: weights + league carried over, schedules restart at 0
    ck_update = json.loads((run / "events.jsonl").read_text().splitlines()[-1])["update"]
    init = tmp_path / "init"
    res3 = _run(init, ["--updates", "2", "--init-from", str(run / "latest.pt"),
                       "--ref-policy", str(run / "latest.pt"), "--kl-lb-coef", "0.05"])
    assert res3.returncode == 0, res3.stderr[-3000:]
    assert "initialised from" in res3.stdout and "KL anchor" in res3.stdout
    ups3_recs = [e for e in map(json.loads, (init / "events.jsonl").read_text().splitlines()) if "env_steps_s" in e]
    assert any(abs(e["kl_lb"]) > 0 for e in ups3_recs), [e["kl_lb"] for e in ups3_recs]   # anchored to the reference
    cfg = json.loads((init / "config.json").read_text())
    assert cfg["init_from"] == str(run / "latest.pt") and cfg["init_update"] is not None
    ups3 = [e["update"] for e in map(json.loads, (init / "events.jsonl").read_text().splitlines()) if "env_steps_s" in e]
    assert ups3 == [0, 1]
    # exactly the snapshots the checkpoint references (current last-best + pool) are carried over,
    # not the source run's whole history of promotions
    import torch
    lg = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["league"]
    referenced = {f"{n}.pt" for n in ([lg["last_best"]] if lg["last_best"] else []) + list(lg["pool"])}
    dst_league = {p.name for p in (init / "league").glob("*.pt")}
    assert referenced and referenced <= dst_league, (referenced, dst_league)
    assert ck_update >= 3
    # the new run's own snapshots continue the source chain's numbering: no name collides with a copied one
    lg2 = torch.load(init / "latest.pt", map_location="cpu", weights_only=False)["league"]
    pool2 = list(lg2["pool"])
    assert len(pool2) == len(set(pool2)), pool2
    own = sorted(dst_league - referenced)
    assert own, "the resumed run made no snapshot of its own"
    assert all(int(n[1:7] if n.startswith("u") else n[2:8]) > ck_update for n in own), (own, ck_update)


def test_sigusr1_checkpoints_and_exits_75(tmp_path):
    """The USR1 checkpoint path: every rank checkpoints at the end of the current
    update and exits 75 so the batch script can requeue."""
    import signal
    import time

    run = tmp_path / "usr1"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29519",
           str(REPO / "tools" / "fs1" / "train.py"), "--run-dir", str(run), "--cpu",
           "--n-envs", "6", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--mlp-ratio", "2",
           "--d-target", "16", "--horizon", "6", "--minibatch", "16", "--episode-steps", "30",
           "--gate-every", "1000", "--pool-every", "1000", "--ckpt-every", "1000", "--warmup", "1",
           "--chunk-envs", "1", "--updates", "400", "--opener-max-len", "8"]
    env = dict(os.environ, FS1_DDP_BACKEND="gloo", CUDA_VISIBLE_DEVICES="", FS1_RAYON_THREADS="1",
               OMP_NUM_THREADS="1")
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not (run / "events.jsonl").exists():
            time.sleep(0.5)
        assert (run / "events.jsonl").exists(), "trainer never produced an update"
        time.sleep(2.0)
        kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        assert kids, "no worker processes found"
        for pid in kids:
            os.kill(int(pid), signal.SIGUSR1)
        out, _ = proc.communicate(timeout=180)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "exiting 75 for requeue" in out, out[-2000:]
    assert (run / "latest.pt").exists()
    # torchrun reports the workers' non-zero exit as a failure; the workers themselves exited 75
    assert "exitcode" in out and "75" in out, out[-1500:]
