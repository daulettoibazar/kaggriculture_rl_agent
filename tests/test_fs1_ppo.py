"""fs1 PPO pieces: GAE on a toy, the running return scale, and a CPU end-to-end run of the
trainer (tiny model, short episodes) that resumes from its own checkpoint."""
import json
import pathlib
import sys

import pytest
import torch

from src.fs1.ppo import RunningScale, compute_gae

pytestmark = pytest.mark.kaggri


def test_gae_matches_hand_computation_and_respects_terminals():
    reward = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    value = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    done = torch.tensor([[False], [True], [False], [False]])
    last_value = torch.tensor([2.0])
    adv, ret = compute_gae(reward, value, done, last_value, gamma=1.0, lam=1.0)
    # t=1 is terminal: return = r1 = 1 ; t=0: r0 + ret1 ... (lambda=1 -> Monte Carlo returns)
    assert torch.allclose(ret[1], torch.tensor([1.0]))
    assert torch.allclose(ret[0], torch.tensor([2.0]))
    # after the terminal, a fresh episode bootstraps from last_value
    assert torch.allclose(ret[3], torch.tensor([3.0]))
    assert torch.allclose(ret[2], torch.tensor([4.0]))
    assert torch.allclose(adv, ret - value)
    # lambda < 1 shrinks the advantage toward the one-step TD error
    adv2, _ = compute_gae(reward, value, done, last_value, gamma=1.0, lam=0.0)
    assert torch.allclose(adv2[2], torch.tensor([1.0 + 0.5 - 0.5]))


def test_running_scale_tracks_rms():
    rs = RunningScale(ema=0.5, device="cpu")
    rs.update(torch.full((100,), 4.0))
    assert abs(float(rs.scale) - 4.0) < 1e-5
    rs.update(torch.zeros(100))
    assert abs(float(rs.scale) - (8.0) ** 0.5) < 1e-5
    d = rs.state_dict()
    rs2 = RunningScale(0.5, "cpu")
    rs2.load_state_dict(d)
    assert float(rs2.scale) == float(rs.scale)


def test_trainer_runs_on_cpu_and_resumes(tmp_path):
    from tools.fs1 import train as T

    run = tmp_path / "run"
    base = ["--run-dir", str(run), "--cpu", "--n-envs", "8", "--d-model", "32", "--n-layers", "1",
            "--n-heads", "2", "--mlp-ratio", "2", "--d-target", "16", "--horizon", "8", "--minibatch", "32",
            "--grad-accum", "2", "--episode-steps", "40", "--gate-every", "2", "--gate-games", "4",
            "--pool-every", "2", "--ckpt-every", "1", "--keep-every", "2", "--warmup", "1", "--chunk-envs", "1",
            "--opener-max-len", "10"]
    T.main(base + ["--updates", "2"])
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    ups = [e for e in events if "env_steps_s" in e]
    assert [e["update"] for e in ups] == [0, 1]
    assert (run / "latest.pt").exists() and (run / "ckpt_000002.pt").exists()
    assert any(e.get("event") == "gate" for e in events)
    assert (run / "league").exists() and list((run / "league").glob("*.pt"))
    for e in ups:
        assert all(torch.isfinite(torch.tensor(float(e[k]))) for k in ("loss", "pg", "vf", "ent", "kl_old"))
        assert e["n_valid"] > 0
    # resume continues from update 2 without touching the earlier records
    T.main(base + ["--updates", "3"])
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    ups = [e["update"] for e in events if "env_steps_s" in e]
    assert ups == [0, 1, 2]
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["update"] == 2 and ck["league"]["last_best"] is not None


def test_trainer_flags_train_temperature_win_adv_mb_over_valid(tmp_path):
    from tools.fs1 import train as T

    run = tmp_path / "run"
    T.main(["--run-dir", str(run), "--cpu", "--n-envs", "8", "--d-model", "32", "--n-layers", "1",
            "--n-heads", "2", "--mlp-ratio", "2", "--d-target", "16", "--horizon", "8", "--minibatch", "16",
            "--grad-accum", "1", "--episode-steps", "40", "--gate-every", "2", "--gate-games", "4",
            "--pool-every", "2", "--ckpt-every", "1", "--keep-every", "2", "--warmup", "1", "--chunk-envs", "1",
            "--opener-max-len", "0", "--league-frac", "0.5", "--w-win", "1", "--w-win-start", "1", "--w-win-hold", "1",
            "--train-temperature", "0.5", "--win-adv-coef", "0.5", "--mb-over-valid", "--debug-checks", "--updates", "3"])
    ups = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    ups = [e for e in ups if "env_steps_s" in e]
    assert len(ups) == 3
    for e in ups:
        assert all(torch.isfinite(torch.tensor(float(e[k]))) for k in ("loss", "pg", "vf", "vf_win", "ent", "kl_old"))
        # 16 envs-worth of rows x 8 steps, a quarter of them frozen league seats after the first gate
        assert e["n_minibatches"] <= 8 * 16 // 16
        assert "adv_win_std" in e
    assert ups[-1]["n_minibatches"] < 8 * 16 // 16      # league seats exist by now: fewer than T*N / minibatch
