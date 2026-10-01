"""Phantom-seat market pressure (src/fs1/phantom.py) and the trailing KL anchor (--ref-refresh-every)."""
import json
import os
import pathlib

import numpy as np
import pytest
import torch

for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

T_EP = 40

_CHA22X = pathlib.Path(__file__).resolve().parents[1] / "dist/tape_v21_cha22x/policy.py"
needs_cha22x = pytest.mark.skipif(not _CHA22X.exists(), reason="cha22 agent not vendored; see README")


def _streams_file(tmp_path, n=3):
    """Synthetic top-team streams: heavy WHEAT sells on every step (a phantom banks nothing, so it never runs out)."""
    streams = []
    for i in range(n):
        market = [[["SELL", "WHEAT", 40 + i]] if t % 2 == 0 else [] for t in range(T_EP)]
        money = [3000.0 + 100.0 * t for t in range(T_EP)]           # the recorded seat earns $100/step
        streams.append({"ep": i, "seat": 1, "team": f"t{i}", "lb_rank": 1 + i, "final": 3000.0 + 100.0 * T_EP,
                        "market": market, "money": money})
    p = tmp_path / "streams.json"
    p.write_text(json.dumps({"since": "test", "min_rank": 20, "streams": streams}))
    return str(p)


def test_phantom_seat_moves_only_the_market(tmp_path):
    import kaggri
    from src.fs1.phantom import PhantomStreams

    env = kaggri.PyVecEnv(2, {"episodeSteps": T_EP}, [11, 12], auto_reset=True, chunk_envs=1,
                          obs_dtype="float32", tile_mask=True)
    env.reset()
    ph = PhantomStreams(_streams_file(tmp_path), envs=[0], seed=0, episode_steps=T_EP)
    seats = ph.seat_array(2)
    assert seats.tolist() == [1, -1]
    env.set_phantom_seats(seats)
    a = np.zeros((2, 2, env.player_action_dim), dtype=np.int64)
    n_orders = 0
    for _ in range(10):
        rows = [[None, None] for _ in range(2)]
        ph.fill(rows, np.asarray(env.env_steps()))
        assert rows[1] == [None, None] and rows[0][0] is None and rows[0][1]["hands"] == []
        n_orders += len(rows[0][1]["market"])
        env.step(a, rows)
    assert n_orders == 5                                     # the synthetic stream sells on even steps only
    inv0 = env.state_dict(0)["market"]["inventory"]["WHEAT"]
    inv1 = env.state_dict(1)["market"]["inventory"]["WHEAT"]
    assert inv0 > inv1 + 100, (inv0, inv1)                 # the stream's sells landed in env 0's market only
    assert np.asarray(env.money)[0].tolist() == [3000.0, 3000.0]   # the phantom banked nothing
    # the virtual opponent: seat 1 of the phantom env gets the recorded money curve, seat 0 and env 1 untouched
    phi_prev = torch.full((2, 2), 5000.0); phi_next = torch.full((2, 2), 5100.0); fm = torch.full((2, 2), 3000.0)
    steps = np.array([7, 7])
    p2, n2, f2 = ph.virtual_opponent(phi_prev, phi_next, fm, np.array([False, False]), steps)
    assert p2[0, 1].item() == 3700.0 and n2[0, 1].item() == 3800.0 and f2[0, 1].item() == 3000.0
    assert p2[0, 0].item() == 5000.0 and torch.equal(p2[1], phi_prev[1]) and torch.equal(n2[1], phi_next[1])
    p3, n3, f3 = ph.virtual_opponent(phi_prev, phi_next, fm, np.array([True, False]), steps)
    assert n3[0, 1].item() == 7000.0 and f3[0, 1].item() == 7000.0   # done: the recorded final
    assert torch.equal(phi_prev, torch.full((2, 2), 5000.0))         # inputs are not modified in place
    # a fresh stream is drawn on auto-reset
    before = ph.cur[0]
    ph.on_done(np.array([True, False]))
    assert ph.cur[0] in range(3) and isinstance(before, int)


@needs_cha22x
def test_trainer_phantom_and_ref_refresh(tmp_path, capsys):
    from tools.fs1 import train as T

    common = ["--cpu", "--n-envs", "8", "--d-model", "32", "--n-layers", "1", "--n-heads", "2", "--mlp-ratio", "2",
              "--d-target", "16", "--horizon", "8", "--minibatch", "16", "--grad-accum", "1", "--episode-steps",
              str(T_EP), "--gate-every", "100", "--gate-games", "4", "--pool-every", "100", "--ckpt-every", "1",
              "--keep-every", "100", "--warmup", "1", "--chunk-envs", "1", "--opener-max-len", "0",
              "--league-frac", "0.5", "--train-temperature", "0.5"]
    run1 = tmp_path / "run1"
    T.main(["--run-dir", str(run1), "--updates", "2"] + common)
    assert (run1 / "latest.pt").exists()
    # run 2: the whole public block becomes phantom envs (no program is ever started), and the KL anchor is
    # refreshed from the live weights every update
    run2 = tmp_path / "run2"
    T.main(["--run-dir", str(run2), "--updates", "3", "--init-from", str(run1 / "latest.pt"), "--init-no-league",
            "--init-no-optimizer", "--ref-policy", str(run1 / "latest.pt"), "--kl-lb-coef", "0.1",
            "--ref-refresh-every", "1", "--public-opponents", "cha22x_bundle", "--public-envs", "2",
            "--phantom-envs", "2", "--phantom-streams", _streams_file(tmp_path)] + common)
    out = capsys.readouterr().out
    assert "phantom seats: 2 envs/rank" in out
    ups = [json.loads(l) for l in (run2 / "events.jsonl").read_text().splitlines()]
    ups = [e for e in ups if "env_steps_s" in e]
    assert len(ups) == 3
    for e in ups:
        assert all(torch.isfinite(torch.tensor(float(e[k]))) for k in ("loss", "pg", "vf", "ent", "kl_old", "kl_lb"))
    assert "win_vs_public" not in ups[-1] or not ups[-1]["win_vs_public"]   # no program block ran
    assert "phantom_money_mean" in ups[-1] and "phantom_margin_mean" in ups[-1]
    # with the anchor refreshed to the live weights every update, the KL to it stays tiny
    assert abs(float(ups[-1]["kl_lb"])) < 0.05
