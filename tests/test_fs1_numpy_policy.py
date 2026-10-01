"""The bundle's numpy forward reproduces the torch policy: same greedy actions on real states,
logits within float tolerance, and the price-history twin matches the env wrapper."""
import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.export import policy_arrays
from src.fs1.model import Fs1Config, Fs1Policy
from test_fs1_model import real_batch  # tests/ is on sys.path via conftest rootdir

pytestmark = pytest.mark.kaggri


def test_numpy_policy_matches_torch_greedy_actions():
    import sys
    sys.path.insert(0, "submission")
    from fs1_numpy_policy import NumpyFs1Policy

    torch.manual_seed(3)
    m = Fs1Policy(Fs1Config(d_model=64, n_layers=2, n_heads=4, mlp_ratio=2, d_target=16, hist_k=4)).eval()
    aug, um, tm, _ = real_batch(n_envs=8, steps=25)
    # give the market head some non-trivial logits so argmax is not always category 0
    with torch.no_grad():
        m.market_head[-1].bias.zero_()
    npol = NumpyFs1Policy(policy_arrays(m, dtype=np.float32))
    a_t, *_ = m.act(aug, um, tm, None, greedy=True)
    a_n = npol.act(aug.float().numpy(), um.numpy(), tm.numpy(), temperature=0.0)
    assert a_n.shape == a_t.shape
    agree = (a_n == a_t.numpy()).mean()
    assert agree > 0.98, agree                       # ties at float precision may flip a few
    # trunk parity to float tolerance
    with torch.no_grad():
        h_t = m.trunk(aug).numpy()
    h_n = npol.trunk(aug.float().numpy())
    assert np.abs(h_t - h_n).max() < 2e-3, np.abs(h_t - h_n).max()
    # float16 export still agrees on nearly every action
    npol16 = NumpyFs1Policy(policy_arrays(m, dtype=np.float16))
    a_16 = npol16.act(aug.float().numpy(), um.numpy(), tm.numpy())
    assert (a_16 == a_t.numpy()).mean() > 0.95


def test_price_history_twin_matches_env_wrapper():
    import sys
    sys.path.insert(0, "submission")
    from fs1_numpy_policy import PriceHistory
    from src.fs1.env import Fs1VecEnv

    n, k = 3, 4
    env = Fs1VecEnv(n, [1, 2, 3], "cpu", hist_k=k, episode_steps=60, stagger=False)
    twins = [[PriceHistory(k) for _ in range(2)] for _ in range(n)]
    # the twin sees the (f16-quantised) observation the wrapper published; its deltas therefore
    # differ from the wrapper's f32-based ones by at most f16 rounding of a value near 1.0
    raw = lambda: env.obs_aug[..., :C.N_FEATURES].float().numpy()  # noqa: E731
    for i in range(n):
        for p in range(2):
            twins[i][p].augment(raw()[i, p].reshape(-1))
    rng = np.random.default_rng(0)
    for t in range(30):
        a = torch.zeros(n, 2, C.PLAYER_ACTION_DIM, dtype=torch.long)
        a[:, :, 2 * C.MAX_UNITS + 2] = torch.as_tensor(rng.integers(0, 4, (n, 2)))   # random wheat-seed buys
        a[:, :, 2 * C.MAX_UNITS + 12] = torch.as_tensor(rng.integers(0, 5, (n, 2)))  # random wheat sell fracs
        env.step(a)
        for i in range(n):
            for p in range(2):
                aug_twin = twins[i][p].augment(raw()[i, p].reshape(-1))
                aug_env = env.obs_aug[i, p].float().numpy()
                assert np.allclose(aug_twin[:, C.N_FEATURES:], aug_env[:, C.N_FEATURES:], atol=4e-3), (t, i, p)
