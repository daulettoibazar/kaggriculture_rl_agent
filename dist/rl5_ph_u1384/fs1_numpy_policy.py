"""Numpy-only inference for the fs1 entity-token transformer (no torch in the bundle).

Mirrors `src/fs1/model.py` exactly: shared stem + position embedding, pre-LN blocks with
tanh-GELU MLPs, verb head masked by the unit legality mask, target head masked by the tile-verb
plane for the chosen verb, 21 categorical market heads. Greedy by default; `temperature > 0`
samples. Loads the float16 array dict written by `src/fs1/export.py`, upcast to float32.

Layout constants are duplicated here on purpose so the bundle does not import the repo.
"""
from __future__ import annotations

import numpy as np

N_TOKENS, N_FEATURES = 140, 48
TOK_TILE0, TOK_TILE1 = 1, 101
TOK_UNIT0, TOK_UNIT1 = 101, 118
TOK_PRODUCT0, TOK_PRODUCT1 = 118, 127
TOK_MARKET_READ = 139
HDR = 8
P_PRICE_REL = 10
MAX_UNITS, N_TARGETS, N_VERBS, VERB_IDLE = 17, 100, 38, 0
MARKET_HEAD_SIZES = (11, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5)
MARKET_OFFSETS = tuple(int(x) for x in np.concatenate([[0], np.cumsum(MARKET_HEAD_SIZES)[:-1]]))
PLAYER_ACTION_DIM = 2 * MAX_UNITS + len(MARKET_HEAD_SIZES)


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _masked_argmax_or_sample(logits, mask, rng, temperature):
    """logits [..., K], mask [..., K] bool. All-zero mask rows fall back to unmasked."""
    any_ok = mask.any(-1, keepdims=True)
    m = np.where(any_ok, mask, True)
    z = np.where(m, logits, -np.inf)
    if temperature <= 0:
        return z.argmax(-1)
    p = _softmax(z / temperature)
    flat = p.reshape(-1, p.shape[-1])
    u = rng.random(flat.shape[0])
    cdf = np.cumsum(flat, axis=1)
    idx = (cdf < u[:, None]).sum(1)
    return np.minimum(idx, flat.shape[1] - 1).reshape(p.shape[:-1])


RELAXED_VERBS = (3, 7, 10, 11, 12, 13, 14, 15, 16, 17)   # Fertilize, Feed, Plant x5, PlaceAnimal x3


def relax_unit_mask(unit_mask, tile_mask):
    """Numpy twin of `src/fs1/masks.py` (rl_last F1 fix): Plant/Fertilize/Feed/PlaceAnimal are legal
    for every live unit whenever the tile-verb plane has a target. [.., 17, 38], [.., 100, 38] bool."""
    unit_mask = np.asarray(unit_mask, dtype=bool)
    tile_mask = np.asarray(tile_mask, dtype=bool)
    v = list(RELAXED_VERBS)
    alive = unit_mask.any(-1)
    plausible = tile_mask[..., v].any(-2)
    out = unit_mask.copy()
    out[..., v] = out[..., v] | (alive[..., :, None] & plausible[..., None, :])
    return out


class NumpyFs1Policy:
    def __init__(self, arrays: dict):
        f32 = lambda k: np.asarray(arrays[k], dtype=np.float32)  # noqa: E731
        self.d = int(arrays["cfg_d_model"]); self.L = int(arrays["cfg_n_layers"])
        self.H = int(arrays["cfg_n_heads"]); self.dt = int(arrays["cfg_d_target"])
        self.hist_k = int(arrays["cfg_hist_k"])
        self.stem_w, self.stem_b, self.pos = f32("stem.w").T, f32("stem.b"), f32("pos")
        self.blocks = []
        for i in range(self.L):
            g = lambda n, i=i: f32(f"b{i}.{n}")  # noqa: E731
            self.blocks.append(dict(ln1g=g("ln1.g"), ln1b=g("ln1.b"), qkv_w=g("qkv.w").T, qkv_b=g("qkv.b"),
                                    proj_w=g("proj.w").T, proj_b=g("proj.b"), ln2g=g("ln2.g"), ln2b=g("ln2.b"),
                                    fc1_w=g("fc1.w").T, fc1_b=g("fc1.b"), fc2_w=g("fc2.w").T, fc2_b=g("fc2.b")))
        self.lnf_g, self.lnf_b = f32("lnf.g"), f32("lnf.b")
        self.verb_w1, self.verb_b1 = f32("verb.w1").T, f32("verb.b1")
        self.verb_w2, self.verb_b2 = f32("verb.w2").T, f32("verb.b2")
        self.verb_emb = f32("verb_emb")
        self.tq_w, self.tq_b, self.tk_w, self.tk_b = f32("tq.w").T, f32("tq.b"), f32("tk.w").T, f32("tk.b")
        self.market_w1, self.market_b1 = f32("market.w1").T, f32("market.b1")
        self.market_w2, self.market_b2 = f32("market.w2").T, f32("market.b2")
        self.residual = "ugate.w" in arrays                   # rl_last Option B gates
        if self.residual:
            self.ugate_w, self.ugate_b = f32("ugate.w").T, f32("ugate.b")
            self.mgate_w, self.mgate_b = f32("mgate.w").T, f32("mgate.b")
            self.sell_only = bool(int(arrays.get("cfg_sell_only", 0)))    # open_gates == "sell"

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    # ------------------------------------------------------------------ forward
    def trunk(self, obs_aug):
        """obs_aug [B, 140, 48 + hist_k] float32 -> [B, 140, d]."""
        x = obs_aug @ self.stem_w + self.stem_b + self.pos
        B, T, D = x.shape
        H, dh = self.H, D // self.H
        for blk in self.blocks:
            h = _ln(x, blk["ln1g"], blk["ln1b"])
            qkv = h @ blk["qkv_w"] + blk["qkv_b"]
            q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
            q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            att = _softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh))
            a = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
            x = x + a @ blk["proj_w"] + blk["proj_b"]
            h2 = _ln(x, blk["ln2g"], blk["ln2b"])
            x = x + _gelu_tanh(h2 @ blk["fc1_w"] + blk["fc1_b"]) @ blk["fc2_w"] + blk["fc2_b"]
        return _ln(x, self.lnf_g, self.lnf_b)

    def act(self, obs_aug, unit_mask, tile_mask, temperature=0.0, rng=None):
        """obs_aug [B,140,48+k] float; unit_mask [B,17,38] bool; tile_mask [B,100,38] bool
        -> actions [B, 55] int64 (targets | verbs | market)."""
        rng = rng or np.random.default_rng(0)
        h = self.trunk(np.asarray(obs_aug, dtype=np.float32))
        return self._branch(h, unit_mask, tile_mask, temperature, rng)

    def act_residual(self, obs_aug, unit_mask, tile_mask, strategy_open):
        """Greedy Option B: (actions [B, 55], override [B, 38] bool: 17 units then 21 market heads).
        Twin of `Fs1Policy.act_residual(greedy=True)`: gate = logit > 0, restricted to live units,
        and the 12 strategy heads only where `strategy_open` [B] is True."""
        h = self.trunk(np.asarray(obs_aug, dtype=np.float32))
        actions = self._branch(h, unit_mask, tile_mask, 0.0, np.random.default_rng(0))
        ug = (h[:, TOK_UNIT0:TOK_UNIT1] @ self.ugate_w + self.ugate_b)[..., 0]           # [B, 17]
        mg = h[:, TOK_MARKET_READ] @ self.mgate_w + self.mgate_b                          # [B, 21]
        gate = np.concatenate([ug, mg], axis=1) > 0
        gate[:, :MAX_UNITS] &= np.asarray(unit_mask, dtype=bool).any(-1)
        gate[:, MAX_UNITS:MAX_UNITS + 12] &= np.asarray(strategy_open, dtype=bool)[:, None]
        if self.sell_only:                                   # units and the 12 spend heads: always FOLLOW
            gate[:, :MAX_UNITS + 12] = False
        return actions, gate

    def _branch(self, h, unit_mask, tile_mask, temperature, rng):
        B = h.shape[0]
        unit_tok = h[:, TOK_UNIT0:TOK_UNIT1]
        tile_tok = h[:, TOK_TILE0:TOK_TILE1]
        vm = np.asarray(unit_mask, dtype=bool).copy()
        vm[..., VERB_IDLE] = True
        verb_logits = _gelu_tanh(unit_tok @ self.verb_w1 + self.verb_b1) @ self.verb_w2 + self.verb_b2
        verb = _masked_argmax_or_sample(verb_logits, vm, rng, temperature)                 # [B, 17]
        q = (unit_tok + self.verb_emb[verb]) @ self.tq_w + self.tq_b                        # [B, 17, dt]
        k = tile_tok @ self.tk_w + self.tk_b                                                # [B, 100, dt]
        tgt_logits = np.einsum("bud,btd->but", q, k) / np.sqrt(self.dt)
        tmv = np.asarray(tile_mask, dtype=bool).transpose(0, 2, 1)                          # [B, 38, 100]
        tmask = np.take_along_axis(tmv, verb[:, :, None].repeat(N_TARGETS, axis=2), axis=1)  # [B, 17, 100]
        target = _masked_argmax_or_sample(tgt_logits, tmask, rng, temperature)
        has_target = (verb != VERB_IDLE) & tmask.any(-1)
        target = np.where(has_target, target, 0)
        mk = _gelu_tanh(h[:, TOK_MARKET_READ] @ self.market_w1 + self.market_b1) @ self.market_w2 + self.market_b2
        market = np.zeros((B, len(MARKET_HEAD_SIZES)), dtype=np.int64)
        for i, (off, size) in enumerate(zip(MARKET_OFFSETS, MARKET_HEAD_SIZES)):
            market[:, i] = _masked_argmax_or_sample(mk[:, off:off + size], np.ones((B, size), bool), rng, temperature)
        return np.concatenate([target, verb, market], axis=1).astype(np.int64)


class PriceHistory:
    """The bundle-side twin of `Fs1VecEnv`'s history: last `k` deltas of the base-relative price
    per product, zero at episode start, newest first."""

    def __init__(self, k: int):
        self.k = k
        self.hist = np.zeros((TOK_PRODUCT1 - TOK_PRODUCT0, k), dtype=np.float32)
        self.prev = None

    def augment(self, obs_flat):
        """obs_flat [6720] (any float dtype) -> [140, 48 + k] float32 with the history columns."""
        o = np.asarray(obs_flat, dtype=np.float32).reshape(N_TOKENS, N_FEATURES)
        rel = o[TOK_PRODUCT0:TOK_PRODUCT1, HDR + P_PRICE_REL]
        delta = np.zeros_like(rel) if self.prev is None else rel - self.prev
        if self.k > 0:
            self.hist = np.roll(self.hist, 1, axis=1)
            self.hist[:, 0] = delta
        self.prev = rel.copy()
        aug = np.zeros((N_TOKENS, self.k), dtype=np.float32)
        if self.k > 0:
            aug[TOK_PRODUCT0:TOK_PRODUCT1] = self.hist
        return np.concatenate([o, aug], axis=1)
