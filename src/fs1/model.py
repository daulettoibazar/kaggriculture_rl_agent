"""Entity-token transformer actor-critic for fs1.

Input: the simulator's 140 x 48 token observation for one seat, with `hist_k` extra columns
per token (price-delta history, non-zero only on PRODUCT tokens) appended by the env wrapper.
One shared linear stem, a learned per-position embedding (tile position, unit slot, product id
and token class all ride on it), pre-LN transformer blocks, then:

  verb head    per unit token -> 38 logits, hard-masked by the unit legality mask
  target head  q = W_q(unit token + E_verb[verb]);  k = W_k(tile token);  100 logits,
               hard-masked by the tile-verb plausibility plane for that verb
  market head  token 139 -> 128 logits split into the 21 categorical heads, no-op prior at init
  value head   token 0 (+ the opponent seat's token 0 when centralised) -> [total return, win]

`act` samples (verb, then target, then market) and returns the 55-int action row the simulator
consumes; `evaluate` scores stored actions for the PPO update. The joint log-prob of a seat is
the sum over its 17 verb, 17 target and 21 market choices; Idle contributes no target term and
absent units (all-zero mask rows) are forced to Idle with zero log-prob.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C

N_SPEND = 12                               # market heads 0..11: hire, land, 5 seeds, 3 animals, wheat, fertilizer


@dataclass
class Fs1Config:
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 8
    mlp_ratio: int = 4
    d_target: int = 64
    hist_k: int = 4
    market_noop_bias: float = 8.0          # spend heads (hire, land, seeds, animals, products)
    market_sell_noop_bias: float = 4.0     # the 9 sell heads
    centralised_critic: bool = True
    n_styles: int = 0                      # > 0: learned per-team style vector added to token 0 (replay BC)
    n_extra_in: int = 0                    # appended input columns (rl_last Option B: cha22x's proposal)
    residual: bool = False                 # Option B: FOLLOW/override gate per unit and per market head
    override_bias: float = -8.0            # gate logit at init: P(override) = sigmoid(-8) = 3.4e-4
    open_gates: str = "all"                # "all" | "sell": v1 opens only the 9 sell-head gates (critic's fragility test)

    @property
    def in_dim(self) -> int:
        return C.N_FEATURES + self.hist_k + self.n_extra_in

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over the last axis with `mask == 0` entries excluded.

    Rows whose mask is entirely zero fall back to the unmasked distribution so no NaN can
    arise; callers zero the corresponding log-prob / entropy terms.
    """
    logits = logits.float()
    any_ok = mask.any(dim=-1, keepdim=True)
    m = torch.where(any_ok, mask, torch.ones_like(mask))
    masked = logits.masked_fill(~m, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def _sample(logp: torch.Tensor, greedy: bool, temperature: float) -> torch.Tensor:
    """Sample indices from log-probs over the last axis. Shape-agnostic."""
    if greedy:
        return logp.argmax(dim=-1)
    lp = logp if temperature == 1.0 else F.log_softmax(logp / temperature, dim=-1)
    flat = lp.reshape(-1, lp.shape[-1])
    # Gumbel-max is robust to -inf entries and needs no normalisation pass.
    g = -torch.empty_like(flat).exponential_().log()
    idx = (flat + g).argmax(dim=-1)
    return idx.view(lp.shape[:-1])


def _entropy(logp: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return -(p * logp.masked_fill(logp == float("-inf"), 0.0)).sum(-1)


class Fs1Policy(nn.Module):
    def __init__(self, cfg: Fs1Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stem = nn.Linear(cfg.in_dim, d)
        self.pos = nn.Parameter(torch.zeros(C.N_TOKENS, d))
        self.style_emb = nn.Embedding(cfg.n_styles, d) if cfg.n_styles > 0 else None
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)

        self.verb_head = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, C.N_VERBS))
        self.verb_emb = nn.Embedding(C.N_VERBS, d)
        self.tq = nn.Linear(d, cfg.d_target)
        self.tk = nn.Linear(d, cfg.d_target)
        self.market_head = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, C.MARKET_LOGITS))
        vin = 2 * d if cfg.centralised_critic else d
        self.value_head = nn.Sequential(nn.Linear(vin, d), nn.GELU(approximate="tanh"), nn.Linear(d, 2))
        if cfg.residual:
            self.unit_gate = nn.Linear(d, 1)                      # per unit token: override logit
            self.market_gate = nn.Linear(d, C.N_MARKET_HEADS)     # from the market token: one per head

        # runtime (not config) switch for rl_last's two advantage streams: `act` / `evaluate` return
        # logp as [B, 2] = (step part: units + sell heads, strategy part: the 12 spend heads)
        self.split_logp = False
        # runtime switch (--train-temperature): every categorical head is softmax(logits / logit_temp), in
        # `act` and `evaluate` alike, so PPO trains the tempered policy the bundle plays (export at T = logit_temp)
        self.logit_temp = 1.0
        self.register_buffer("market_offsets", torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(C.MARKET_HEAD_SIZES), 0)[:-1].tolist()), dtype=torch.long),
            persistent=False)
        self.register_buffer("market_sizes", torch.tensor(C.MARKET_HEAD_SIZES, dtype=torch.long), persistent=False)
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.verb_emb.weight, std=0.3)   # must be visible next to a unit token of norm ~sqrt(d)
        if self.style_emb is not None:
            nn.init.zeros_(self.style_emb.weight)         # style 0 / an untrained style = the unconditioned policy
        if self.cfg.residual:
            for g in (self.unit_gate, self.market_gate):
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, self.cfg.override_bias)
        # residual projections scaled down with depth (GPT-2 style)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
        # target head: orthogonal, so no shared tile bias dominates the unit query
        nn.init.orthogonal_(self.tq.weight)
        nn.init.orthogonal_(self.tk.weight)
        # market heads: category 0 is "do nothing" on every head; the 12 spend heads (hire,
        # land, 5 seeds, 3 animals, 2 products) get the large prior, the 9 sell heads a smaller one
        with torch.no_grad():
            last = self.market_head[-1]
            last.bias.zero_()
            for i, off in enumerate(self.market_offsets.tolist()):
                last.bias[off] = self.cfg.market_sell_noop_bias if i >= 12 else self.cfg.market_noop_bias

    # ----------------------------------------------------------------- trunk
    def trunk(self, obs_aug: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        """obs_aug: [B, 140, 48 + hist_k] (any float dtype) -> [B, 140, d]. `style` [B] long picks
        the team style vector added to token 0 (ignored when the model has no styles)."""
        x = self.stem(obs_aug.to(self.stem.weight.dtype)) + self.pos
        if style is not None and self.style_emb is not None:
            x = x.clone()
            x[:, C.TOK_GLOBAL] = x[:, C.TOK_GLOBAL] + self.style_emb(style.long()).to(x.dtype)
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

    def _heads(self, h: torch.Tensor):
        unit_tok = h[:, C.TOK_UNIT0:C.TOK_UNIT1]                 # [B, 17, d]
        tile_tok = h[:, C.TOK_TILE0:C.TOK_TILE1]                 # [B, 100, d]
        verb_logits = self.verb_head(unit_tok)                   # [B, 17, 38]
        tile_k = self.tk(tile_tok)                               # [B, 100, dt]
        market_logits = self.market_head(h[:, C.TOK_MARKET_READ])  # [B, 128]
        if self.logit_temp != 1.0:
            verb_logits, market_logits = verb_logits / self.logit_temp, market_logits / self.logit_temp
        return unit_tok, tile_k, verb_logits, market_logits

    def _target_logits(self, unit_tok, tile_k, verb):
        q = self.tq(unit_tok + self.verb_emb(verb))              # [B, 17, dt]
        return torch.einsum("bud,btd->but", q, tile_k) / (math.sqrt(self.cfg.d_target) * self.logit_temp)

    def value_token(self, h: torch.Tensor) -> torch.Tensor:
        """The token the critic reads, [B, d]; stored per step so the update never re-runs the
        opponent's trunk (the centralised critic sees the opponent's token as a fixed input)."""
        return h[:, C.TOK_VALUE_READ]

    def _value(self, v0: torch.Tensor, other: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.centralised_critic:
            v0 = torch.cat([v0, (v0 if other is None else other).to(v0.dtype)], dim=-1)
        return self.value_head(v0).float()                       # [B, 2]

    @staticmethod
    def _effective_unit_mask(unit_mask: torch.Tensor) -> torch.Tensor:
        m = unit_mask.bool().clone()
        m[..., C.VERB_IDLE] = True                                # Idle always available (absent units too)
        return m

    @staticmethod
    def _target_mask_for(tile_mask: torch.Tensor, verb: torch.Tensor) -> torch.Tensor:
        """tile_mask [B, 100, 38] -> [B, 17, 100] plausibility of each tile for each unit's verb."""
        tmv = tile_mask.bool().transpose(1, 2)                   # [B, 38, 100]
        idx = verb.unsqueeze(-1).expand(-1, -1, C.N_TARGETS)     # [B, 17, 100]
        return tmv.gather(1, idx)

    def _market_logp(self, market_logits: torch.Tensor, market_lock=None):
        """-> list of per-head log-prob tensors [B, size_h]. `market_lock` (bool [21]) forces a
        head to its no-op category 0 (curriculum lock on expensive purchases)."""
        out = []
        for i, (off, size) in enumerate(zip(self.market_offsets.tolist(), C.MARKET_HEAD_SIZES)):
            lg = market_logits[:, off:off + size].float()
            if market_lock is not None and bool(market_lock[i]):
                lg = lg.clone()
                lg[:, 1:] = float("-inf")
            out.append(F.log_softmax(lg, dim=-1))
        return out

    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs_aug, unit_mask, tile_mask, pair_index=None, greedy=False, temperature=1.0,
            market_lock=None, style=None):
        """Returns actions [B, 55] int64, logp [B], entropy [B], value [B, 2], value token [B, d].

        `pair_index[i]` is the row of seat i's opponent (same env, same step) for the
        centralised critic; None means the critic sees only its own token. `market_lock` is an
        optional bool [21] forcing heads to no-op (must be passed identically to `evaluate`).
        """
        h = self.trunk(obs_aug, style)
        actions, lp, ent, lp_strat, v0 = self._sample_branch(h, unit_mask, tile_mask, greedy, temperature, market_lock)
        if self.split_logp:
            lp = torch.stack([lp - lp_strat, lp_strat], -1)
        other = None if pair_index is None else v0[pair_index]
        return actions, lp, ent, self._value(v0, other), v0

    # -------------------------------------------------------------- evaluate
    def evaluate(self, obs_aug, unit_mask, tile_mask, actions, opp_tok=None, market_lock=None, style=None):
        """Score stored actions: logp [B], entropy [B], value [B, 2].

        `opp_tok` is the opponent's stored value token [B, d] (from `act` at rollout time);
        `market_lock` must match what `act` used for these samples."""
        h = self.trunk(obs_aug, style)
        unit_tok, tile_k, verb_logits, market_logits = self._heads(h)
        target = actions[:, :C.MAX_UNITS].long()
        verb = actions[:, C.MAX_UNITS:2 * C.MAX_UNITS].long()
        market = actions[:, 2 * C.MAX_UNITS:].long()

        vmask = self._effective_unit_mask(unit_mask)
        verb_lp = _masked_log_softmax(verb_logits, vmask)
        lp = verb_lp.gather(-1, verb.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = _entropy(verb_lp).sum(-1)

        tmask = self._target_mask_for(tile_mask, verb)
        tgt_lp = _masked_log_softmax(self._target_logits(unit_tok, tile_k, verb), tmask)
        has_target = (verb != C.VERB_IDLE) & tmask.any(-1)
        t_lp = tgt_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        lp = lp + torch.where(has_target, t_lp, torch.zeros_like(t_lp)).sum(-1)
        ent = ent + torch.where(has_target, _entropy(tgt_lp), torch.zeros_like(t_lp)).sum(-1)

        lp_strat = torch.zeros_like(lp)
        for i, head_lp in enumerate(self._market_logp(market_logits, market_lock)):
            li = head_lp.gather(-1, market[:, i:i + 1]).squeeze(-1)
            lp = lp + li
            if i < N_SPEND:
                lp_strat = lp_strat + li
            ent = ent + _entropy(head_lp)
        if self.split_logp:
            lp = torch.stack([lp - lp_strat, lp_strat], -1)
        return lp, ent, self._value(self.value_token(h), opp_tok)

    def imitation_nll(self, obs_aug, unit_mask, tile_mask, actions, unit_label, market_label, market_lock=None,
                      style=None):
        """Per-component negative log-likelihood of labelled actions (imitation of a teacher).

        `actions` [B, 55] are the teacher's macro rows; `unit_label` bool [B, 17] marks the units
        whose (verb, target) is labelled, `market_label` bool [B, 21] the labelled market heads. A
        label that is illegal under the masks (log-prob -inf) is dropped rather than propagated.
        Returns (nll_sum [B, 2], n_components [B, 2]) split into unit heads | market heads so the
        caller can normalise per component and log the two parts."""
        h = self.trunk(obs_aug, style)
        unit_tok, tile_k, verb_logits, market_logits = self._heads(h)
        target = actions[:, :C.MAX_UNITS].long()
        verb = actions[:, C.MAX_UNITS:2 * C.MAX_UNITS].long()
        market = actions[:, 2 * C.MAX_UNITS:].long()
        vmask = self._effective_unit_mask(unit_mask)
        verb_lp = _masked_log_softmax(verb_logits, vmask).gather(-1, verb.unsqueeze(-1)).squeeze(-1)   # [B, 17]
        tmask = self._target_mask_for(tile_mask, verb)
        tgt_lp = _masked_log_softmax(self._target_logits(unit_tok, tile_k, verb), tmask)
        t_lp = tgt_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        has_target = (verb != C.VERB_IDLE) & tmask.any(-1)
        ok_v = unit_label & torch.isfinite(verb_lp)
        ok_t = ok_v & has_target & torch.isfinite(t_lp)
        nll_u = -(torch.where(ok_v, verb_lp, torch.zeros_like(verb_lp)).sum(-1)
                  + torch.where(ok_t, t_lp, torch.zeros_like(t_lp)).sum(-1))
        n_u = ok_v.float().sum(-1) + ok_t.float().sum(-1)
        nll_m = torch.zeros_like(nll_u)
        n_m = torch.zeros_like(n_u)
        for i, head_lp in enumerate(self._market_logp(market_logits, market_lock)):
            m_lp = head_lp.gather(-1, market[:, i:i + 1]).squeeze(-1)
            ok_m = market_label[:, i] & torch.isfinite(m_lp)
            nll_m = nll_m - torch.where(ok_m, m_lp, torch.zeros_like(m_lp))
            n_m = n_m + ok_m.float()
        return torch.stack([nll_u, nll_m], -1), torch.stack([n_u, n_m], -1)      # [B, 2] unit | market

    def forward(self, obs_aug, unit_mask, tile_mask, actions, opp_tok=None, market_lock=None, bc=None, *residual_args):
        """DDP syncs gradients through `forward`, so the training pass goes through here. With
        `bc = (obs, unit_mask, tile_mask, actions, unit_label, market_label)` the same call also
        returns the imitation NLL of that batch (one forward per minibatch keeps DDP's bookkeeping).
        A residual (Option B) model routes to `evaluate_residual(obs, um, tm, actions, gate, opp_tok,
        market_lock, strategy_open)`: here `opp_tok` is the gate, `market_lock` the opp token, `bc`
        the market lock and `residual_args` = (strategy_open,)."""
        if self.cfg.residual:
            return self.evaluate_residual(obs_aug, unit_mask, tile_mask, actions, opp_tok, market_lock, bc,
                                          *residual_args)
        lp, ent, val = self.evaluate(obs_aug, unit_mask, tile_mask, actions, opp_tok, market_lock)
        if bc is None:
            return lp, ent, val
        nll, n = self.imitation_nll(*bc, market_lock=market_lock)
        return lp, ent, val, nll, n

    def load_base(self, state_dict: dict) -> "Fs1Policy":
        """Load an fs1 checkpoint (e.g. u913) into this possibly wider model: the stem's first
        columns take the checkpoint's stem, the appended `n_extra_in` columns stay ZERO (so the
        extra inputs are invisible at init), and the gate / style parameters keep their init."""
        own = self.state_dict()
        sd = dict(state_dict)
        w = sd["stem.weight"]
        if w.shape != own["stem.weight"].shape:
            assert w.shape[0] == own["stem.weight"].shape[0] and w.shape[1] < own["stem.weight"].shape[1], w.shape
            full = torch.zeros_like(own["stem.weight"])
            full[:, :w.shape[1]] = w
            sd["stem.weight"] = full
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.split(".")[0] in ("unit_gate", "market_gate", "style_emb") for k in missing), missing
        return self

    # ------------------------------------------------------------ residual (Option B)
    def _gate_open(self, like: torch.Tensor) -> torch.Tensor:
        """[B, 38] bool: gates the config lets the policy choose (the rest are FOLLOW, no log-prob)."""
        g = torch.ones_like(like, dtype=torch.bool)
        if self.cfg.open_gates == "sell":
            g[:, :C.MAX_UNITS + 12] = False
        else:
            assert self.cfg.open_gates == "all", self.cfg.open_gates
        return g

    def _residual_parts(self, h, unit_mask, tile_mask, verb, target, market, gate, market_lock=None):
        """Log-probs of (gate, override branch) for given choices. gate [B, 38] bool: 17 units then
        21 market heads, True = override. Returns lp [B, 2] = (step part: units + sell heads 12..20,
        strategy part: spend heads 0..11), entropy [B], p_override [B, 38]."""
        unit_tok, tile_k, verb_logits, market_logits = self._heads(h)
        g_logit = torch.cat([self.unit_gate(unit_tok).squeeze(-1),
                             self.market_gate(h[:, C.TOK_MARKET_READ])], dim=-1).float()     # [B, 38]
        live = torch.cat([unit_mask.any(-1), torch.ones_like(g_logit[:, C.MAX_UNITS:], dtype=torch.bool)], -1)
        live = live & self._gate_open(g_logit)
        gate = gate & live
        g_lp = torch.where(gate, F.logsigmoid(g_logit), F.logsigmoid(-g_logit))
        g_lp = torch.where(live, g_lp, torch.zeros_like(g_lp))
        p_ov = torch.sigmoid(g_logit) * live
        g_ent = torch.where(live, F.binary_cross_entropy_with_logits(g_logit, torch.sigmoid(g_logit), reduction="none"),
                            torch.zeros_like(g_logit))

        vmask = self._effective_unit_mask(unit_mask)
        verb_lp_all = _masked_log_softmax(verb_logits, vmask)
        verb_lp = verb_lp_all.gather(-1, verb.unsqueeze(-1)).squeeze(-1)                  # [B, 17]
        tmask = self._target_mask_for(tile_mask, verb)
        tgt_lp_all = _masked_log_softmax(self._target_logits(unit_tok, tile_k, verb), tmask)
        has_target = (verb != C.VERB_IDLE) & tmask.any(-1)
        t_lp = torch.where(has_target, tgt_lp_all.gather(-1, target.unsqueeze(-1)).squeeze(-1), torch.zeros_like(verb_lp))
        g_u = gate[:, :C.MAX_UNITS]
        unit_branch = torch.where(g_u, verb_lp + t_lp, torch.zeros_like(verb_lp))
        unit_ent = torch.where(g_u, _entropy(verb_lp_all), torch.zeros_like(verb_lp))

        m_lp = []
        m_ent = []
        for i, head_lp in enumerate(self._market_logp(market_logits, market_lock)):
            m_lp.append(head_lp.gather(-1, market[:, i:i + 1]).squeeze(-1))
            m_ent.append(_entropy(head_lp))
        m_lp = torch.stack(m_lp, -1)                                                        # [B, 21]
        g_m = gate[:, C.MAX_UNITS:]
        m_branch = torch.where(g_m, m_lp, torch.zeros_like(m_lp))
        m_ent = torch.where(g_m, torch.stack(m_ent, -1), torch.zeros_like(m_lp))

        n_spend = N_SPEND
        step = g_lp[:, :C.MAX_UNITS].sum(-1) + unit_branch.sum(-1) \
            + g_lp[:, C.MAX_UNITS + n_spend:].sum(-1) + m_branch[:, n_spend:].sum(-1)
        strat = g_lp[:, C.MAX_UNITS:C.MAX_UNITS + n_spend].sum(-1) + m_branch[:, :n_spend].sum(-1)
        ent = g_ent.sum(-1) + unit_ent.sum(-1) + m_ent.sum(-1)
        return torch.stack([step, strat], -1), ent, p_ov

    @torch.no_grad()
    def act_residual(self, obs_aug, unit_mask, tile_mask, pair_index=None, greedy=False, temperature=1.0,
                     market_lock=None, strategy_open=None):
        """Option B. Returns actions [B, 55] (the override branch, sampled everywhere; only the
        columns whose gate is True are executed), gate [B, 38] bool (17 units, 21 market heads),
        logp [B, 2] (step | strategy), value [B, 2], value token [B, d].

        `strategy_open` [B] bool: False forces FOLLOW on the 12 spend heads (the day cadence: the
        strategy gates are read at hour 0 only); None = open every turn."""
        h = self.trunk(obs_aug)
        actions, _, _, _, v0 = self._sample_branch(h, unit_mask, tile_mask, greedy, temperature, market_lock)
        unit_tok = h[:, C.TOK_UNIT0:C.TOK_UNIT1]
        g_logit = torch.cat([self.unit_gate(unit_tok).squeeze(-1),
                             self.market_gate(h[:, C.TOK_MARKET_READ])], dim=-1).float()
        if greedy:
            gate = g_logit > 0
        else:
            gate = torch.rand_like(g_logit) < torch.sigmoid(g_logit / temperature)
        gate = gate & torch.cat([unit_mask.any(-1), torch.ones_like(gate[:, C.MAX_UNITS:])], -1) & self._gate_open(gate)
        if strategy_open is not None:
            gate[:, C.MAX_UNITS:C.MAX_UNITS + 12] &= strategy_open.bool()[:, None]
        target = actions[:, :C.MAX_UNITS]
        verb = actions[:, C.MAX_UNITS:2 * C.MAX_UNITS]
        market = actions[:, 2 * C.MAX_UNITS:]
        lp, _, _ = self._residual_parts(h, unit_mask, tile_mask, verb, target, market, gate, market_lock)
        if strategy_open is not None:        # a closed gate is forced, not chosen: no log-prob
            lp[:, 1] = torch.where(strategy_open.bool(), lp[:, 1], torch.zeros_like(lp[:, 1]))
        other = None if pair_index is None else v0[pair_index]
        return actions, gate, lp, self._value(v0, other), v0

    def evaluate_residual(self, obs_aug, unit_mask, tile_mask, actions, gate, opp_tok=None, market_lock=None,
                          strategy_open=None):
        """Score stored Option B choices: logp [B, 2] (step | strategy), entropy [B], value [B, 2],
        p_override [B, 38] (for the FOLLOW anchor)."""
        h = self.trunk(obs_aug)
        target = actions[:, :C.MAX_UNITS].long()
        verb = actions[:, C.MAX_UNITS:2 * C.MAX_UNITS].long()
        market = actions[:, 2 * C.MAX_UNITS:].long()
        lp, ent, p_ov = self._residual_parts(h, unit_mask, tile_mask, verb, target, market, gate.bool(), market_lock)
        if strategy_open is not None:
            lp = torch.stack([lp[:, 0], torch.where(strategy_open.bool(), lp[:, 1], torch.zeros_like(lp[:, 1]))], -1)
        return lp, ent, self._value(self.value_token(h), opp_tok), p_ov

    def _sample_branch(self, h, unit_mask, tile_mask, greedy, temperature, market_lock):
        """The plain policy's sampling on a computed trunk (shared by `act` and `act_residual`)."""
        unit_tok, tile_k, verb_logits, market_logits = self._heads(h)
        B = h.shape[0]
        v0 = self.value_token(h)
        vmask = self._effective_unit_mask(unit_mask)
        verb_lp = _masked_log_softmax(verb_logits, vmask)
        verb = _sample(verb_lp, greedy, temperature)
        lp = verb_lp.gather(-1, verb.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = _entropy(verb_lp).sum(-1)
        tmask = self._target_mask_for(tile_mask, verb)
        tgt_lp = _masked_log_softmax(self._target_logits(unit_tok, tile_k, verb), tmask)
        target = _sample(tgt_lp, greedy, temperature)
        has_target = (verb != C.VERB_IDLE) & tmask.any(-1)
        target = torch.where(has_target, target, torch.zeros_like(target))
        t_lp = tgt_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        lp = lp + torch.where(has_target, t_lp, torch.zeros_like(t_lp)).sum(-1)
        ent = ent + torch.where(has_target, _entropy(tgt_lp), torch.zeros_like(t_lp)).sum(-1)
        market = torch.empty(B, C.N_MARKET_HEADS, dtype=torch.long, device=h.device)
        lp_strat = torch.zeros_like(lp)
        for i, head_lp in enumerate(self._market_logp(market_logits, market_lock)):
            c = _sample(head_lp, greedy, temperature)
            market[:, i] = c
            li = head_lp.gather(-1, c.unsqueeze(-1)).squeeze(-1)
            lp = lp + li
            if i < N_SPEND:
                lp_strat = lp_strat + li
            ent = ent + _entropy(head_lp)
        return torch.cat([target, verb, market], dim=-1), lp, ent, lp_strat, v0

    def fold_style(self, style: int) -> "Fs1Policy":
        """A plain (n_styles = 0) copy with style `style` baked into pos[TOK_GLOBAL], for export and
        for PPO fine-tuning with one fixed style: same outputs as `style=` on every input."""
        cfg = Fs1Config.from_dict({**self.cfg.to_dict(), "n_styles": 0})
        m = Fs1Policy(cfg).to(self.pos.device)
        sd = {k: v for k, v in self.state_dict().items() if not k.startswith("style_emb.")}
        m.load_state_dict(sd)
        if self.style_emb is not None:
            with torch.no_grad():
                m.pos[C.TOK_GLOBAL] += self.style_emb.weight[int(style)].to(m.pos.dtype)
        return m.eval()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
