#!/usr/bin/env python3
"""Generate full-episode lockstep parity fixtures for the Rust port.

For each scenario this script

  1. builds an episode with `kaggle_environments.make("kaggriculture", ...)`,
  2. drives `kaggriculture.interpreter` **directly** (so deliberately malformed
     actions survive; `env.step` would reject them at the schema and flip the agent
     status to INVALID, ending the episode), emulating the two things the framework
     does around the interpreter: `obs.step = len(env.steps)` and `env.done`,
  3. records every action it fed in, plus a canonical digest of the *entire* state
     after every step.

`tests/sim_parity.rs` replays the recorded actions through `kaggri_core::sim::step`
and asserts the digests match byte for byte, which covers every tile field, both
sheds and seed counts, every unit inventory *including key order*, market inventory
and prices, town shops, money (as raw f64 bits), positions, hires_today, unlocked
quadrants, step/day/hour and the final rewards.

Usage:  python3 tests/gen_sim_fixtures.py [outdir]
"""

import random
import struct
import sys
from os import path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

SHED_ORDER = K.PRODUCTS + list(K.ANIMALS)
CROP_ORDER = list(K.CROPS)
ANIMAL_ORDER = list(K.ANIMALS)
SHOP_ORDER = list(K.SHOPS)
QUAD_ORDER = ["NW", "NE", "SW", "SE"]
DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]

# ---------------------------------------------------------------------------
# canonical digest
# ---------------------------------------------------------------------------


def f64bits(v):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def tile_code(t):
    if t is None:
        return "."
    if t == "LOCKED":
        return "#"
    kind = t.get("kind")
    if kind == "WEED":
        return "W"
    if kind == "PLANT":
        return "P%d:%d:%d:%d:%d:%d:%d" % (
            CROP_ORDER.index(t["crop"]),
            t["planted_day"],
            1 if t["watered_today"] else 0,
            t["consecutive_unwatered"],
            t["yield_units"],
            t["max_lifespan_step"],
            t["fertilized_until_day"],
        )
    if "animal" in t:
        return "A%d:%d:%d:%d:%d:%d:%d:%d" % (
            ANIMAL_ORDER.index(t["animal"]),
            t["placed_day"],
            t["yield_units"],
            t["consecutive_unfed"],
            1 if t["fed_today"] else 0,
            1 if t["cared_today"] else 0,
            1 if t["fertilizer_available"] else 0,
            t.get("pending_care_bonus", 0),
        )
    if kind == "COOP":
        return "C"
    if kind == "PASTURE":
        return "S"
    raise AssertionError("unknown tile %r" % (t,))


def digest(state, step, day, hour, done):
    obs0 = state[0].observation
    market = obs0.market
    town = obs0.town
    out = [
        "S%d" % step,
        "D%d" % day,
        "H%d" % hour,
        "Q%d" % (1 if done else 0),
        "R" + ",".join(f64bits(s.reward if s.reward is not None else 0.0) for s in state),
        "MI" + ",".join(str(market["inventory"][p]) for p in K.PRODUCTS),
        "MP" + ",".join(str(market["prices"][p]) for p in K.PRODUCTS),
        "TS" + ",".join(str(SHOP_ORDER.index(s)) for s in town["unlocked_shops"]),
    ]
    for i, s in enumerate(state):
        farm = obs0.farms[i]
        priv = s.observation.private
        # The Rust port models the shed and the seed bag as fixed-order tables, which
        # is only legal because the environment never inserts or deletes a key. Prove
        # it on every step of every fixture rather than trusting the read.
        assert list(priv["shed"].keys()) == SHED_ORDER, list(priv["shed"].keys())
        assert list(priv["seeds"].keys()) == CROP_ORDER, list(priv["seeds"].keys())
        assert list(market["inventory"].keys()) == K.PRODUCTS
        assert list(market["prices"].keys()) == K.PRODUCTS
        for inv in priv["inventories"]:
            for k, v in inv.items():
                assert k in SHED_ORDER and v > 0, (k, v)
        out.append(
            "F%d M%s P%d,%d H%s Q%s N%d T%s"
            % (
                i,
                f64bits(farm["money"]),
                farm["farmer"][0],
                farm["farmer"][1],
                ";".join("%d,%d" % (p[0], p[1]) for p in farm["hands"]),
                ",".join(str(QUAD_ORDER.index(q)) for q in farm["unlocked_quadrants"]),
                farm["hires_today"],
                ";".join(tile_code(t) for row in farm["tiles"] for t in row),
            )
        )
        out.append(
            "V%d SH%s SD%s IV%s"
            % (
                i,
                ",".join(str(priv["shed"][k]) for k in SHED_ORDER),
                ",".join(str(priv["seeds"][c]) for c in CROP_ORDER),
                "/".join(
                    ",".join("%d=%d" % (SHED_ORDER.index(k), v) for k, v in inv.items())
                    for inv in priv["inventories"]
                ),
            )
        )
    return "|".join(out)


# ---------------------------------------------------------------------------
# action encoding
# ---------------------------------------------------------------------------


def enc_tok(v):
    if v is None:
        return "N"
    if isinstance(v, bool):
        return "B:1" if v else "B:0"
    if isinstance(v, int):
        return "I:%d" % v
    if isinstance(v, float):
        return "F:%r" % v
    if isinstance(v, str):
        assert " " not in v and "," not in v and "|" not in v, v
        return "S:" + v
    return "O"


def enc_action(a):
    if not isinstance(a, list):
        return "X"
    if len(a) == 0:
        return "-"
    return " ".join(enc_tok(t) for t in a)


def enc_actions(lst):
    if not isinstance(lst, list):
        return "X"
    return ",".join(enc_action(a) for a in lst)


# ---------------------------------------------------------------------------
# agents (they only need to produce *interesting* traces, not good play)
# ---------------------------------------------------------------------------

BAD_OPS = ["FOO", "plant", "", "HARVEST_ALL", "PICKUP", "PLACE", "PLANT"]
BAD_ITEMS = ["FOO", "wheat", "", "GOOSE_EGG", "SEED"]


class Agent:
    """Smart-random unit control plus a market order generator.

    `husbandry=True` switches to a strict-priority animal keeper: it buys sheep and
    wheat, builds pastures, places animals, and then FEEDs *and* CAREs the same animal
    on the same day (which is what makes the `pending_care_bonus` ordering in
    `_daily_refresh_animals` observable), harvesting and dumping wool for sale.
    """

    def __init__(self, seed, chaos=0.15, husbandry=False):
        self.rng = random.Random(seed)
        self.chaos = chaos
        self.husbandry = husbandry

    # -- malformed action injection -------------------------------------
    def junk_unit(self):
        r = self.rng.random()
        if r < 0.15:
            return [self.rng.choice(BAD_OPS)]
        if r < 0.3:
            return ["PLANT", self.rng.choice(BAD_ITEMS)]
        if r < 0.45:
            return ["PICKUP", self.rng.choice(BAD_ITEMS + K.PRODUCTS), self.rng.choice([0, -3, 2])]
        if r < 0.6:
            return ["PLACE", self.rng.choice(BAD_ITEMS + ANIMAL_ORDER), self.rng.choice([0, -1, 5])]
        if r < 0.7:
            return []
        if r < 0.8:
            return "NOT_A_LIST"
        if r < 0.9:
            return ["DIG"]
        return [self.rng.choice(DIRS)]

    def junk_order(self):
        r = self.rng.random()
        if r < 0.2:
            return [self.rng.choice(["SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"]),
                    self.rng.choice(BAD_ITEMS), 1]
        if r < 0.4:
            return ["SELL", self.rng.choice(K.PRODUCTS), self.rng.choice([0, -5])]
        if r < 0.5:
            return ["SELL", self.rng.choice(K.PRODUCTS)]
        if r < 0.6:
            return []
        if r < 0.7:
            return ["NOPE", "WHEAT", 1]
        if r < 0.8:
            return ["BUY_PRODUCT", self.rng.choice(K.PRODUCTS), 2]
        if r < 0.9:
            return ["HIRE", "extra"]
        return "NOT_A_LIST"

    # -- unit behaviour -------------------------------------------------
    def husbandry_action(self, obs, farm, priv, pos, inv):
        rng = self.rng
        x, y = pos[0], pos[1]
        bs = len(farm["tiles"])
        tile = farm["tiles"][y][x]
        held_animals = [k for k in inv if k in K.ANIMALS]
        if isinstance(tile, dict) and "animal" in tile:
            if not tile["fed_today"] and inv.get("WHEAT", 0) > 0:
                return ["FEED"]
            if not tile["cared_today"]:
                return ["CARE"]
            if tile["yield_units"] > 0:
                return ["HARVEST"]
            if tile["fertilizer_available"]:
                return ["COLLECT_FERTILIZER"]
            return [rng.choice(DIRS)]
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and "animal" not in tile:
            for a in held_animals:
                if K.ANIMALS[a]["structure"] == tile["kind"]:
                    return ["PLACE", a, 1]
            return [rng.choice(DIRS)]
        if tile is None and held_animals:
            a = held_animals[0]
            return ["BUILD_COOP"] if K.ANIMALS[a]["structure"] == "COOP" else ["BUILD_PASTURE"]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ["DIG"]
        if K._is_shed_adjacent((x, y), bs):
            shed_animals = [k for k in K.ANIMALS if priv["shed"].get(k, 0) > 0]
            if shed_animals and not held_animals:
                return ["PICKUP", shed_animals[0], 1]
            if any(k not in ("WHEAT",) and v > 0 for k, v in inv.items() if k not in K.ANIMALS):
                return ["DROP"]
            if inv.get("WHEAT", 0) < 3 and priv["shed"].get("WHEAT", 0) > 0:
                return ["PICKUP", "WHEAT", 5]
        return [rng.choice(DIRS)]

    def husbandry_market(self, obs, farm, priv):
        rng = self.rng
        orders = []
        shed_total = sum(priv["shed"].values())
        if priv["shed"].get("WOOL", 0) > 0:
            orders.append(["SELL", "WOOL", priv["shed"]["WOOL"]])
        for item in ("MILK", "EGG", "FERTILIZER"):
            if priv["shed"].get(item, 0) > 0:
                orders.append(["SELL", item, priv["shed"][item]])
        if priv["shed"].get("WHEAT", 0) < 12 and shed_total < 80:
            orders.append(["BUY_PRODUCT", "WHEAT", 6])
        if farm["money"] > 3000 and rng.random() < 0.35 and shed_total < 60:
            orders.append(["BUY_ANIMAL", "SHEEP" if rng.random() < 0.7 else "COW", 1])
        if obs.get("hour", 0) < 3 and rng.random() < 0.8:
            orders.append(["HIRE"])
        if len(farm["unlocked_quadrants"]) < 4 and rng.random() < 0.05:
            orders.append(["BUY_LAND"])
        return orders

    def unit_action(self, obs, farm, priv, pos, inv):
        rng = self.rng
        if rng.random() < self.chaos:
            return self.junk_unit()
        if self.husbandry:
            return self.husbandry_action(obs, farm, priv, pos, inv)
        x, y = pos[0], pos[1]
        bs = len(farm["tiles"])
        tile = farm["tiles"][y][x]
        day = obs.get("day", 0)
        held_animals = [k for k in inv if k in K.ANIMALS]
        shed_adj = K._is_shed_adjacent((x, y), bs)

        # Standing on an empty structure with a matching animal in hand -> place it.
        if isinstance(tile, dict) and "animal" not in tile and tile.get("kind") in ("COOP", "PASTURE"):
            for a in held_animals:
                if K.ANIMALS[a]["structure"] == tile["kind"]:
                    return ["PLACE", a, 1]
        if isinstance(tile, dict) and "animal" in tile:
            r = rng.random()
            if r < 0.3 and inv.get("WHEAT", 0) > 0 and not tile["fed_today"]:
                return ["FEED"]
            if r < 0.5 and not tile["cared_today"]:
                return ["CARE"]
            if r < 0.7 and tile["fertilizer_available"]:
                return ["COLLECT_FERTILIZER"]
            if r < 0.9 and tile["yield_units"] > 0:
                return ["HARVEST"]
            return [rng.choice(DIRS)]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            cd = K.CROPS[tile["crop"]]
            if not tile["watered_today"] and rng.random() < 0.8:
                return ["WATER"]
            if tile["yield_units"] > 0:
                # Deliberately also try HARVEST before first_yield_day: one-time crops
                # carry yield_units == 1 from the moment they are planted, so this is
                # the only way to exercise the immature-harvest guard.
                mature = day - tile["planted_day"] >= cd["first_yield_day"]
                if rng.random() < (0.7 if mature else 0.35):
                    return ["HARVEST"]
            if inv.get("FERTILIZER", 0) > 0 and rng.random() < 0.4:
                return ["FERTILIZE"]
            return [rng.choice(DIRS)]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            if rng.random() < 0.7:
                return ["DIG"]
            return [rng.choice(DIRS)]
        if tile is None:
            avail = [c for c, n in priv["seeds"].items() if n > 0]
            r = rng.random()
            if avail and r < 0.45:
                return ["PLANT", rng.choice(avail)]
            if r < 0.55 and held_animals:
                a = held_animals[0]
                return ["BUILD_COOP"] if K.ANIMALS[a]["structure"] == "COOP" else ["BUILD_PASTURE"]
            if r < 0.60:
                return [rng.choice(["BUILD_COOP", "BUILD_PASTURE"])]
            return [rng.choice(DIRS)]
        # LOCKED or anything else
        if shed_adj:
            r = rng.random()
            if r < 0.35 and any(v > 0 for v in inv.values()):
                return ["DROP"]
            if r < 0.75:
                have = [k for k, v in priv["shed"].items() if v > 0]
                if have:
                    return ["PICKUP", rng.choice(have), rng.choice([1, 2, 5])]
            return [rng.choice(DIRS)]
        return [rng.choice(DIRS)]

    def market(self, obs, farm, priv, step, hour):
        rng = self.rng
        if self.husbandry:
            return self.husbandry_market(obs, farm, priv)
        orders = []
        n = rng.choice([0, 1, 1, 2, 3, 4, 12])  # 12 exercises the max-orders cut
        for _ in range(n):
            if rng.random() < self.chaos:
                orders.append(self.junk_order())
                continue
            r = rng.random()
            if r < 0.22:
                orders.append(["BUY_SEED", rng.choice(CROP_ORDER), rng.choice([1, 2, 4])])
            elif r < 0.32:
                orders.append(["BUY_ANIMAL", rng.choice(ANIMAL_ORDER), 1])
            elif r < 0.42:
                orders.append(["BUY_PRODUCT", rng.choice(["WHEAT", "FERTILIZER"]), rng.choice([1, 3, 8])])
            elif r < 0.72:
                have = [k for k, v in priv["shed"].items() if v > 0 and k in K.PRODUCTS]
                if have:
                    item = rng.choice(have)
                    orders.append(["SELL", item, rng.choice([1, 3, priv["shed"][item], 50])])
            elif r < 0.9:
                orders.append(["HIRE"])
            else:
                orders.append(["BUY_LAND"])
        return orders

    def __call__(self, obs, state_i):
        farm = obs.farms[obs.player]
        priv = obs.private
        invs = priv["inventories"]
        farmer = self.unit_action(obs, farm, priv, farm["farmer"], invs[0] if invs else {})
        hands = []
        for h_idx, hpos in enumerate(farm["hands"]):
            inv = invs[h_idx + 1] if h_idx + 1 < len(invs) else {}
            hands.append(self.unit_action(obs, farm, priv, hpos, inv))
        # Occasionally submit an extra phantom hand action (Python counts it towards
        # the atomic-PLANT demand even though the hand does not exist).
        if self.rng.random() < 0.05:
            hands.append(["PLANT", self.rng.choice(CROP_ORDER)])
        if self.rng.random() < 0.02:
            hands = "NOT_A_LIST"
        market = self.market(obs, farm, priv, obs.get("step", 0), obs.get("hour", 0))
        return {"farmer": farmer, "hands": hands, "market": market}



class ScriptAgent:
    """A fully deterministic animal keeper on a fixed daily route.

    Its job is to hit the states a random walker never reaches: an animal that is both
    FED and CARED every single day *and* harvested every day, so scheduled production
    lands unclamped with a non-zero `pending_care_bonus`. That is the only way the
    production-before-care-increment ordering in `_daily_refresh_animals` becomes
    observable. `animals` is [(animal, dx, dy)] relative to the shed tile (4,4).
    """

    def __init__(self, animals):
        self.animals = animals

    def __call__(self, obs, state_i):
        player = obs.player
        farm = obs.farms[player]
        priv = obs.private
        hour = obs.get("hour", 0)
        bs = len(farm["tiles"])
        half = bs // 2
        home = (half - 1, half - 1)
        inv = priv["inventories"][0] if priv["inventories"] else {}
        market = []
        farmer = ["PASS"]

        # 9 hours of route per animal, then the shed chores.
        per = 9
        n = len(self.animals)
        wheat_per_day = n + 2
        if hour == 0:
            market.append(["BUY_PRODUCT", "WHEAT", wheat_per_day])
            # One animal per empty slot, grouped by species.
            need = {}
            for a, dx, dy in self.animals:
                t = self.tile_at(farm, home, dx, dy)
                if not (isinstance(t, dict) and "animal" in t):
                    need[a] = need.get(a, 0) + 1
            for a, k in need.items():
                have = priv["shed"].get(a, 0)
                if k > have:
                    market.append(["BUY_ANIMAL", a, k - have])
            farmer = ["PICKUP", "WHEAT", wheat_per_day]
        elif hour < per * n + 1:
            k = (hour - 1) // per
            phase = (hour - 1) % per
            animal, dx, dy = self.animals[k]
            tx, ty = home[0] + dx, home[1] + dy
            x, y = farm["farmer"][0], farm["farmer"][1]
            tile = self.tile_at(farm, home, dx, dy)
            here = farm["tiles"][y][x]
            if phase == 0:
                slot_filled = isinstance(tile, dict) and "animal" in tile
                farmer = (
                    ["PICKUP", animal, 1]
                    if priv["shed"].get(animal, 0) > 0 and not slot_filled
                    else ["PASS"]
                )
            elif phase == 1:
                farmer = self.step_towards((x, y), (tx, ty))
            elif phase == 2:
                farmer = self.step_towards((x, y), (tx, ty))
                if (x, y) == (tx, ty):
                    if here is None:
                        kind = K.ANIMALS[animal]["structure"]
                        farmer = ["BUILD_COOP"] if kind == "COOP" else ["BUILD_PASTURE"]
                    elif isinstance(here, dict) and here.get("kind") == "WEED":
                        farmer = ["DIG"]
            elif phase == 3:
                farmer = ["PLACE", animal, 1] if inv.get(animal, 0) > 0 else ["PASS"]
            elif phase == 4:
                farmer = ["FEED"] if inv.get("WHEAT", 0) > 0 else ["PASS"]
            elif phase == 5:
                farmer = ["CARE"]
            elif phase == 6:
                farmer = ["HARVEST"]
            elif phase == 7:
                farmer = ["COLLECT_FERTILIZER"]
            else:
                farmer = self.step_towards((x, y), home)
        elif hour == per * n + 1:
            x, y = farm["farmer"][0], farm["farmer"][1]
            farmer = self.step_towards((x, y), home)
        elif hour == per * n + 2:
            farmer = ["DROP"]
        elif hour == per * n + 3:
            for item in ("EGG", "MILK", "WOOL", "FERTILIZER"):
                if priv["shed"].get(item, 0) > 0:
                    market.append(["SELL", item, priv["shed"][item]])
            if priv["shed"].get("WHEAT", 0) > 3 * wheat_per_day:
                market.append(["SELL", "WHEAT", priv["shed"]["WHEAT"] - wheat_per_day])
        return {"farmer": farmer, "hands": [], "market": market}

    @staticmethod
    def tile_at(farm, home, dx, dy):
        bs = len(farm["tiles"])
        tx, ty = home[0] + dx, home[1] + dy
        if 0 <= tx < bs and 0 <= ty < bs:
            return farm["tiles"][ty][tx]
        return None

    @staticmethod
    def step_towards(pos, target):
        x, y = pos
        tx, ty = target
        if x < tx:
            return ["EAST"]
        if x > tx:
            return ["WEST"]
        if y < ty:
            return ["SOUTH"]
        if y > ty:
            return ["NORTH"]
        return ["PASS"]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(name, config, agent_seeds, chaos, outdir, husbandry=(False, False), scripted=None):
    env = make("kaggriculture", configuration=dict(config), debug=True)
    env.reset()
    state = env.state
    cfg = env.configuration
    tpd = max(1, int(K.get(cfg, "turnsPerDay", 24)))
    episode_steps = int(cfg.episodeSteps)
    seed = env.info["seed"]

    if scripted is not None:
        agents = [ScriptAgent(scripted[0]), ScriptAgent(scripted[1])]
    else:
        agents = [
            Agent(agent_seeds[0], chaos[0], husbandry=husbandry[0]),
            Agent(agent_seeds[1], chaos[1], husbandry=husbandry[1]),
        ]

    lines = [
        "# kaggriculture parity fixture: %s" % name,
        "CFG episode_steps=%d board_size=%d starting_money=%d max_orders=%d turns_per_day=%d "
        "shed_capacity=%d weed_spawn_chance=%r town_shop_unlock_interval=%d "
        "town_center_sell_interval=%d town_shop_sell_interval=%d farm_hand_cost_mult=%d seed=%d"
        % (
            episode_steps,
            int(K.get(cfg, "boardSize", 10)),
            int(K.get(cfg, "startingMoney", 3000)),
            int(K.get(cfg, "maxMarketOrdersPerTurn", 10)),
            int(K.get(cfg, "turnsPerDay", 24)),
            int(K.get(cfg, "shedCapacity", 100)),
            float(K.get(cfg, "weedSpawnChance", 0.005)),
            int(K.get(cfg, "townShopUnlockInterval", 3)),
            int(K.get(cfg, "townCenterSellInterval", 24)),
            int(K.get(cfg, "townShopSellInterval", 4)),
            int(K.get(cfg, "farmHandCostMult", 1)),
            seed,
        ),
    ]

    stats = {
        "max_hands": 0,
        "animals_placed": 0,
        "escapes": 0,
        "weeds": 0,
        "quadrants": 1,
        "max_shed": 0,
        "floor_prices": 0,
        "plant_to_weed": 0,
        "max_price": 0,
        "min_price": 10 ** 9,
        "min_inv": 10 ** 9,
        "max_inv": 0,
        "steps": 0,
    }
    prev_grid = [None, None]

    for step in range(episode_steps):
        if state[0].status != "ACTIVE":
            break
        acts = []
        for i in range(2):
            a = agents[i](state[i].observation, state[i])
            acts.append(a)
            state[i].action = a
        day = step // tpd
        state = K.interpreter(state, env)
        env.state = state
        # framework: obs.step = 0 if done else len(env.steps); env.steps grows by one
        # per env.step() call, so after processing step `step` the counter is step + 1.
        done = all(s.status != "ACTIVE" for s in state)
        state[0].observation.step = 0 if done else step + 1
        env.steps.append(state)

        obs0 = state[0].observation
        lines.append("STEP %d" % step)
        for i in range(2):
            lines.append(
                "A%d %s|%s|%s"
                % (
                    i,
                    enc_action(acts[i]["farmer"]),
                    enc_actions(acts[i]["hands"]),
                    enc_actions(acts[i]["market"]),
                )
            )
        lines.append("D " + digest(state, step + 1, obs0.day, obs0.hour, done))

        # coverage stats
        stats["steps"] = step + 1
        n_animals = 0
        for i in range(2):
            farm = obs0.farms[i]
            stats["max_hands"] = max(stats["max_hands"], len(farm["hands"]))
            stats["quadrants"] = max(stats["quadrants"], len(farm["unlocked_quadrants"]))
            stats["max_shed"] = max(stats["max_shed"], sum(state[i].observation.private["shed"].values()))
            grid = [tile_code(t)[0] for row in farm["tiles"] for t in row]
            if prev_grid[i] is not None:
                for a, b in zip(prev_grid[i], grid):
                    if a == "A" and b in ("C", "S"):
                        stats["escapes"] += 1
                    if a == "P" and b == "W":
                        stats["plant_to_weed"] += 1
            prev_grid[i] = grid
            stats["weeds"] += sum(1 for c in grid if c == "W")
            n_animals += sum(1 for c in grid if c == "A")
        stats["animals_placed"] = max(stats["animals_placed"], n_animals)
        prices = [obs0.market["prices"][p] for p in K.PRODUCTS]
        invs = [obs0.market["inventory"][p] for p in K.PRODUCTS]
        stats["floor_prices"] = max(stats["floor_prices"], sum(1 for p in prices if p == 1))
        stats["max_price"] = max(stats["max_price"], max(prices))
        stats["min_price"] = min(stats["min_price"], min(prices))
        stats["min_inv"] = min(stats["min_inv"], min(invs))
        stats["max_inv"] = max(stats["max_inv"], max(invs))
        if done:
            break

    out = path.join(outdir, "sim_trace_%s.txt" % name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    money = [obs0.farms[i]["money"] for i in range(2)]
    print(
        "%-9s steps=%-4d money=%s hands<=%d animals<=%d escapes=%d plant->weed=%d quads<=%d "
        "shed<=%d floorP<=%d price=[%d,%d] inv=[%d,%d] shops=%d -> %s (%.0f KiB)"
        % (
            name,
            stats["steps"],
            money,
            stats["max_hands"],
            stats["animals_placed"],
            stats["escapes"],
            stats["plant_to_weed"],
            stats["quadrants"],
            stats["max_shed"],
            stats["floor_prices"],
            stats["min_price"],
            stats["max_price"],
            stats["min_inv"],
            stats["max_inv"],
            len(obs0.town["unlocked_shops"]),
            path.basename(out),
            path.getsize(out) / 1024.0,
        )
    )


def run_via_env_run(name, config, agents, outdir):
    """Record a trace produced through the *real* framework path (`env.run`).

    Everything else in this file drives `K.interpreter` directly so malformed actions
    survive schema validation. This one goes through `env.step`/`env.run` instead, which
    independently pins the framework bookkeeping the port has to reproduce: `obs.step`
    advancing one per call, `obs.day`/`obs.hour`, when DONE fires, and the final
    rewards. Only well-formed agents can be used here.
    """
    env = make("kaggriculture", configuration=dict(config), debug=True)
    env.run(list(agents))
    cfg = env.configuration
    seed = env.info["seed"]
    lines = [
        "# kaggriculture parity fixture: %s (recorded through env.run%s)" % (name, tuple(agents)),
        "CFG episode_steps=%d board_size=%d starting_money=%d max_orders=%d turns_per_day=%d "
        "shed_capacity=%d weed_spawn_chance=%r town_shop_unlock_interval=%d "
        "town_center_sell_interval=%d town_shop_sell_interval=%d farm_hand_cost_mult=%d seed=%d"
        % (
            int(cfg.episodeSteps),
            int(K.get(cfg, "boardSize", 10)),
            int(K.get(cfg, "startingMoney", 3000)),
            int(K.get(cfg, "maxMarketOrdersPerTurn", 10)),
            int(K.get(cfg, "turnsPerDay", 24)),
            int(K.get(cfg, "shedCapacity", 100)),
            float(K.get(cfg, "weedSpawnChance", 0.005)),
            int(K.get(cfg, "townShopUnlockInterval", 3)),
            int(K.get(cfg, "townCenterSellInterval", 24)),
            int(K.get(cfg, "townShopSellInterval", 4)),
            int(K.get(cfg, "farmHandCostMult", 1)),
            seed,
        ),
    ]
    for k in range(1, len(env.steps)):
        st = env.steps[k]
        obs0 = st[0].observation
        done = all(a.status != "ACTIVE" for a in st)
        lines.append("STEP %d" % (k - 1))
        for i in range(2):
            a = st[i].action if isinstance(st[i].action, dict) else {}
            lines.append(
                "A%d %s|%s|%s"
                % (
                    i,
                    enc_action(a.get("farmer", ["PASS"])),
                    enc_actions(a.get("hands", [])),
                    enc_actions(a.get("market", [])),
                )
            )
        lines.append("D " + digest(st, k, obs0.day, obs0.hour, done))
    out = path.join(outdir, "sim_trace_%s.txt" % name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        "%-9s steps=%-4d rewards=%s statuses=%s -> %s (%.0f KiB)"
        % (
            name,
            len(env.steps) - 1,
            [a.reward for a in env.steps[-1]],
            [a.status for a in env.steps[-1]],
            path.basename(out),
            path.getsize(out) / 1024.0,
        )
    )


SCENARIOS = [
    # name, configuration, agent seeds, chaos levels
    ("default", {"seed": 12345, "episodeSteps": 720}, (1, 2), (0.10, 0.25)),
    ("bigseed", {"seed": 2147483629, "episodeSteps": 720}, (3, 4), (0.15, 0.15)),
    (
        "stress",
        {
            "seed": 999,
            "episodeSteps": 240,
            "turnsPerDay": 8,
            "shedCapacity": 20,
            "weedSpawnChance": 0.25,
            "townShopUnlockInterval": 1,
            "townShopSellInterval": 1,
            "townCenterSellInterval": 3,
            "maxMarketOrdersPerTurn": 3,
            "farmHandCostMult": 2,
            "startingMoney": 50000,
        },
        (5, 6),
        (0.2, 0.2),
    ),
    (
        "rich",
        {"seed": 7, "episodeSteps": 480, "startingMoney": 200000, "weedSpawnChance": 0.02},
        (7, 8),
        (0.05, 0.05),
    ),
    (
        # Town demand every single turn: drags market inventory far below I0 and
        # exercises the runaway hinge branch (carrot/tomato/egg) at large x.
        "drain",
        {
            "seed": 424242,
            "episodeSteps": 720,
            "townShopUnlockInterval": 1,
            "townShopSellInterval": 1,
            "townCenterSellInterval": 1,
            "startingMoney": 100000,
        },
        (11, 12),
        (0.08, 0.08),
    ),
    (
        # Two dedicated animal keepers: FEED and CARE land on the same animal on the
        # same day (exercising pending_care_bonus), and the wool they dump is enough
        # to push the wool price down towards the $1 floor.
        "husbandry",
        {"seed": 31337, "episodeSteps": 720, "startingMoney": 100000, "weedSpawnChance": 0.001},
        (13, 14),
        (0.02, 0.02),
        (True, True),
    ),
    (
        # Deterministic keepers: fed + cared + harvested every day, so scheduled
        # production is unclamped and the care-bonus ordering is observable.
        "scripted",
        {"seed": 555, "episodeSteps": 720, "startingMoney": 20000, "weedSpawnChance": 0.02},
        (0, 0),
        (0.0, 0.0),
        (False, False),
        ([("GOOSE", 0, -1), ("COW", -1, 0)], [("SHEEP", 0, -1), ("GOOSE", -1, -1)]),
    ),
    (
        # Eight sheep kept alive, fed and cared for all season: the wool they dump
        # drives the wool price to the $1 floor, where SELL must stop adding supply.
        "floor",
        {
            "seed": 20250817,
            "episodeSteps": 1440,
            "turnsPerDay": 48,
            "boardSize": 8,
            "startingMoney": 20000,
            "weedSpawnChance": 0.01,
        },
        (0, 0),
        (0.0, 0.0),
        (False, False),
        (
            [("SHEEP", 0, -1), ("SHEEP", -1, 0), ("SHEEP", -1, -1), ("SHEEP", 0, -2)],
            [("SHEEP", 0, -1), ("SHEEP", -1, 0), ("SHEEP", -1, -1), ("SHEEP", -2, 0)],
        ),
    ),
    (
        "tiny",
        {
            "seed": 0,
            "episodeSteps": 96,
            "boardSize": 6,
            "turnsPerDay": 6,
            "shedCapacity": 8,
            "weedSpawnChance": 0.5,
            "maxMarketOrdersPerTurn": 1,
        },
        (9, 10),
        (0.35, 0.35),
    ),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else path.join(path.dirname(path.abspath(__file__)), "data")
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    for scen in SCENARIOS:
        name, config, seeds, chaos = scen[:4]
        husbandry = scen[4] if len(scen) > 4 else (False, False)
        scripted = scen[5] if len(scen) > 5 else None
        if only and name not in only:
            continue
        run(name, config, seeds, chaos, outdir, husbandry, scripted)
    if not only or "envrun" in only:
        run_via_env_run(
            "envrun", {"seed": 4242, "episodeSteps": 720}, ("starter", "starter"), outdir
        )


if __name__ == "__main__":
    main()
