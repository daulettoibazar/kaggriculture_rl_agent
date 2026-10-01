"""Token layout, slot indices and game economics for fs1.

Everything here mirrors `rust/OBS_LAYOUT.md` (normative) and `docs/README.md`. It is
deliberately free of a `kaggri` import so the training parent process can import it before
forking; `tests/test_fs1_constants.py` asserts the values against the module constants and
`kaggri.default_config()`.
"""
from __future__ import annotations

# --- observation ------------------------------------------------------------------------
N_TOKENS = 140
N_FEATURES = 48
OBS_DIM = N_TOKENS * N_FEATURES            # 6720
HDR = 8                                    # payload slot s lives at column HDR + s

TOK_GLOBAL = 0
TOK_TILE0, TOK_TILE1 = 1, 101              # 100 tiles, t = 1 + y*10 + x
TOK_UNIT0, TOK_UNIT1 = 101, 118            # farmer + 16 hands
TOK_PRODUCT0, TOK_PRODUCT1 = 118, 127      # 9 products
TOK_OPP0, TOK_OPP1 = 127, 135              # 8 opponent summary tokens
TOK_PRIVATE = 135
TOK_SCRATCH0, TOK_SCRATCH1 = 136, 140
TOK_MARKET_READ = 139                      # the layout's intended market-head register
TOK_VALUE_READ = 0

# token classes in header one-hot order (slots 0..6)
CLASS_GLOBAL, CLASS_TILE, CLASS_UNIT, CLASS_PRODUCT, CLASS_OPP, CLASS_PRIVATE, CLASS_SCRATCH = range(7)
N_CLASSES = 7

# GLOBAL payload slots
G_STEP, G_DAY, G_HOUR, G_STEPS_LEFT = 0, 1, 2, 3
G_MONEY_LIN, G_MONEY_LOG, G_OPP_MONEY_LIN = 7, 8, 9
G_UNLOCKED_QUADRANTS = 15                  # /4

# TILE payload slots
T_KIND0 = 0                                # 7 kinds: empty, LOCKED, WEED, COOP, PASTURE, PLANT, structure
T_KIND_EMPTY, T_KIND_LOCKED, T_KIND_WEED, T_KIND_COOP, T_KIND_PASTURE, T_KIND_PLANT, T_KIND_STRUCT = range(7)
T_CROP0 = 7                                # 5 crops one-hot (PLANT only)
T_ANIMAL0 = 12                             # 3 animals one-hot (structure only)
T_AGE, T_YIELD, T_AT_CAP, T_HARVESTABLE = 20, 21, 22, 23
T_WATERED, T_DIES_TONIGHT = 25, 27
T_AGE_SCALE, T_YIELD_SCALE = 30.0, 6.0

# UNIT payload slots
U_EXISTS, U_IS_FARMER, U_X, U_Y = 0, 1, 2, 3
U_CARRY0 = 6                               # 12 shed items /32 clip [0,4]
N_SHED_ITEMS = 12
U_CARRY_SCALE = 32.0

# PRODUCT payload slots
P_ONEHOT0 = 0                              # 9 products
P_INV_VS_ANCHOR = 9
P_PRICE_REL, P_PRICE_MOVE, P_PRICE_ABS, P_AT_FLOOR, P_PRICE_LOG = 10, 11, 12, 13, 14
P_HELD, P_OWN_HARVESTABLE, P_OPP_HARVESTABLE = 15, 16, 17
P_DEMAND_DAY, P_DEMAND_TICK = 18, 19
P_DEPTH, P_DUMP_REV, P_BASE, P_SEED_COST, P_ANIMAL_COST = 22, 23, 24, 25, 26
P_PRICE_ABS_SCALE = 300.0
P_BASE_SCALE = 300.0

# PRIVATE payload slots
V_SEEDS0 = 0                               # 5 crops /32 clip [0,4]
V_ANIMALS0 = 5                             # 3 unplaced animals /8 clip [0,4]
V_SHED0 = 11                               # 9 products /shedCapacity
V_SEED_SCALE, V_ANIMAL_SCALE = 32.0, 8.0

# --- action space -----------------------------------------------------------------------
MAX_UNITS = 17
N_TARGETS = 100
N_VERBS = 38
VERB_IDLE = 0
N_MARKET_HEADS = 21
MARKET_HEAD_SIZES = (11, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5)
MARKET_LOGITS = sum(MARKET_HEAD_SIZES)     # 128
PLAYER_ACTION_DIM = 2 * MAX_UNITS + N_MARKET_HEADS   # 55
MASK_DIM = MAX_UNITS * N_VERBS             # 646
TILE_MASK_DIM = N_TARGETS * N_VERBS        # 3800
QTY_BUCKETS = (0, 1, 2, 4, 8, 16, 32)
SELL_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)

# --- economics (docs/README.md, verified against kaggle_environments 1.32.7) -----------
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SHED_ITEMS = PRODUCTS + ANIMALS            # UNIT carry order

SEED_COST = (10.0, 20.0, 50.0, 100.0, 80.0)
FIRST_YIELD_DAY = (2.0, 2.0, 8.0, 10.0, 10.0)
MAX_YIELD_DAY = (4.0, 3.0, 11.0, 16.0, 10.0)
BASE_PRICE = (25.0, 35.0, 60.0, 120.0, 250.0, 50.0, 160.0, 200.0, 100.0)
ANIMAL_COST = (300.0, 400.0, 500.0)
ANIMAL_PRODUCT = (5, 6, 7)                 # GOOSE->EGG, COW->MILK, SHEEP->WOOL (PRODUCTS index)
LAND_PRICES = (1000.0, 2000.0, 4000.0)      # 2nd, 3rd, 4th quadrant
LAND_PAID_CUM = (0.0, 1000.0, 3000.0, 7000.0)   # cumulative spend for 1..4 unlocked quadrants
STARTING_MONEY = 3000.0
SHED_CAPACITY = 100
EPISODE_STEPS = 720
TURNS_PER_DAY = 24
