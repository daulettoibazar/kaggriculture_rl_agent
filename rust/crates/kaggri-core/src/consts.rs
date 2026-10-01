//! Static game data, ported 1:1 from `kaggriculture.py`.
//!
//! Every table here is a transliteration of a module-level constant in the Python
//! source. Ordering is load-bearing in several places and is preserved exactly:
//!
//! * [`PRODUCTS`] — order of `market["inventory"]` / `market["prices"]` keys and of
//!   `_refresh_prices` iteration.
//! * [`SHED_ITEMS`] — `_new_private()` seeds the shed with `PRODUCTS + list(ANIMALS)`,
//!   in that order, and no code path ever adds or deletes a shed key (verified
//!   empirically over full episodes), so the shed is a fixed 12-slot table.
//! * [`SHOPS_SORTED`] — `rng.choice(sorted(SHOPS))` draws from the *alphabetically
//!   sorted* shop names; the index the RNG produces must map through this array.
//! * [`TOWN_CENTER_PRODUCTS`] — `[p for p in PRODUCTS if p != "FERTILIZER"]`.
//! * [`LAND_ORDER`] / [`LAND_PRICES`] — the n-th `BUY_LAND` unlocks `LAND_ORDER[n]`.

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/// `MARKET_I0` — default market inventory anchor for every product.
pub const MARKET_I0: i64 = 10_000;

/// `PRICE_FLOOR` — `market_price` never returns less than this.
pub const PRICE_FLOOR: i64 = 1;

/// `HINGE_GAIN` — quadratic gain of the `hinge` shape above the knee.
pub const HINGE_GAIN: f64 = 8.0;

/// `MAX_SHOP_INSTANCES` — the town stops unlocking shops past this many instances.
pub const MAX_SHOP_INSTANCES: usize = 8;

/// `FARM_HAND_COST_MULT` — default multiplier on the Fibonacci hire cost.
pub const FARM_HAND_COST_MULT: i64 = 1;

pub const N_PRODUCTS: usize = 9;
pub const N_CROPS: usize = 5;
pub const N_ANIMALS: usize = 3;
pub const N_SHED_ITEMS: usize = 12;
pub const N_SHOPS: usize = 8;

// ---------------------------------------------------------------------------
// Product
// ---------------------------------------------------------------------------

/// `PRODUCTS` as an index. Discriminants are the list positions in Python.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Product {
    Wheat = 0,
    Carrot = 1,
    Tomato = 2,
    Strawberry = 3,
    Melon = 4,
    Egg = 5,
    Milk = 6,
    Wool = 7,
    Fertilizer = 8,
}

/// `PRODUCTS` in Python list order.
pub const PRODUCTS: [Product; N_PRODUCTS] = [
    Product::Wheat,
    Product::Carrot,
    Product::Tomato,
    Product::Strawberry,
    Product::Melon,
    Product::Egg,
    Product::Milk,
    Product::Wool,
    Product::Fertilizer,
];

const PRODUCT_NAMES: [&str; N_PRODUCTS] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];

/// `TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]`.
pub const TOWN_CENTER_PRODUCTS: [Product; 8] = [
    Product::Wheat,
    Product::Carrot,
    Product::Tomato,
    Product::Strawberry,
    Product::Melon,
    Product::Egg,
    Product::Milk,
    Product::Wool,
];

impl Product {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        PRODUCT_NAMES[self as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        PRODUCTS.iter().copied().find(|p| p.as_str() == name)
    }

    #[inline]
    pub const fn from_idx(i: usize) -> Option<Self> {
        if i < N_PRODUCTS {
            Some(PRODUCTS[i])
        } else {
            None
        }
    }

    /// The shed slot this product occupies (`PRODUCTS` come first in the shed).
    #[inline]
    pub const fn shed(self) -> ShedItem {
        SHED_ITEMS[self as usize]
    }

    /// `BUY_PRODUCT` only quotes `WHEAT` and `FERTILIZER`.
    #[inline]
    pub const fn is_buyable(self) -> bool {
        matches!(self, Product::Wheat | Product::Fertilizer)
    }
}

// ---------------------------------------------------------------------------
// Crop
// ---------------------------------------------------------------------------

/// `CROPS` as an index. Discriminants are the Python dict insertion order.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Crop {
    Wheat = 0,
    Carrot = 1,
    Tomato = 2,
    Strawberry = 3,
    Melon = 4,
}

/// `CROPS` keys in Python dict order (this is also `private["seeds"]` key order).
pub const CROP_LIST: [Crop; N_CROPS] = [
    Crop::Wheat,
    Crop::Carrot,
    Crop::Tomato,
    Crop::Strawberry,
    Crop::Melon,
];

const CROP_NAMES: [&str; N_CROPS] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];

/// One row of the Python `CROPS` table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CropData {
    pub seed: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}

/// `CROPS`, indexed by [`Crop`].
pub const CROPS: [CropData; N_CROPS] = [
    // WHEAT
    CropData { seed: 10, first_yield_day: 2, max_yield_day: 4, interval: 0, max_yield: 6, ongoing: false },
    // CARROT
    CropData { seed: 20, first_yield_day: 2, max_yield_day: 3, interval: 0, max_yield: 4, ongoing: false },
    // TOMATO
    CropData { seed: 50, first_yield_day: 8, max_yield_day: 8, interval: 1, max_yield: 4, ongoing: true },
    // STRAWBERRY
    CropData { seed: 100, first_yield_day: 10, max_yield_day: 10, interval: 2, max_yield: 4, ongoing: true },
    // MELON
    CropData { seed: 80, first_yield_day: 10, max_yield_day: 12, interval: 0, max_yield: 6, ongoing: false },
];

impl Crop {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        CROP_NAMES[self as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        CROP_LIST.iter().copied().find(|c| c.as_str() == name)
    }

    #[inline]
    pub const fn from_idx(i: usize) -> Option<Self> {
        if i < N_CROPS {
            Some(CROP_LIST[i])
        } else {
            None
        }
    }

    #[inline]
    pub const fn data(self) -> &'static CropData {
        &CROPS[self as usize]
    }

    /// Harvested crops land in the inventory under the crop's own name, which is
    /// also a product name for all five crops.
    #[inline]
    pub const fn product(self) -> Product {
        match self {
            Crop::Wheat => Product::Wheat,
            Crop::Carrot => Product::Carrot,
            Crop::Tomato => Product::Tomato,
            Crop::Strawberry => Product::Strawberry,
            Crop::Melon => Product::Melon,
        }
    }

    #[inline]
    pub const fn shed(self) -> ShedItem {
        self.product().shed()
    }
}

// ---------------------------------------------------------------------------
// Animal
// ---------------------------------------------------------------------------

/// The `kind` of an animal-housing tile.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Structure {
    Coop,
    Pasture,
}

/// `ANIMALS` as an index. Discriminants are the Python dict insertion order.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Animal {
    Goose = 0,
    Cow = 1,
    Sheep = 2,
}

/// `list(ANIMALS)` in Python dict order (also the shed tail order).
pub const ANIMAL_LIST: [Animal; N_ANIMALS] = [Animal::Goose, Animal::Cow, Animal::Sheep];

const ANIMAL_NAMES: [&str; N_ANIMALS] = ["GOOSE", "COW", "SHEEP"];

/// One row of the Python `ANIMALS` table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AnimalData {
    pub cost: i64,
    pub structure: Structure,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub product: Product,
}

/// `ANIMALS`, indexed by [`Animal`].
pub const ANIMALS: [AnimalData; N_ANIMALS] = [
    // GOOSE
    AnimalData { cost: 300, structure: Structure::Coop, first_yield_day: 4, interval: 1, max_held: 4, product: Product::Egg },
    // COW
    AnimalData { cost: 400, structure: Structure::Pasture, first_yield_day: 8, interval: 2, max_held: 6, product: Product::Milk },
    // SHEEP
    AnimalData { cost: 500, structure: Structure::Pasture, first_yield_day: 6, interval: 3, max_held: 6, product: Product::Wool },
];

impl Animal {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        ANIMAL_NAMES[self as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        ANIMAL_LIST.iter().copied().find(|a| a.as_str() == name)
    }

    #[inline]
    pub const fn from_idx(i: usize) -> Option<Self> {
        if i < N_ANIMALS {
            Some(ANIMAL_LIST[i])
        } else {
            None
        }
    }

    #[inline]
    pub const fn data(self) -> &'static AnimalData {
        &ANIMALS[self as usize]
    }

    /// Animals bought from the market are stored in the shed under their own name.
    #[inline]
    pub const fn shed(self) -> ShedItem {
        SHED_ITEMS[N_PRODUCTS + self as usize]
    }
}

// ---------------------------------------------------------------------------
// ShedItem — the union of PRODUCTS and ANIMALS, in shed key order
// ---------------------------------------------------------------------------

/// Everything that can sit in the shed or in a unit inventory.
///
/// `_new_private()` builds the shed as `{item: 0 for item in PRODUCTS + list(ANIMALS)}`
/// and nothing ever inserts or deletes a shed key afterwards, so this enum's order is
/// the shed's permanent key order. Unit inventories can only ever contain these same
/// items (`_inv_add` is called with a crop name, an animal product, `"FERTILIZER"`, or
/// a key `PICKUP`ed out of the shed).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ShedItem {
    Wheat = 0,
    Carrot = 1,
    Tomato = 2,
    Strawberry = 3,
    Melon = 4,
    Egg = 5,
    Milk = 6,
    Wool = 7,
    Fertilizer = 8,
    Goose = 9,
    Cow = 10,
    Sheep = 11,
}

/// Shed key order: `PRODUCTS + list(ANIMALS)`.
pub const SHED_ITEMS: [ShedItem; N_SHED_ITEMS] = [
    ShedItem::Wheat,
    ShedItem::Carrot,
    ShedItem::Tomato,
    ShedItem::Strawberry,
    ShedItem::Melon,
    ShedItem::Egg,
    ShedItem::Milk,
    ShedItem::Wool,
    ShedItem::Fertilizer,
    ShedItem::Goose,
    ShedItem::Cow,
    ShedItem::Sheep,
];

impl ShedItem {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        if (self as usize) < N_PRODUCTS {
            PRODUCT_NAMES[self as usize]
        } else {
            ANIMAL_NAMES[self as usize - N_PRODUCTS]
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        SHED_ITEMS.iter().copied().find(|i| i.as_str() == name)
    }

    #[inline]
    pub const fn from_idx(i: usize) -> Option<Self> {
        if i < N_SHED_ITEMS {
            Some(SHED_ITEMS[i])
        } else {
            None
        }
    }

    #[inline]
    pub const fn product(self) -> Option<Product> {
        Product::from_idx(self as usize)
    }

    #[inline]
    pub const fn animal(self) -> Option<Animal> {
        if (self as usize) >= N_PRODUCTS {
            Animal::from_idx(self as usize - N_PRODUCTS)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Shops
// ---------------------------------------------------------------------------

/// `SHOPS` keys in Python dict insertion order.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Shop {
    Bakery = 0,
    PizzaShop = 1,
    BrunchSpot = 2,
    YarnStore = 3,
    IceCreamShop = 4,
    PetCafe = 5,
    SmoothieShop = 6,
    FarmersMarket = 7,
}

/// `list(SHOPS)` — dict insertion order.
pub const SHOP_LIST: [Shop; N_SHOPS] = [
    Shop::Bakery,
    Shop::PizzaShop,
    Shop::BrunchSpot,
    Shop::YarnStore,
    Shop::IceCreamShop,
    Shop::PetCafe,
    Shop::SmoothieShop,
    Shop::FarmersMarket,
];

/// `sorted(SHOPS)` — what `rng.choice` indexes into in `_end_of_day`.
pub const SHOPS_SORTED: [Shop; N_SHOPS] = [
    Shop::Bakery,
    Shop::BrunchSpot,
    Shop::FarmersMarket,
    Shop::IceCreamShop,
    Shop::PetCafe,
    Shop::PizzaShop,
    Shop::SmoothieShop,
    Shop::YarnStore,
];

const SHOP_NAMES: [&str; N_SHOPS] = [
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
];

const BAKERY_P: [Product; 2] = [Product::Egg, Product::Wheat];
const PIZZA_P: [Product; 3] = [Product::Milk, Product::Tomato, Product::Wheat];
const BRUNCH_P: [Product; 3] = [Product::Egg, Product::Wheat, Product::Strawberry];
const YARN_P: [Product; 1] = [Product::Wool];
const ICE_CREAM_P: [Product; 3] = [Product::Strawberry, Product::Milk, Product::Wheat];
const PET_CAFE_P: [Product; 1] = [Product::Carrot];
const SMOOTHIE_P: [Product; 2] = [Product::Strawberry, Product::Milk];
const FARMERS_P: [Product; 4] = [
    Product::Wheat,
    Product::Carrot,
    Product::Tomato,
    Product::Strawberry,
];

impl Shop {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        SHOP_NAMES[self as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        SHOP_LIST.iter().copied().find(|s| s.as_str() == name)
    }

    #[inline]
    pub const fn from_idx(i: usize) -> Option<Self> {
        if i < N_SHOPS {
            Some(SHOP_LIST[i])
        } else {
            None
        }
    }

    /// `SHOPS[name]` — the products this shop instance consumes, in list order.
    #[inline]
    pub const fn products(self) -> &'static [Product] {
        match self {
            Shop::Bakery => &BAKERY_P,
            Shop::PizzaShop => &PIZZA_P,
            Shop::BrunchSpot => &BRUNCH_P,
            Shop::YarnStore => &YARN_P,
            Shop::IceCreamShop => &ICE_CREAM_P,
            Shop::PetCafe => &PET_CAFE_P,
            Shop::SmoothieShop => &SMOOTHIE_P,
            Shop::FarmersMarket => &FARMERS_P,
        }
    }

    /// `multiplier = 2 if len(products) == 1 else 1`.
    #[inline]
    pub const fn multiplier(self) -> i64 {
        if self.products().len() == 1 {
            2
        } else {
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Quadrants / land
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Quadrant {
    NW = 0,
    NE = 1,
    SW = 2,
    SE = 3,
}

const QUADRANT_NAMES: [&str; 4] = ["NW", "NE", "SW", "SE"];

/// `LAND_ORDER` — NW is free; the rest unlock in this order.
pub const LAND_ORDER: [Quadrant; 3] = [Quadrant::NE, Quadrant::SW, Quadrant::SE];

/// `LAND_PRICES`.
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];

impl Quadrant {
    #[inline]
    pub fn as_str(self) -> &'static str {
        QUADRANT_NAMES[self as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "NW" => Some(Quadrant::NW),
            "NE" => Some(Quadrant::NE),
            "SW" => Some(Quadrant::SW),
            "SE" => Some(Quadrant::SE),
            _ => None,
        }
    }
}

/// `_quadrant_of(x, y, board_size)`.
#[inline]
pub fn quadrant_of(x: i64, y: i64, board_size: usize) -> Quadrant {
    let half = (board_size / 2) as i64;
    match (y < half, x < half) {
        (true, true) => Quadrant::NW,
        (true, false) => Quadrant::NE,
        (false, true) => Quadrant::SW,
        (false, false) => Quadrant::SE,
    }
}

/// `_shed_access_tiles(board_size)` — four inner corners, NWSE order.
#[inline]
pub fn shed_access_tiles(board_size: usize) -> [[i64; 2]; 4] {
    let half = (board_size / 2) as i64;
    [
        [half - 1, half - 1],
        [half, half - 1],
        [half - 1, half],
        [half, half],
    ]
}

/// `_is_shed_adjacent(pos, board_size)`.
#[inline]
pub fn is_shed_adjacent(x: i64, y: i64, board_size: usize) -> bool {
    let half = (board_size / 2) as i64;
    (x == half - 1 || x == half) && (y == half - 1 || y == half)
}

/// `_default_spawn(board_size)` — first shed-access tile inside NW.
#[inline]
pub fn default_spawn(board_size: usize) -> [i64; 2] {
    for t in shed_access_tiles(board_size) {
        if quadrant_of(t[0], t[1], board_size) == Quadrant::NW {
            return t;
        }
    }
    [0, 0]
}

// ---------------------------------------------------------------------------
// Movement
// ---------------------------------------------------------------------------

/// `FARMER_MOVES` — (dx, dy); y grows downward.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Dir {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
}

impl Dir {
    #[inline]
    pub const fn delta(self) -> (i64, i64) {
        match self {
            Dir::North => (0, -1),
            Dir::South => (0, 1),
            Dir::East => (1, 0),
            Dir::West => (-1, 0),
        }
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Dir::North => "NORTH",
            Dir::South => "SOUTH",
            Dir::East => "EAST",
            Dir::West => "WEST",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "NORTH" => Some(Dir::North),
            "SOUTH" => Some(Dir::South),
            "EAST" => Some(Dir::East),
            "WEST" => Some(Dir::West),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Market price curve parameters
// ---------------------------------------------------------------------------

/// The `f` in `price(inv) = base ± amp * f(|inv - I0|)`.
///
/// Unknown function names fall through to `return x` in `_shape`, which is exactly
/// what `linear` does, so [`Shape::from_name`] maps anything unrecognised to
/// [`Shape::Linear`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Shape {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Hinge,
}

impl Shape {
    pub fn as_str(self) -> &'static str {
        match self {
            Shape::Linear => "linear",
            Shape::Sq => "sq",
            Shape::Sqrt => "sqrt",
            Shape::Log => "log",
            Shape::Log10 => "log10",
            Shape::Hinge => "hinge",
        }
    }

    /// Mirrors `_shape`'s dispatch: anything that is not one of the six known names
    /// behaves like `linear`.
    pub fn from_name(name: &str) -> Self {
        match name {
            "sq" => Shape::Sq,
            "sqrt" => Shape::Sqrt,
            "log" => Shape::Log,
            "log10" => Shape::Log10,
            "hinge" => Shape::Hinge,
            _ => Shape::Linear,
        }
    }
}

/// One row of `MARKET_PARAMS`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ItemParams {
    pub base: f64,
    pub i0: i64,
    pub t: f64,
    pub below_func: Shape,
    pub below_target: f64,
    pub above_func: Shape,
    pub above_target: f64,
}

/// `MARKET_PARAMS` (or a `_resolve_market_params` result), indexed by [`Product`].
#[derive(Clone, Debug, PartialEq)]
pub struct MarketParams {
    pub items: [ItemParams; N_PRODUCTS],
}

impl MarketParams {
    #[inline]
    pub fn get(&self, item: Product) -> &ItemParams {
        &self.items[item as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, item: Product) -> &mut ItemParams {
        &mut self.items[item as usize]
    }
}

/// `MARKET_PARAMS`.
pub const MARKET_PARAMS: MarketParams = MarketParams {
    items: [
        // WHEAT
        ItemParams { base: 25.0, i0: MARKET_I0, t: 400.0, below_func: Shape::Sqrt, below_target: 0.80, above_func: Shape::Log, above_target: 0.20 },
        // CARROT
        ItemParams { base: 35.0, i0: MARKET_I0, t: 450.0, below_func: Shape::Hinge, below_target: 1.00, above_func: Shape::Sqrt, above_target: 0.70 },
        // TOMATO
        ItemParams { base: 60.0, i0: MARKET_I0, t: 200.0, below_func: Shape::Hinge, below_target: 0.40, above_func: Shape::Sqrt, above_target: 0.60 },
        // STRAWBERRY
        ItemParams { base: 120.0, i0: MARKET_I0, t: 100.0, below_func: Shape::Sqrt, below_target: 0.70, above_func: Shape::Linear, above_target: 1.60 },
        // MELON
        ItemParams { base: 250.0, i0: MARKET_I0, t: 300.0, below_func: Shape::Log, below_target: 0.20, above_func: Shape::Sq, above_target: 3.60 },
        // EGG
        ItemParams { base: 50.0, i0: MARKET_I0, t: 332.0, below_func: Shape::Hinge, below_target: 0.40, above_func: Shape::Log, above_target: 0.20 },
        // MILK
        ItemParams { base: 160.0, i0: MARKET_I0, t: 122.0, below_func: Shape::Sqrt, below_target: 0.60, above_func: Shape::Linear, above_target: 1.60 },
        // WOOL
        ItemParams { base: 200.0, i0: MARKET_I0, t: 105.0, below_func: Shape::Log, below_target: 0.20, above_func: Shape::Sq, above_target: 3.20 },
        // FERTILIZER
        ItemParams { base: 100.0, i0: MARKET_I0, t: 200.0, below_func: Shape::Linear, below_target: 0.40, above_func: Shape::Linear, above_target: 0.40 },
    ],
};

impl Default for MarketParams {
    #[inline]
    fn default() -> Self {
        MARKET_PARAMS
    }
}

/// A sparse per-product override, i.e. one value of `configuration["marketParams"]`.
///
/// `None` means "key absent", which `_resolve_market_params` leaves at the default.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ItemParamsPatch {
    pub base: Option<f64>,
    pub i0: Option<i64>,
    pub t: Option<f64>,
    pub below_func: Option<Shape>,
    pub below_target: Option<f64>,
    pub above_func: Option<Shape>,
    pub above_target: Option<f64>,
}

impl MarketParams {
    /// `_resolve_market_params(overrides)` — merge sparse per-product patches onto the
    /// [`MARKET_PARAMS`] defaults.
    ///
    /// Python silently ignores unknown product keys and non-dict patch values; there is
    /// no way to express either here, so callers must drop them while converting.
    pub fn resolve(overrides: &[(Product, ItemParamsPatch)]) -> MarketParams {
        let mut resolved = MARKET_PARAMS;
        for (item, patch) in overrides {
            let p = resolved.get_mut(*item);
            if let Some(v) = patch.base {
                p.base = v;
            }
            if let Some(v) = patch.i0 {
                p.i0 = v;
            }
            if let Some(v) = patch.t {
                p.t = v;
            }
            if let Some(v) = patch.below_func {
                p.below_func = v;
            }
            if let Some(v) = patch.below_target {
                p.below_target = v;
            }
            if let Some(v) = patch.above_func {
                p.above_func = v;
            }
            if let Some(v) = patch.above_target {
                p.above_target = v;
            }
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_names_round_trip() {
        for p in PRODUCTS {
            assert_eq!(Product::from_name(p.as_str()), Some(p));
            assert_eq!(p.shed().as_str(), p.as_str());
        }
        assert_eq!(Product::from_name("GOOSE"), None);
    }

    #[test]
    fn shed_key_order_matches_python() {
        // list(_new_private()["shed"].keys()) in kaggle-environments 1.32.7
        let expected = [
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
            "EGG",
            "MILK",
            "WOOL",
            "FERTILIZER",
            "GOOSE",
            "COW",
            "SHEEP",
        ];
        let got: Vec<&str> = SHED_ITEMS.iter().map(|i| i.as_str()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn sorted_shops_is_alphabetical() {
        let names: Vec<&str> = SHOPS_SORTED.iter().map(|s| s.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(
            names,
            vec![
                "BAKERY",
                "BRUNCH_SPOT",
                "FARMERS_MARKET",
                "ICE_CREAM_SHOP",
                "PET_CAFE",
                "PIZZA_SHOP",
                "SMOOTHIE_SHOP",
                "YARN_STORE"
            ]
        );
    }

    #[test]
    fn shed_access_and_spawn() {
        assert_eq!(shed_access_tiles(10), [[4, 4], [5, 4], [4, 5], [5, 5]]);
        assert_eq!(default_spawn(10), [4, 4]);
        assert!(is_shed_adjacent(4, 4, 10));
        assert!(is_shed_adjacent(5, 5, 10));
        assert!(!is_shed_adjacent(3, 4, 10));
        assert_eq!(quadrant_of(0, 0, 10), Quadrant::NW);
        assert_eq!(quadrant_of(9, 0, 10), Quadrant::NE);
        assert_eq!(quadrant_of(0, 9, 10), Quadrant::SW);
        assert_eq!(quadrant_of(9, 9, 10), Quadrant::SE);
    }

    #[test]
    fn shop_multipliers() {
        assert_eq!(Shop::YarnStore.multiplier(), 2);
        assert_eq!(Shop::PetCafe.multiplier(), 2);
        assert_eq!(Shop::Bakery.multiplier(), 1);
        assert_eq!(Shop::FarmersMarket.products().len(), 4);
    }

    #[test]
    fn market_params_resolve_is_sparse() {
        let p = MarketParams::resolve(&[(
            Product::Wool,
            ItemParamsPatch {
                above_target: Some(0.95),
                ..Default::default()
            },
        )]);
        assert_eq!(p.get(Product::Wool).above_target, 0.95);
        // everything else untouched
        assert_eq!(p.get(Product::Wool).base, 200.0);
        assert_eq!(p.get(Product::Wool).above_func, Shape::Sq);
        assert_eq!(p.get(Product::Melon), MARKET_PARAMS.get(Product::Melon));
        assert_eq!(MarketParams::resolve(&[]), MARKET_PARAMS);
    }

    #[test]
    fn town_center_excludes_fertilizer() {
        assert!(!TOWN_CENTER_PRODUCTS.contains(&Product::Fertilizer));
        assert_eq!(TOWN_CENTER_PRODUCTS.len(), N_PRODUCTS - 1);
    }
}
