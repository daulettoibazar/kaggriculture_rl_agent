//! Stable categorical ABI for exact low-level unit commands.
//!
//! Used to decode recorded replays into exact commands.  Category zero is `PASS`, so a zero-filled raw
//! row remains the safest possible action.  Invalid indices also resolve to `PASS`.
//! Existing categories must never be reordered; append if a future replay corpus needs
//! more quantities.

use crate::actions::{MarketOrder, UnitOp};
use crate::consts::{Dir, Product, ANIMAL_LIST, CROP_LIST, PRODUCTS};
use crate::macros::PICKUP_ITEMS;

pub const RAW_ACTION_VERSION: u32 = 1;
pub const RAW_UNIT_ACTIONS: usize = 207;
pub const MARKET_ACTION_VERSION: u32 = 1;
pub const RAW_MARKET_SLOTS: usize = 10;
pub const RAW_MARKET_ACTIONS: usize = 1679;

const PASS: i64 = 0;
const MOVE_LO: i64 = 1;
const SIMPLE_LO: i64 = 5;
const PLANT_LO: i64 = 15;
const PICKUP_LO: i64 = 20;
const ANIMAL_PLACE_LO: i64 = 60;
const PRODUCT_PLACE_LO: i64 = 63;

/// Decode one raw category.  Out-of-range values are deliberately safe no-ops.
#[inline]
pub fn raw_unit_op(index: i64) -> UnitOp {
    match index {
        PASS => UnitOp::Pass,
        // Direction order is part of the ABI, and intentionally matches the Python list.
        1 => UnitOp::Move(Dir::North),
        2 => UnitOp::Move(Dir::South),
        3 => UnitOp::Move(Dir::East),
        4 => UnitOp::Move(Dir::West),
        5 => UnitOp::Drop,
        6 => UnitOp::Water,
        7 => UnitOp::Harvest,
        8 => UnitOp::Fertilize,
        9 => UnitOp::Dig,
        10 => UnitOp::BuildCoop,
        11 => UnitOp::BuildPasture,
        12 => UnitOp::Feed,
        13 => UnitOp::CollectFertilizer,
        14 => UnitOp::Care,
        i if (PLANT_LO..PICKUP_LO).contains(&i) => {
            UnitOp::Plant(CROP_LIST[(i - PLANT_LO) as usize])
        }
        i if (PICKUP_LO..ANIMAL_PLACE_LO).contains(&i) => {
            let j = (i - PICKUP_LO) as usize;
            UnitOp::Pickup {
                item: PICKUP_ITEMS[j / 8],
                n: (j % 8 + 1) as i64,
            }
        }
        i if (ANIMAL_PLACE_LO..PRODUCT_PLACE_LO).contains(&i) => {
            let animal = ANIMAL_LIST[(i - ANIMAL_PLACE_LO) as usize];
            UnitOp::Place { item: animal.shed(), n: 1 }
        }
        i if (PRODUCT_PLACE_LO..RAW_UNIT_ACTIONS as i64).contains(&i) => {
            let j = (i - PRODUCT_PLACE_LO) as usize;
            UnitOp::Place {
                item: PRODUCTS[j / 16].shed(),
                n: (j % 16 + 1) as i64,
            }
        }
        _ => UnitOp::Pass,
    }
}

/// Decode one ordered market category.  Category zero is END/PAD and is handled by the
/// queue decoder; quantity-zero categories deliberately return `None`, matching
/// `_parse_order` while still occupying their queue slot.
#[inline]
pub fn raw_market_order(index: i64) -> Option<MarketOrder> {
    const SEED_LO: i64 = 3;
    const ANIMAL_LO: i64 = SEED_LO + 5 * 101;
    const PRODUCT_LO: i64 = ANIMAL_LO + 3 * 17;
    const SELL_LO: i64 = PRODUCT_LO + 2 * 101;
    match index {
        1 => Some(MarketOrder::Hire),
        2 => Some(MarketOrder::BuyLand),
        i if (SEED_LO..ANIMAL_LO).contains(&i) => {
            let j = (i - SEED_LO) as usize;
            let n = (j % 101) as i64;
            (n > 0).then(|| MarketOrder::BuySeed {
                crop: CROP_LIST[j / 101],
                n,
            })
        }
        i if (ANIMAL_LO..PRODUCT_LO).contains(&i) => {
            let j = (i - ANIMAL_LO) as usize;
            let n = (j % 17) as i64;
            (n > 0).then(|| MarketOrder::BuyAnimal {
                animal: ANIMAL_LIST[j / 17],
                n,
            })
        }
        i if (PRODUCT_LO..SELL_LO).contains(&i) => {
            let j = (i - PRODUCT_LO) as usize;
            let n = (j % 101) as i64;
            let buyable = [Product::Wheat, Product::Fertilizer];
            (n > 0).then(|| MarketOrder::BuyProduct {
                item: buyable[j / 101],
                n,
            })
        }
        i if (SELL_LO..RAW_MARKET_ACTIONS as i64).contains(&i) => {
            let j = (i - SELL_LO) as usize;
            let q = j % 102;
            let n = if q == 101 { 999 } else { q as i64 };
            (n > 0).then(|| MarketOrder::Sell {
                item: PRODUCTS[j / 102],
                n,
            })
        }
        _ => None,
    }
}

const _: () = {
    assert!(MOVE_LO == 1);
    assert!(SIMPLE_LO == 5);
    assert!(PRODUCT_PLACE_LO + 9 * 16 == RAW_UNIT_ACTIONS as i64);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{Animal, Crop, Product, ShedItem};

    #[test]
    fn boundary_categories_are_pinned() {
        assert_eq!(raw_unit_op(0), UnitOp::Pass);
        assert_eq!(raw_unit_op(1), UnitOp::Move(Dir::North));
        assert_eq!(raw_unit_op(4), UnitOp::Move(Dir::West));
        assert_eq!(raw_unit_op(15), UnitOp::Plant(Crop::Wheat));
        assert_eq!(raw_unit_op(19), UnitOp::Plant(Crop::Melon));
        assert_eq!(
            raw_unit_op(20),
            UnitOp::Pickup { item: ShedItem::Wheat, n: 1 }
        );
        assert_eq!(
            raw_unit_op(59),
            UnitOp::Pickup { item: ShedItem::Sheep, n: 8 }
        );
        assert_eq!(
            raw_unit_op(60),
            UnitOp::Place { item: Animal::Goose.shed(), n: 1 }
        );
        assert_eq!(
            raw_unit_op(63),
            UnitOp::Place { item: Product::Wheat.shed(), n: 1 }
        );
        assert_eq!(
            raw_unit_op(206),
            UnitOp::Place { item: Product::Fertilizer.shed(), n: 16 }
        );
        assert_eq!(raw_unit_op(-1), UnitOp::Pass);
        assert_eq!(raw_unit_op(207), UnitOp::Pass);
    }

    #[test]
    fn ordered_market_boundaries_are_pinned() {
        assert_eq!(raw_market_order(0), None);
        assert_eq!(raw_market_order(1), Some(MarketOrder::Hire));
        assert_eq!(raw_market_order(2), Some(MarketOrder::BuyLand));
        assert_eq!(raw_market_order(3), None); // BUY_SEED WHEAT 0 occupies a dead slot
        assert_eq!(
            raw_market_order(10),
            Some(MarketOrder::BuySeed { crop: Crop::Wheat, n: 7 })
        );
        assert_eq!(
            raw_market_order(1678),
            Some(MarketOrder::Sell { item: Product::Fertilizer, n: 999 })
        );
        assert_eq!(raw_market_order(1679), None);
    }
}
