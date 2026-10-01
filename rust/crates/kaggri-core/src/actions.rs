//! Action types and the parsing that mirrors `_parse_order` and the op dispatch at
//! the top of `_apply_unit_action`.
//!
//! Everything malformed becomes a silent no-op, exactly as in Python. Two classes of
//! normalisation happen here, and both are behaviour-preserving:
//!
//! 1. **Unknown item / crop / op names collapse to [`UnitOp::Noop`]**. In Python an
//!    unknown name still runs through the dispatch, but every path it can reach ends
//!    in a bare `return`: `PICKUP` finds `shed.get(item, 0) == 0`, `PLACE` finds
//!    `inv.get(item, 0) == 0` (unit inventories can only ever hold shed items),
//!    `PLANT` fails `crop not in CROPS`, and an unknown op falls off the end of the
//!    function. None of them mutate anything.
//!
//! 2. **A market order whose item is invalid for its op parses to `None`**. Python
//!    builds the order state anyway and then hits the `else` branch of the quote loop,
//!    which sets `order_states[player_id] = None` on the first pass — before quoting
//!    or committing anything. The observable effect is identical to never having had
//!    an order in that slot, and the slot itself is preserved (it still counts towards
//!    `max_len`, so `_refresh_prices` still runs for that index).
//!
//! The one place Python is *not* a silent no-op is `int(...)` on a non-numeric
//! argument: `_apply_unit_action` calls `int(action[2])` unguarded, so
//! `["PICKUP", "WHEAT", "abc"]` raises `ValueError` out of the interpreter (only when
//! the unit happens to be shed-adjacent — the conversion sits after that check).
//! [`parse_unit_op`] reports that as `Err(ActionParseError)` rather than inventing a
//! behaviour; `_parse_order` *does* catch `(TypeError, ValueError)` there and returns
//! `None`, which [`parse_market_order`] reproduces.

use crate::consts::*;

// ---------------------------------------------------------------------------
// Raw tokens
// ---------------------------------------------------------------------------

/// A single element of a raw action list, as it arrives from Python.
///
/// The environment receives arbitrary JSON, so an action element can be any Python
/// object. Only the cases the interpreter can distinguish are modelled.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    /// `None`.
    Null,
    /// Anything else (list, dict, object): never equal to an op name, and `int()`
    /// raises `TypeError` on it.
    Other,
}

impl Tok {
    pub fn s(v: &str) -> Tok {
        Tok::Str(v.to_string())
    }

    /// The string value, or `None` for non-strings (Python compares with `==`, and a
    /// non-string is never equal to an op or item name).
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Tok::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Raised where CPython would raise out of the interpreter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ActionParseError {
    /// `int(x)` on a value CPython refuses to convert.
    ///
    /// In `_apply_unit_action` this escapes the interpreter as a `ValueError` /
    /// `TypeError`; there is no defined game behaviour to port, so callers must not
    /// feed such actions to the simulation.
    IntConversion,
}

/// `int(x)` for the subset of Python objects an action can contain.
///
/// `int()` truncates floats towards zero, maps `True`/`False` to 1/0, parses strings
/// (with surrounding whitespace, an optional sign, and `_` digit separators), and
/// raises on `None`, containers, and non-finite floats.
pub fn py_int(tok: &Tok) -> Result<i64, ActionParseError> {
    match tok {
        Tok::Int(i) => Ok(*i),
        Tok::Bool(b) => Ok(*b as i64),
        Tok::Float(f) => {
            if f.is_finite() {
                Ok(f.trunc() as i64)
            } else {
                // int(inf) -> OverflowError, int(nan) -> ValueError.
                Err(ActionParseError::IntConversion)
            }
        }
        Tok::Str(s) => parse_py_int_str(s).ok_or(ActionParseError::IntConversion),
        Tok::Null | Tok::Other => Err(ActionParseError::IntConversion),
    }
}

/// `int(str)` — decimal only, optional sign, `_` allowed between digits.
fn parse_py_int_str(s: &str) -> Option<i64> {
    let t = s.trim_matches(|c: char| c.is_whitespace());
    let (neg, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() {
        return None;
    }
    let mut acc: i64 = 0;
    let mut prev_underscore = true; // a leading '_' is invalid
    for c in digits.chars() {
        if c == '_' {
            if prev_underscore {
                return None;
            }
            prev_underscore = true;
            continue;
        }
        let d = c.to_digit(10)? as i64;
        prev_underscore = false;
        acc = acc.checked_mul(10)?.checked_add(d)?;
    }
    if prev_underscore {
        return None; // trailing '_'
    }
    Some(if neg { -acc } else { acc })
}

// ---------------------------------------------------------------------------
// UnitOp
// ---------------------------------------------------------------------------

/// One farmer/hand action.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnitOp {
    /// Anything malformed, unknown, or naming an item that cannot exist: a silent
    /// no-op that is indistinguishable from `PASS` in every code path.
    Noop,
    /// `PASS`.
    Pass,
    /// `NORTH` / `SOUTH` / `EAST` / `WEST`.
    Move(Dir),
    /// `DROP` — dump the whole inventory into the shed (shed-adjacent only).
    Drop,
    /// `PICKUP <item> [n]` (`n` defaults to 1 and may be non-positive).
    Pickup { item: ShedItem, n: i64 },
    /// `PLACE <item> [n]` — animal placement onto a matching empty structure, or a
    /// shed drop; the sim picks the branch exactly as Python does.
    Place { item: ShedItem, n: i64 },
    /// `PLANT <crop>`.
    Plant(Crop),
    /// `WATER`.
    Water,
    /// `HARVEST`.
    Harvest,
    /// `FERTILIZE`.
    Fertilize,
    /// `DIG`.
    Dig,
    /// `BUILD_COOP`.
    BuildCoop,
    /// `BUILD_PASTURE`.
    BuildPasture,
    /// `FEED`.
    Feed,
    /// `COLLECT_FERTILIZER`.
    CollectFertilizer,
    /// `CARE`.
    Care,
}

impl Default for UnitOp {
    /// The environment's default farmer action is `["PASS"]`.
    fn default() -> Self {
        UnitOp::Pass
    }
}

/// The dispatch at the top of `_apply_unit_action`.
///
/// `tokens.is_empty()` covers both `not isinstance(action, list)` and `not action`.
pub fn parse_unit_op(tokens: &[Tok]) -> Result<UnitOp, ActionParseError> {
    let Some(op) = tokens.first().and_then(Tok::as_str) else {
        // Either the action is not a list, is empty, or its op is not a string —
        // in all three cases nothing matches and the function returns.
        return Ok(UnitOp::Noop);
    };
    if let Some(dir) = Dir::from_name(op) {
        return Ok(UnitOp::Move(dir));
    }
    Ok(match op {
        "PASS" => UnitOp::Pass,
        "DROP" => UnitOp::Drop,
        "PICKUP" => {
            if tokens.len() < 2 {
                UnitOp::Noop
            } else {
                // Python evaluates int(action[2]) *before* looking the item up.
                let n = qty_arg(tokens)?;
                match tokens[1].as_str().and_then(ShedItem::from_name) {
                    Some(item) => UnitOp::Pickup { item, n },
                    None => UnitOp::Noop,
                }
            }
        }
        "PLACE" => {
            if tokens.len() < 2 {
                UnitOp::Noop
            } else {
                let item = tokens[1].as_str().and_then(ShedItem::from_name);
                let n = qty_arg(tokens)?;
                match item {
                    Some(item) => UnitOp::Place { item, n },
                    None => UnitOp::Noop,
                }
            }
        }
        "PLANT" => {
            if tokens.len() < 2 {
                UnitOp::Noop
            } else {
                match tokens[1].as_str().and_then(Crop::from_name) {
                    Some(crop) => UnitOp::Plant(crop),
                    None => UnitOp::Noop,
                }
            }
        }
        "WATER" => UnitOp::Water,
        "HARVEST" => UnitOp::Harvest,
        "FERTILIZE" => UnitOp::Fertilize,
        "DIG" => UnitOp::Dig,
        "BUILD_COOP" => UnitOp::BuildCoop,
        "BUILD_PASTURE" => UnitOp::BuildPasture,
        "FEED" => UnitOp::Feed,
        "COLLECT_FERTILIZER" => UnitOp::CollectFertilizer,
        "CARE" => UnitOp::Care,
        _ => UnitOp::Noop,
    })
}

/// `n = int(action[2]) if len(action) >= 3 else 1`.
#[inline]
fn qty_arg(tokens: &[Tok]) -> Result<i64, ActionParseError> {
    if tokens.len() >= 3 {
        py_int(&tokens[2])
    } else {
        Ok(1)
    }
}

/// [`parse_unit_op`] with the CPython-raises case folded into a no-op.
///
/// Convenient for fuzzing, but be aware that the real environment would abort the
/// episode there instead.
pub fn parse_unit_op_lossy(tokens: &[Tok]) -> UnitOp {
    parse_unit_op(tokens).unwrap_or(UnitOp::Noop)
}

// ---------------------------------------------------------------------------
// MarketOrder
// ---------------------------------------------------------------------------

/// One market queue entry, already validated against the op's item domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MarketOrder {
    /// `HIRE` — atomic, no quantity.
    Hire,
    /// `BUY_LAND` — atomic, no quantity.
    BuyLand,
    /// `SELL <product> <n>`; `n > 0`.
    Sell { item: Product, n: i64 },
    /// `BUY_PRODUCT <WHEAT|FERTILIZER> <n>`; `n > 0`.
    BuyProduct { item: Product, n: i64 },
    /// `BUY_SEED <crop> <n>`; `n > 0`.
    BuySeed { crop: Crop, n: i64 },
    /// `BUY_ANIMAL <animal> <n>`; `n > 0`.
    BuyAnimal { animal: Animal, n: i64 },
}

impl MarketOrder {
    /// `ostate["remaining"]` at parse time (1 for the atomic ops).
    #[inline]
    pub fn quantity(self) -> i64 {
        match self {
            MarketOrder::Hire | MarketOrder::BuyLand => 1,
            MarketOrder::Sell { n, .. }
            | MarketOrder::BuyProduct { n, .. }
            | MarketOrder::BuySeed { n, .. }
            | MarketOrder::BuyAnimal { n, .. } => n,
        }
    }
}

/// `_parse_order(order)`.
///
/// Returns `Ok(None)` for every input Python turns into `None`, plus the
/// item-invalid-for-op cases explained in the module docs. `Err` is impossible here
/// (Python catches the conversion errors); the `Result` is kept for symmetry with
/// [`parse_unit_op`] and to keep the door open for stricter callers.
pub fn parse_market_order(tokens: &[Tok]) -> Result<Option<MarketOrder>, ActionParseError> {
    let Some(op) = tokens.first().and_then(Tok::as_str) else {
        return Ok(None);
    };
    match op {
        "HIRE" => return Ok(Some(MarketOrder::Hire)),
        "BUY_LAND" => return Ok(Some(MarketOrder::BuyLand)),
        "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL" => {}
        _ => return Ok(None),
    }
    if tokens.len() < 3 {
        return Ok(None);
    }
    // try: n = int(order[2]) except (TypeError, ValueError): return None
    let Ok(n) = py_int(&tokens[2]) else {
        return Ok(None);
    };
    if n <= 0 {
        return Ok(None);
    }
    let name = tokens[1].as_str();
    Ok(match op {
        "SELL" => name
            .and_then(Product::from_name)
            .map(|item| MarketOrder::Sell { item, n }),
        "BUY_PRODUCT" => name
            .and_then(Product::from_name)
            .filter(|p| p.is_buyable())
            .map(|item| MarketOrder::BuyProduct { item, n }),
        "BUY_SEED" => name
            .and_then(Crop::from_name)
            .map(|crop| MarketOrder::BuySeed { crop, n }),
        "BUY_ANIMAL" => name
            .and_then(Animal::from_name)
            .map(|animal| MarketOrder::BuyAnimal { animal, n }),
        _ => unreachable!(),
    })
}

/// [`parse_market_order`], discarding the (currently impossible) error case.
pub fn parse_market_order_lossy(tokens: &[Tok]) -> Option<MarketOrder> {
    parse_market_order(tokens).unwrap_or(None)
}

// ---------------------------------------------------------------------------
// PlayerAction
// ---------------------------------------------------------------------------

/// One player's action for one turn.
///
/// `hands` may be longer than the player's actual hand count — extra entries are
/// no-ops for movement/tile ops but **still count towards the atomic-PLANT demand**,
/// exactly as in Python (`unit_actions = [farmer_action, *hands_actions]`).
///
/// `market` keeps one slot per submitted order, including unparseable ones (`None`),
/// because the number of slots drives `_process_market`'s outer loop and therefore how
/// many times `_refresh_prices` runs.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PlayerAction {
    pub farmer: UnitOp,
    pub hands: Vec<UnitOp>,
    pub market: Vec<Option<MarketOrder>>,
}

impl PlayerAction {
    /// `{"farmer": ["PASS"], "hands": [], "market": []}`.
    pub fn pass() -> Self {
        PlayerAction::default()
    }

    /// Parse a whole action from raw token lists.
    ///
    /// `farmer: None` models a missing or non-list `"farmer"` key, which Python turns
    /// into `["PASS"]` when the key is absent and into a no-op when it is present but
    /// not a list. Both are supplied here as `None` → [`UnitOp::Noop`]; pass
    /// `Some(&[Tok::s("PASS")])` for the absent-key case if the distinction matters
    /// (it does not: `Pass` and `Noop` are the same no-op).
    pub fn parse(
        farmer: Option<&[Tok]>,
        hands: &[Vec<Tok>],
        market: &[Vec<Tok>],
    ) -> Result<Self, ActionParseError> {
        let farmer = match farmer {
            Some(t) => parse_unit_op(t)?,
            None => UnitOp::Noop,
        };
        let mut h = Vec::with_capacity(hands.len());
        for a in hands {
            h.push(parse_unit_op(a)?);
        }
        let mut m = Vec::with_capacity(market.len());
        for o in market {
            m.push(parse_market_order(o)?);
        }
        Ok(PlayerAction {
            farmer,
            hands: h,
            market: m,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[&str]) -> Vec<Tok> {
        v.iter().map(|s| Tok::s(s)).collect()
    }

    #[test]
    fn unit_op_dispatch() {
        assert_eq!(parse_unit_op(&toks(&["NORTH"])), Ok(UnitOp::Move(Dir::North)));
        assert_eq!(parse_unit_op(&toks(&["WEST"])), Ok(UnitOp::Move(Dir::West)));
        assert_eq!(parse_unit_op(&toks(&["PASS"])), Ok(UnitOp::Pass));
        assert_eq!(parse_unit_op(&[]), Ok(UnitOp::Noop));
        assert_eq!(parse_unit_op(&toks(&["NOPE"])), Ok(UnitOp::Noop));
        assert_eq!(parse_unit_op(&[Tok::Int(3)]), Ok(UnitOp::Noop));
        assert_eq!(parse_unit_op(&toks(&["PLANT"])), Ok(UnitOp::Noop));
        assert_eq!(parse_unit_op(&toks(&["PLANT", "FOO"])), Ok(UnitOp::Noop));
        assert_eq!(
            parse_unit_op(&toks(&["PLANT", "MELON"])),
            Ok(UnitOp::Plant(Crop::Melon))
        );
        assert_eq!(parse_unit_op(&toks(&["PICKUP"])), Ok(UnitOp::Noop));
        assert_eq!(
            parse_unit_op(&toks(&["PICKUP", "WHEAT"])),
            Ok(UnitOp::Pickup {
                item: ShedItem::Wheat,
                n: 1
            })
        );
        assert_eq!(
            parse_unit_op(&[Tok::s("PICKUP"), Tok::s("WHEAT"), Tok::Int(-4)]),
            Ok(UnitOp::Pickup {
                item: ShedItem::Wheat,
                n: -4
            })
        );
        assert_eq!(
            parse_unit_op(&[Tok::s("PICKUP"), Tok::s("FOO"), Tok::Int(2)]),
            Ok(UnitOp::Noop)
        );
        assert_eq!(
            parse_unit_op(&[Tok::s("PICKUP"), Tok::s("WHEAT"), Tok::s("abc")]),
            Err(ActionParseError::IntConversion)
        );
        assert_eq!(
            parse_unit_op(&[Tok::s("PLACE"), Tok::s("GOOSE"), Tok::Float(2.9)]),
            Ok(UnitOp::Place {
                item: ShedItem::Goose,
                n: 2
            })
        );
    }

    #[test]
    fn market_order_parsing_mirrors_python() {
        // _parse_order(['HIRE']) -> {'type': 'HIRE'}
        assert_eq!(parse_market_order(&toks(&["HIRE"])), Ok(Some(MarketOrder::Hire)));
        assert_eq!(
            parse_market_order(&toks(&["HIRE", "x"])),
            Ok(Some(MarketOrder::Hire))
        );
        assert_eq!(
            parse_market_order(&toks(&["BUY_LAND"])),
            Ok(Some(MarketOrder::BuyLand))
        );
        assert_eq!(parse_market_order(&[]), Ok(None));
        assert_eq!(parse_market_order(&toks(&["NOPE", "WHEAT", "1"])), Ok(None));
        // len(order) < 3 -> None
        assert_eq!(parse_market_order(&toks(&["SELL", "WHEAT"])), Ok(None));
        // int('abc') -> caught -> None
        assert_eq!(parse_market_order(&toks(&["SELL", "WHEAT", "abc"])), Ok(None));
        // n <= 0 -> None
        assert_eq!(
            parse_market_order(&[Tok::s("SELL"), Tok::s("WHEAT"), Tok::Int(0)]),
            Ok(None)
        );
        assert_eq!(
            parse_market_order(&[Tok::s("SELL"), Tok::s("WHEAT"), Tok::Int(-2)]),
            Ok(None)
        );
        // int('2') and int(2.7) and int(True)
        assert_eq!(
            parse_market_order(&toks(&["SELL", "WHEAT", "2"])),
            Ok(Some(MarketOrder::Sell {
                item: Product::Wheat,
                n: 2
            }))
        );
        assert_eq!(
            parse_market_order(&[Tok::s("SELL"), Tok::s("WHEAT"), Tok::Float(2.7)]),
            Ok(Some(MarketOrder::Sell {
                item: Product::Wheat,
                n: 2
            }))
        );
        assert_eq!(
            parse_market_order(&[Tok::s("SELL"), Tok::s("WHEAT"), Tok::Bool(true)]),
            Ok(Some(MarketOrder::Sell {
                item: Product::Wheat,
                n: 1
            }))
        );
        // item invalid for the op -> equivalent to no order
        assert_eq!(parse_market_order(&toks(&["SELL", "GOOSE", "1"])), Ok(None));
        assert_eq!(
            parse_market_order(&toks(&["BUY_PRODUCT", "MELON", "1"])),
            Ok(None)
        );
        assert_eq!(
            parse_market_order(&toks(&["BUY_PRODUCT", "FERTILIZER", "3"])),
            Ok(Some(MarketOrder::BuyProduct {
                item: Product::Fertilizer,
                n: 3
            }))
        );
        assert_eq!(
            parse_market_order(&toks(&["BUY_SEED", "EGG", "1"])),
            Ok(None)
        );
        assert_eq!(
            parse_market_order(&toks(&["BUY_ANIMAL", "SHEEP", "2"])),
            Ok(Some(MarketOrder::BuyAnimal {
                animal: Animal::Sheep,
                n: 2
            }))
        );
    }

    #[test]
    fn py_int_semantics() {
        assert_eq!(py_int(&Tok::Int(7)), Ok(7));
        assert_eq!(py_int(&Tok::Bool(false)), Ok(0));
        assert_eq!(py_int(&Tok::Float(-2.9)), Ok(-2));
        assert_eq!(py_int(&Tok::s("  -12 ")), Ok(-12));
        assert_eq!(py_int(&Tok::s("1_000")), Ok(1000));
        assert_eq!(py_int(&Tok::s("1__0")), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::s("_1")), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::s("1_")), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::s("")), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::s("3.5")), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::Null), Err(ActionParseError::IntConversion));
        assert_eq!(py_int(&Tok::Other), Err(ActionParseError::IntConversion));
        assert_eq!(
            py_int(&Tok::Float(f64::INFINITY)),
            Err(ActionParseError::IntConversion)
        );
    }
}
