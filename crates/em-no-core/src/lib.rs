// crates/em-no-core/src/lib.rs

mod fno;
// 如果 layer.rs / data_1d.rs 还没内容，可以先只 mod fno
// mod layer;
// mod data_1d;

pub use fno::{FNO1D, FNO1DConfig};