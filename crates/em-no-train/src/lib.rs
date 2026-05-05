mod train_1d;

pub use train_1d::train_fno1d_on;
mod vis_types;
mod vis_maxwell;

pub use vis_types::MaxwellSampleVis;
pub use vis_maxwell::get_one_maxwell_vis_sample;