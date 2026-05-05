mod heat1d;
mod maxwell1d;

pub use heat1d::{Heat1DConfig, Heat1DDataset};
pub use maxwell1d::{Maxwell1DConfig, Maxwell1DDataset};

use candle_core::{Device, Result as CandleResult, Tensor};

pub trait TimeEvolution1D {
    /// 空间网格点数 H
    fn grid_size(&self) -> usize;
    /// 输入通道数 C_in（例如热方程 1，Maxwell 2 或更多）
    fn in_channels(&self) -> usize;
    /// 输出通道数 C_out
    fn out_channels(&self) -> usize;

    /// 生成 n_samples 个 (input, target) 样本，每个都是 [H, C]
    fn generate_dataset(
        &self,
        n_samples: usize,
        device: &Device,
    ) -> CandleResult<(Vec<Tensor>, Vec<Tensor>)>;
}