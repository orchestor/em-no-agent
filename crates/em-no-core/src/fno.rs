// crates/em-no-core/src/fno.rs

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FNO1DConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub fourier_modes: usize,
    pub layers: usize,
}

impl Default for FNO1DConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 64,
            fourier_modes: 24,
            layers: 4,
        }
    }
}

pub struct FNO1D {
    in_proj: candle_nn::Linear,
    fourier_layers: Vec<FourierLayer>,
    out_proj1: candle_nn::Linear,
    out_proj2: candle_nn::Linear,
}

struct FourierLayer {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    #[allow(dead_code)]
    modes: usize,
}

impl FourierLayer {
    fn new(in_dim: usize, out_dim: usize, modes: usize, vb: VarBuilder, idx: usize) -> Result<Self> {
        // 使用 vb.pp("name") 创建子 VarBuilder（candle 官方用法）
        let vb = vb.pp(&format!("layer_{}", idx));
        let linear1 = candle_nn::linear(in_dim, out_dim, vb.clone())?;
        let linear2 = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self {
            linear1,
            linear2,
            modes,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 教学版：先不实现真正的 FFT 卷积，只用 MLP + relu 近似
        let x1 = self.linear1.forward(x)?.relu()?;
        let x2 = self.linear2.forward(x)?;
        let out = x1.add(&x2)?;
        Ok(out)
    }
}

impl FNO1D {
    pub fn new(config: &FNO1DConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("fno");
        let in_proj = candle_nn::linear(config.in_channels, config.hidden_channels, vb.pp("in_proj"))?;
        let out_proj1 = candle_nn::linear(config.hidden_channels, config.hidden_channels, vb.pp("out_proj1"))?;
        let out_proj2 = candle_nn::linear(config.hidden_channels, config.out_channels, vb.pp("out_proj2"))?;

        let mut fourier_layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let layer = FourierLayer::new(
                config.hidden_channels,
                config.hidden_channels,
                config.fourier_modes,
                vb.pp(&format!("fourier_layer_{}", i)),
                i,
            )?;
            fourier_layers.push(layer);
        }

        Ok(Self {
            in_proj,
            fourier_layers,
            out_proj1,
            out_proj2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, H, C_in]
        let mut h = self.in_proj.forward(x)?; // [B, H, C_hidden]

        for layer in &self.fourier_layers {
            let delta = layer.forward(&h)?;
            h = h.broadcast_add(&delta)?;
            h = h.relu()?;
        }

        h = self.out_proj1.forward(&h)?;
        h = h.relu()?;
        h = self.out_proj2.forward(&h)?; // [B, H, C_out]
        Ok(h)
    }
}

impl Module for FNO1D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}