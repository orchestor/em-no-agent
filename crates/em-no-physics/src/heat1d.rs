use candle_core::{DType, Device, Result as CandleResult, Tensor};
use rand::Rng;
use crate::TimeEvolution1D;

#[derive(Debug, Clone)]
pub struct Heat1DConfig {
    pub length: f32,      // 空间长度 L
    pub n_grid: usize,    // 空间网格数 H
    pub alpha: f32,       // 导热系数
    pub dt: f32,          // 时间步长
    pub n_steps: usize,   // 时间步数，T = n_steps * dt
}

impl Default for Heat1DConfig {
    fn default() -> Self {
        Self {
            length: 1.0,
            n_grid: 64,
            alpha: 0.01,
            dt: 1e-4,
            n_steps: 2000,
        }
    }
}

pub struct Heat1DDataset {
    pub inputs: Vec<Tensor>,  // 每个样本的 u(x, 0): [H, 1]
    pub targets: Vec<Tensor>, // 每个样本的 u(x, T): [H, 1]
}

impl Heat1DDataset {
    pub fn new(
        n_samples: usize,
        config: &Heat1DConfig,
        device: &Device,
    ) -> CandleResult<Self> {
        let mut inputs = Vec::with_capacity(n_samples);
        let mut targets = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let (u0, u_t) = generate_one_sample(config, device)?;
            inputs.push(u0);
            targets.push(u_t);
        }

        Ok(Self { inputs, targets })
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }
}




impl TimeEvolution1D for Heat1DConfig {
    fn grid_size(&self) -> usize {
        self.n_grid
    }

    fn in_channels(&self) -> usize {
        1
    }

    fn out_channels(&self) -> usize {
        1
    }

    fn generate_dataset(
        &self,
        n_samples: usize,
        device: &Device,
    ) -> CandleResult<(Vec<Tensor>, Vec<Tensor>)> {
        let dataset = Heat1DDataset::new(n_samples, self, device)?;
        Ok((dataset.inputs, dataset.targets))
    }
}

/// 生成一个样本：随机初始条件 u(x,0)，数值积分得到 u(x,T)
fn generate_one_sample(
    config: &Heat1DConfig,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let n = config.n_grid;
    let dx = config.length / (n as f32 - 1.0);
    let r = config.alpha * config.dt / (dx * dx);
    // 显式方法稳定性条件: r <= 0.5
    if r > 0.5 {
        eprintln!("Warning: unstable scheme, r = {r} > 0.5");
    }

    // 随机初始条件：几个高斯峰叠加
    let mut rng = rand::thread_rng();
    let mut u0_vec = vec![0f32; n];
    let n_peaks = 3;
    for _ in 0..n_peaks {
        let center = rng.gen_range(0.2f32..0.8f32) * config.length;
        let width = rng.gen_range(0.05f32..0.2f32) * config.length;
        let amp = rng.gen_range(0.5f32..1.5f32);
        for i in 0..n {
            let x = i as f32 * dx;
            let val = amp * (-(x - center).powi(2) / (2.0 * width * width)).exp();
            u0_vec[i] += val;
        }
    }

    let mut u = Tensor::from_vec(u0_vec.clone(), (n,), device)?
        .to_dtype(DType::F32)?;
    // 边界条件：两端固定为 0
    for _step in 0..config.n_steps {
        let u_next = step_heat_1d(&u, r)?;
        u = u_next;
    }

    // 形状改为 [H, 1]
    let u0 = Tensor::from_vec(u0_vec, (n, 1), device)?.to_dtype(DType::F32)?;
    let u_t = u.reshape((n, 1))?;

    Ok((u0, u_t))
}


fn step_heat_1d(u: &Tensor, r: f32) -> CandleResult<Tensor> {
    let n = u.dims()[0];

    // pad 两个 ghost cell，左右各一个
    let zero_left = Tensor::zeros(1, u.dtype(), u.device())?;
    let zero_right = Tensor::zeros(1, u.dtype(), u.device())?;
    let u_padded = Tensor::cat(&[&zero_left, u, &zero_right], 0)?;

    let u_left = u_padded.narrow(0, 0, n)?;       // [n]
    let u_center = u_padded.narrow(0, 1, n)?;     // [n]
    let u_right = u_padded.narrow(0, 2, n)?;      // [n]

    // sum_lr = u_left + u_right
    let sum_lr = (&u_left + &u_right)?;           // Tensor

    // two_u_center = 2.0 * u_center
    let two = Tensor::full(2.0f32, u_center.dims(), u_center.device())?;
    let two_u_center = u_center.mul(&two)?;       // Tensor

    // lap = sum_lr - two_u_center
    let lap = sum_lr.sub(&two_u_center)?;         // Tensor

    // r_lap = r * lap
    let r_tensor = Tensor::full(r, lap.dims(), lap.device())?;
    let r_lap = lap.mul(&r_tensor)?;              // Tensor

    // u_next = u_center + r_lap
    let u_next = u_center.add(&r_lap)?;           // Tensor

    Ok(u_next)
}