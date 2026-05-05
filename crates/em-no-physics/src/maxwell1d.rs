use candle_core::{Device, Result as CandleResult, Tensor};
use rand::Rng;

use crate::TimeEvolution1D;

#[derive(Debug, Clone)]
pub struct Maxwell1DConfig {
    pub length: f32,
    pub n_grid: usize,
    pub dt: f32,
    pub n_steps: usize,
    pub mu_r: f32,

    // 左右背景介质
    pub eps_r_left: f32,
    pub eps_r_right: f32,

    // slab 参数的采样范围（family）
    pub eps_r_slab_min: f32,
    pub eps_r_slab_max: f32,
    pub slab_start_min: f32, // 相对位置 [0,1]
    pub slab_start_max: f32,
    pub slab_width_min: f32, // 相对宽度 [0,1]
    pub slab_width_max: f32,
}

impl Default for Maxwell1DConfig {
    fn default() -> Self {
        Self {
            length: 1.0,
            n_grid: 64,
            dt: 5e-4,
            n_steps: 1200,
            mu_r: 1.0,
            eps_r_left: 1.0,
            eps_r_right: 1.0,
            eps_r_slab_min: 2.0,
            eps_r_slab_max: 6.0,
            slab_start_min: 0.3,
            slab_start_max: 0.6,
            slab_width_min: 0.1,
            slab_width_max: 0.3,
        }
    }
}

// 数据集：每个样本 input [H, 3]，target [H, 2]
// 通道: input(0=E0, 1=H0_interp, 2=eps_r), target(0=E_T, 1=H_T_interp)
pub struct Maxwell1DDataset {
    pub inputs: Vec<Tensor>,  // [H, 3]
    pub targets: Vec<Tensor>, // [H, 2]
}

impl Maxwell1DDataset {
    pub fn new(
        n_samples: usize,
        config: &Maxwell1DConfig,
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
}

/// 真·1D Maxwell FDTD（Yee 格子）生成一个样本
///
/// - E(i, nΔt) 定义在 i*Δx 上，i = 0..n_grid-1
/// - H(i+1/2, (n+1/2)Δt) 定义在 cell 中间，长度 n_grid-1
///
/// 这里使用无量纲化单位：
/// - epsilon = eps_r
/// - mu = mu_r
/// - c = 1 / sqrt(eps_r * mu_r)
///
/// 时间推进采用常见 1D Maxwell 形式：
/// - ∂H/∂t = -(1/mu) ∂E/∂x
/// - ∂E/∂t = -(1/eps) ∂H/∂x
///
/// 本版本升级：
/// 1. 初始 H0 与 E0 匹配，生成主要向右传播的 pulse。
/// 2. 左右边界使用一阶 Mur absorbing boundary，减少反射。
fn generate_one_sample(
    config: &Maxwell1DConfig,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let n = config.n_grid;

    assert!(n >= 3, "n_grid must be >= 3 for Mur boundary");
    assert!(config.length > 0.0, "length must be positive");
    assert!(config.dt > 0.0, "dt must be positive");
    assert!(config.mu_r > 0.0, "mu_r must be positive");
    assert!(config.eps_r_left > 0.0, "eps_r_left must be positive");
    assert!(config.eps_r_right > 0.0, "eps_r_right must be positive");
    assert!(
        config.eps_r_slab_min > 0.0 && config.eps_r_slab_max > 0.0,
        "eps_r_slab_min/max must be positive"
    );

    let length = config.length;
    let dx = length / (n as f32 - 1.0);
    let dt = config.dt;

    let mut rng = rand::thread_rng();

    // 1) 为本样本随机一组 slab 参数
    let eps_slab = rng.gen_range(config.eps_r_slab_min..=config.eps_r_slab_max);
    let slab_start_rel = rng.gen_range(config.slab_start_min..=config.slab_start_max);
    let slab_width_rel = rng.gen_range(config.slab_width_min..=config.slab_width_max);
    let slab_end_rel = (slab_start_rel + slab_width_rel).min(0.95);

    let slab_start_x = slab_start_rel * length;
    let slab_end_x = slab_end_rel * length;

    // 2) 空间分布 eps_r(i)
    let mut eps_r = vec![config.eps_r_left; n];

    for i in 0..n {
        let x = i as f32 * dx;

        eps_r[i] = if x >= slab_start_x && x <= slab_end_x {
            eps_slab
        } else if x > slab_end_x {
            config.eps_r_right
        } else {
            config.eps_r_left
        };
    }

    let mu = config.mu_r;

    // 3) CFL 检查：用最快波速，也就是最小 eps_r * mu
    let min_eps = eps_r.iter().copied().fold(f32::INFINITY, f32::min);
    let c_eff_max = 1.0 / (min_eps * mu).sqrt();
    let cfl = c_eff_max * dt / dx;

    if cfl > 1.0 {
        eprintln!("Warning: Maxwell1D CFL violated, cfl = {cfl}");
    }

    // 4) 更新系数
    let mut ce = vec![0f32; n];
    for i in 0..n {
        ce[i] = dt / (eps_r[i] * dx);
    }

    let ch = dt / (mu * dx);

    // 5) Mur absorbing boundary coefficients
    //
    // 一阶 Mur:
    // E_0^{n+1} = E_1^n + k_left  * (E_1^{n+1}   - E_0^n)
    // E_N^{n+1} = E_{N-1}^n + k_right * (E_{N-1}^{n+1} - E_N^n)
    //
    // k = (c * dt - dx) / (c * dt + dx)
    //
    // 左边界用 left medium wave speed，右边界用 right medium wave speed。
    let c_left = 1.0 / (config.eps_r_left * mu).sqrt();
    let c_right = 1.0 / (config.eps_r_right * mu).sqrt();

    let mur_left = (c_left * dt - dx) / (c_left * dt + dx);
    let mur_right = (c_right * dt - dx) / (c_right * dt + dx);

    // Yee 格子：E 有 n 点，H 有 n-1 点
    let mut e = vec![0f32; n];
    let mut h = vec![0f32; n - 1];

    // 6) 初始条件：主要向右传播的 Gaussian pulse
    //
    // 右行波关系：
    // H = E / Z = sqrt(eps / mu) * E
    //
    // 这里 pulse 放在左背景区，所以用 eps_r_left。
    // 注意：如果 pulse 初始位置太接近 slab，可以改小 center。
    let center = 0.15 * length;
    let width = 0.05 * length;

    let impedance_factor_left = (config.eps_r_left / mu).sqrt();

    // E 在整数格点
    for i in 0..n {
        let x = i as f32 * dx;
        let val = gaussian(x, center, width);
        e[i] = val;
    }

    // H 在半格点，用 cell center 的 Gaussian
    for i in 0..(n - 1) {
        let x_mid = (i as f32 + 0.5) * dx;
        let val_mid = gaussian(x_mid, center, width);
        h[i] = impedance_factor_left * val_mid;
    }

    let e0 = e.clone();
    let h0 = h.clone();

    // 7) 时间推进
    for _step in 0..config.n_steps {
        // 保存旧 E，用于 Mur boundary
        let e_old = e.clone();

        // 更新 H：
        //
        // ∂H/∂t = -(1/mu) ∂E/∂x
        for i in 0..(n - 1) {
            h[i] = h[i] - ch * (e[i + 1] - e[i]);
        }

        // 更新 E interior：
        //
        // ∂E/∂t = -(1/eps) ∂H/∂x
        let mut e_new = e.clone();

        for i in 1..(n - 1) {
            e_new[i] = e[i] - ce[i] * (h[i] - h[i - 1]);
        }

        // 8) Mur absorbing boundary
        //
        // 左边界:
        // E[0]^{n+1} = E[1]^n + k * (E[1]^{n+1} - E[0]^n)
        e_new[0] = e_old[1] + mur_left * (e_new[1] - e_old[0]);

        // 右边界:
        // E[n-1]^{n+1} = E[n-2]^n + k * (E[n-2]^{n+1} - E[n-1]^n)
        e_new[n - 1] = e_old[n - 2] + mur_right * (e_new[n - 2] - e_old[n - 1]);

        e = e_new;
    }

    let e_t = e;
    let h_t = h;

    // 9) 把 staggered H 插值到 E 网格上，得到 H_interp(i) 对齐 E(i)
    fn interp_h_to_e(h: &[f32], n: usize) -> Vec<f32> {
        let mut h_interp = vec![0f32; n];

        if h.is_empty() {
            return h_interp;
        }

        // 内部点：简单平均左右两个半格点
        for i in 1..(n - 1) {
            let left = h[i - 1];
            let right = h[i.min(h.len() - 1)];
            h_interp[i] = 0.5 * (left + right);
        }

        // 边界点：用最近的 H 值
        h_interp[0] = h[0];
        h_interp[n - 1] = h[h.len() - 1];

        h_interp
    }

    let h0_interp = interp_h_to_e(&h0, n);
    let h_t_interp = interp_h_to_e(&h_t, n);

    // 10) 输入: [H, 3] = (E0, H0_interp, eps_r(x))
    let mut u0 = Vec::with_capacity(3 * n);

    for i in 0..n {
        u0.push(e0[i]);
        u0.push(h0_interp[i]);
        u0.push(eps_r[i]);
    }

    // 11) 输出: [H, 2] = (E_T, H_T_interp)
    let mut u_t = Vec::with_capacity(2 * n);

    for i in 0..n {
        u_t.push(e_t[i]);
        u_t.push(h_t_interp[i]);
    }

    let u0 = Tensor::from_vec(u0, (n, 3), device)?;
    let u_t = Tensor::from_vec(u_t, (n, 2), device)?;

    Ok((u0, u_t))
}

fn gaussian(x: f32, center: f32, width: f32) -> f32 {
    (-(x - center).powi(2) / (2.0 * width * width)).exp()
}

// 把 Maxwell1DConfig 接到通用 TimeEvolution1D trait 上
impl TimeEvolution1D for Maxwell1DConfig {
    fn grid_size(&self) -> usize {
        self.n_grid
    }

    fn in_channels(&self) -> usize {
        3 // E0, H0_interp, eps_r(x)
    }

    fn out_channels(&self) -> usize {
        2 // E_T, H_T_interp
    }

    fn generate_dataset(
        &self,
        n_samples: usize,
        device: &Device,
    ) -> CandleResult<(Vec<Tensor>, Vec<Tensor>)> {
        let dataset = Maxwell1DDataset::new(n_samples, self, device)?;
        Ok((dataset.inputs, dataset.targets))
    }
}