use candle_core::{Device, DType, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::optim::{AdamW, Optimizer};

use em_no_core::{FNO1D, FNO1DConfig};
use em_no_physics::TimeEvolution1D;

fn mse_loss(pred: &Tensor, target: &Tensor) -> CandleResult<Tensor> {
    let diff = pred.sub(target)?;  // Result<Tensor> -> Tensor
    let sq = diff.sqr()?;
    let mean = sq.mean_all()?;
    Ok(mean)
}

/// 通用 1D PDE + FNO 训练函数
pub fn train_fno1d_on<P: TimeEvolution1D>(
    problem: &P,
    device: &Device,
    n_samples: usize,
    n_epochs: usize,
) -> CandleResult<()> {
    // 1. 数据集
    let (inputs, targets) = problem.generate_dataset(n_samples, device)?;
    let h = problem.grid_size();
    let c_in = problem.in_channels();
    let c_out = problem.out_channels();

    println!(
        "Train 1D problem: H = {}, C_in = {}, C_out = {}, samples = {}",
        h,
        c_in,
        c_out,
        inputs.len()
    );

    // 2. FNO 配置：对齐通道数
    let fno_cfg = FNO1DConfig {
        in_channels: c_in,
        out_channels: c_out,
        ..FNO1DConfig::default()
    };
    println!("FNO1DConfig: {:?}", fno_cfg);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = FNO1D::new(&fno_cfg, vb)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    // 3. 训练循环
    for epoch in 0..n_epochs {
        let mut total_loss = 0f32;

        for (u0, u_t) in inputs.iter().zip(targets.iter()) {
            // u0, u_t: [H, C]
            let input = u0.reshape((1, h, c_in))?;
            let target = u_t.reshape((1, h, c_out))?;

            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            total_loss += loss.to_scalar::<f32>()?;

            let grads = loss.backward()?;
            opt.step(&grads)?;
        }

        let avg_loss = total_loss / inputs.len() as f32;
        println!("Epoch {epoch}: avg loss = {avg_loss}");
    }

    Ok(())
}