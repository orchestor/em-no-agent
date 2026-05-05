use candle_core::{Device, DType, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::optim::{AdamW, Optimizer};  // 关键：从 optim 模块引入 Optimizer trait
use em_no_core::{FNO1D, FNO1DConfig};
use em_no_physics::{Heat1DConfig, Heat1DDataset};

fn mse_loss(pred: &Tensor, target: &Tensor) -> CandleResult<Tensor> {
    let diff = pred.sub(target)?;      // Result<Tensor> -> Tensor
    let sq = diff.sqr()?;
    let mean = sq.mean_all()?;
    Ok(mean)
}

pub fn train_fno1d_on_heat1d(
    device: &Device,
    n_samples: usize,
    n_epochs: usize,
) -> CandleResult<()> {
    // 1. PDE 数据
    let heat_cfg = Heat1DConfig::default();
    println!("Heat1DConfig: {:?}", heat_cfg);

    let dataset = Heat1DDataset::new(n_samples, &heat_cfg, device)?;
    println!("Dataset size: {}", dataset.len());

    // 2. FNO1D 模型 + VarMap
    let fno_cfg = FNO1DConfig::default();
    println!("FNO1DConfig: {:?}", fno_cfg);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = FNO1D::new(&fno_cfg, vb)?;

    // 3. 优化器（AdamW from candle_nn::optim）
    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    // 4. 训练循环
    for epoch in 0..n_epochs {
        let mut total_loss = 0f32;

        for (u0, u_t) in dataset.inputs.iter().zip(dataset.targets.iter()) {
            // u0, u_t: [H, 1]
            let input = u0.reshape((1, heat_cfg.n_grid, 1))?;
            let target = u_t.reshape((1, heat_cfg.n_grid, 1))?;

            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            total_loss += loss.to_scalar::<f32>()?;

            // backward + step
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }

        let avg_loss = total_loss / dataset.len() as f32;
        println!("Epoch {epoch}: avg loss = {avg_loss}");
    }

    Ok(())
}