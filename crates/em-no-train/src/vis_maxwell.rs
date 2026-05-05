// crates/em-no-train/src/vis_maxwell.rs
use candle_core::{Device, DType, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::optim::{AdamW, Optimizer};

use em_no_core::{FNO1D, FNO1DConfig};
use em_no_physics::{Maxwell1DConfig, Maxwell1DDataset, TimeEvolution1D};

use crate::vis_types::MaxwellSampleVis;

fn mse_loss(pred: &Tensor, target: &Tensor) -> CandleResult<Tensor> {
    let diff = pred.sub(target)?;
    let sq = diff.sqr()?;
    let mean = sq.mean_all()?;
    Ok(mean)
}

pub fn get_one_maxwell_vis_sample(
    device: &Device,
) -> CandleResult<MaxwellSampleVis> {
    let cfg = Maxwell1DConfig::default();
    let h = cfg.n_grid;
    let length = cfg.length;
    let dx = length / (h as f32 - 1.0);

    // 1. 训练一个小模型（也可以改成加载现有权重）
    let n_train_samples = 400;
    let n_epochs = 150;

    let (train_inputs, train_targets) =
        cfg.generate_dataset(n_train_samples, device)?;

    let c_in = cfg.in_channels();
    let c_out = cfg.out_channels();

    let fno_cfg = FNO1DConfig {
        in_channels: c_in,
        out_channels: c_out,
        ..FNO1DConfig::default()
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = FNO1D::new(&fno_cfg, vb)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    for epoch in 0..n_epochs {
        let mut total_loss = 0f32;
        for (u0, u_t) in train_inputs.iter().zip(train_targets.iter()) {
            let input = u0.reshape((1, h, c_in))?;
            let target = u_t.reshape((1, h, c_out))?;
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            total_loss += loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let avg_loss = total_loss / train_inputs.len() as f32;
        println!("[vis] Epoch {epoch}: avg loss = {avg_loss}");
    }

    // 2. 生成一个 eval 样本
    let eval_dataset = Maxwell1DDataset::new(1, &cfg, device)?;
    let u0 = &eval_dataset.inputs[0];
    let u_t = &eval_dataset.targets[0];

    let input = u0.reshape((1, h, c_in))?;
    let pred = model.forward(&input)?;

    // u0: [H, 3] = (E0, H0, eps_r)
    // u_t: [H, 2] = (E_T, H_T)
    let u0_arr = u0.to_vec2::<f32>()?;
    let ut_arr = u_t.to_vec2::<f32>()?;
    let pred_arr = pred.reshape((h, c_out))?.to_vec2::<f32>()?;

    let xs: Vec<f32> = (0..h).map(|i| i as f32 * dx).collect();
    let eps: Vec<f32> = (0..h).map(|i| u0_arr[i][2]).collect();
    let e_true: Vec<f32> = (0..h).map(|i| ut_arr[i][0]).collect();
    let h_true: Vec<f32> = (0..h).map(|i| ut_arr[i][1]).collect();
    let e_pred: Vec<f32> = (0..h).map(|i| pred_arr[i][0]).collect();
    let h_pred: Vec<f32> = (0..h).map(|i| pred_arr[i][1]).collect();

    Ok(MaxwellSampleVis {
        xs,
        eps,
        e_true,
        e_pred,
        h_true,
        h_pred,
    })
}