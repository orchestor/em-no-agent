use candle_core::Device;
use em_no_train::{train_fno1d_on, get_one_maxwell_vis_sample};
use em_no_physics::{Heat1DConfig, Maxwell1DConfig};
use em_no_visual::run_vis;
use pollster::block_on;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = Device::Cpu;

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("heat1d");

    match mode {
        "heat1d" => {
            let problem = Heat1DConfig::default();
            train_fno1d_on(&problem, &device, 20, 10)?;
        }
        "maxwell1d" => {
            let problem = Maxwell1DConfig::default();
            train_fno1d_on(&problem, &device, 20, 10)?;
        }
        "maxwell1d-wgpu" => {
            let sample = get_one_maxwell_vis_sample(&device)?;
            block_on(run_vis(sample))?;
        }
        _ => {
            eprintln!(
                "Usage: em-no-agent [heat1d|maxwell1d|maxwell1d-wgpu]"
            );
        }
    }

    Ok(())
}