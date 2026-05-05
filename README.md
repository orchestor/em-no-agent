# em-no-agent

Physics-driven AI agents for 1D Maxwell equations, implemented end‑to‑end in Rust.

This project combines numerical electromagnetics, Fourier Neural Operators (FNOs), and real‑time GPU visualization into a single Rust codebase. It is intended both as a research playground and a demonstration of production‑grade systems skills: async orchestration, GPU compute integration, and model–physics co‑design.

---

## Key Features

- **Rust + Candle deep learning**
  - Implements a 1D Fourier Neural Operator in pure Rust on top of [candle](https://github.com/huggingface/candle).[web:54]
  - Uses a clean `FNO1DConfig` abstraction (channels, modes, layers) to control model capacity.
  - Training loop, optimizer (AdamW), and data pipeline are all explicit and type‑safe.

- **Physically grounded Maxwell 1D simulator**
  - Custom 1D Yee‑grid FDTD solver for Maxwell’s equations with spatially varying permittivity.
  - Supports Gaussian pulse excitation, random dielectric slabs, and realistic absorbing boundaries (first‑order Mur) instead of hard PEC walls.
  - Inputs/outputs are designed as operators: \( (E_0, H_0, \varepsilon_r(x)) \rightarrow (E_T, H_T) \), not just pointwise regression.

- **Neural operator training on synthetic data**
  - Generates supervised datasets by running the FDTD solver for many random slab configurations.
  - Trains an FNO to approximate the time‑evolution operator of the PDE.
  - Logs per‑epoch loss and supports quick adjustment of model size, number of samples, and training epochs.

- **Real‑time visualization with wgpu + winit 0.30**
  - Uses `wgpu = 29.0.1` and `winit = 0.30.13` for a modern GPU rendering stack.[web:19][web:20]
  - Renders:
    - True electric field \(E_T(x)\) (blue)
    - Predicted field \(\hat{E}_T(x)\) (red)
    - Relative permittivity \(\varepsilon_r(x)\) (gray)
  - Integrates with `ApplicationHandler` and `run_app` (no deprecated event loop APIs).

- **Emphasis on correctness + engineering trade‑offs**
  - Carefully handles lifetimes and ownership around `winit 0.30` and `wgpu 29` (Window + Surface + State).
  - Uses `Arc<Window>` and safe `CurrentSurfaceTexture` handling instead of unbounded unsafe.
  - Explicit separation of concerns:
    - `em-no-physics`: numerical PDE / FDTD
    - `em-no-train`: dataset generation + FNO training
    - `em-no-visual`: wgpu visualization
    - `em-no-cli`: CLI entry points tying everything together

---

## Project Structure

```text
em-no-agent/
  crates/
    em-no-physics/    # Maxwell1DConfig, FDTD solver, Mur boundaries
    em-no-train/      # FNO1D, dataset generation, training and vis sampling
    em-no-visual/     # wgpu + winit visualization (E_true, E_pred, eps_r)
    em-no-cli/        # CLI binaries and user-facing commands
  Cargo.toml
  README.md
```

### Physics: `em-no-physics`

- Defines `Maxwell1DConfig`:
  - Domain length, grid size, time step, number of time steps
  - Background media and random dielectric slab family (position, width, permittivity range)
- Implements a Yee‑grid FDTD time‑stepping scheme:
  - Electric field \(E(i, n\Delta t)\) on grid points
  - Magnetic field \(H(i + 1/2, (n + 1/2)\Delta t)\) on staggered half‑cells
- Uses first‑order Mur absorbing boundary conditions to reduce artificial reflections.
- Provides a `Maxwell1DDataset` type and implements a `TimeEvolution1D` trait to generate pairs:
  - `input: [H, 3] = (E0, H0_interp, eps_r)`
  - `target: [H, 2] = (E_T, H_T_interp)`

### Neural Operator: `em-no-train`

- Defines `FNO1DConfig`:

  ```rust
  pub struct FNO1DConfig {
      pub in_channels: usize,
      pub out_channels: usize,
      pub hidden_channels: usize,
      pub fourier_modes: usize,
      pub layers: usize,
  }
  ```

- Implements `FNO1D` on top of Candle:
  - Spectral convolution in Fourier space
  - Pointwise layers for nonlinearity and channel mixing
- Training logic (`get_one_maxwell_vis_sample`):
  - Generate a dataset of random Maxwell 1D samples on a given `Device` (CPU or CUDA).
  - Train FNO with `AdamW` and MSE loss:
    - Logs `[vis] Epoch k: avg loss = ...`
  - After training, runs the model on one sample to produce a `MaxwellSampleVis`:
    - `xs` (grid coordinates)
    - `eps` (relative permittivity)
    - `e_true, h_true` (PDE solution at final time)
    - `e_pred, h_pred` (FNO prediction)

### Visualization: `em-no-visual`

- Uses `wgpu 29.0.1` and `winit 0.30.13` with `ApplicationHandler`.[web:19][web:20]
- `State` owns:
  - `Arc<Window>` and `Surface<'static>` created via `Instance::create_surface`
  - `Device`, `Queue`, `SurfaceConfiguration`
  - Pipeline and vertex buffers for three line sets:
    - `vb_eps`  (gray): `eps_r(x)`
    - `vb_e_true` (blue): true \(E_T(x)\)
    - `vb_e_pred` (red): predicted \(\hat{E}_T(x)\)
- Render loop:
  - Safely handles `CurrentSurfaceTexture::{Success, Suboptimal, Timeout, Occluded, Outdated, Lost, Validation}`.
  - Reconfigures the surface when needed; skips frames on transient errors instead of panicking.
- `App` implements `ApplicationHandler`:
  - `resumed`: create window, initialize `State` via `pollster::block_on(State::new(...))`, and store it.
  - `window_event`: handle `CloseRequested`, `Resized`, and `RedrawRequested` by resizing and triggering `State::render()`.

### CLI: `em-no-cli`

- Provides user‑facing commands (e.g. from `main.rs`):
  - `maxwell1d` – train on 1D Maxwell dataset without visualization.
  - `maxwell1d-wgpu` – train, then open a visualization window to inspect one sample’s true vs predicted fields and permittivity.

---

## How to Build and Run

### 1. Prerequisites

- Rust toolchain (stable)
- For CPU training:
  - No special requirements beyond what Candle needs in CPU mode.[web:50]
- For GPU training with CUDA (optional):
  - NVIDIA GPU + CUDA toolkit
  - Candle compiled with `cuda` feature enabled.[web:50][web:54]

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/em-no-agent.git
cd em-no-agent
```

### 3. Build

```bash
cargo build
```

### 4. Run training only (no visualization)

```bash
cargo run --bin em-no-agent -- maxwell1d
```

This trains an FNO on synthetically generated Maxwell 1D data and prints per‑epoch loss:

```text
[vis] Epoch 0:  avg loss = ...
[vis] Epoch 1:  avg loss = ...
...
```

### 5. Run training + visualization

```bash
cargo run --bin em-no-agent -- maxwell1d-wgpu
```

This will:

1. Train an FNO on the Maxwell 1D dataset.  
2. Generate one visualization sample.  
3. Open a GPU window showing:
   - Gray: `eps_r(x)` (dielectric profile)
   - Blue: true \(E_T(x)\)
   - Red: predicted \(\hat{E}_T(x)\)

Resize events are handled, and the window can be closed normally.

---

## Why this project

I built `em-no-agent` to explore how far we can push **physics-guided AI** within a single, strongly‑typed systems language:

- Treat PDE solvers and neural operators as composable modules, not black boxes.
- Use Rust’s ownership and lifetime system to keep GPU resources and event loops correct.
- Expose enough configuration to be a real research sandbox (slab distributions, boundary conditions, FNO capacity), while still being small enough to read end‑to‑end.

For hiring managers: this repository is a concrete demonstration of my ability to:

- Design and implement numerical physics simulations from first principles.  
- Build and train custom neural architectures (FNO) without relying on Python frameworks.  
- Integrate GPU rendering and modern event systems (winit 0.30 + wgpu 29) into a cohesive tool.  
- Make pragmatic engineering trade‑offs between safety (lifetimes, ownership) and ergonomics.