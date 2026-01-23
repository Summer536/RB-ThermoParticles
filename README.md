# RB-ThermoParticles(CUDA)

## Overview
This CUDA code solves 3D Rayleigh-Benard (RB) convection on GPU and supports finite-size spherical particles with particle dynamics and particle heat transfer. The fluid/thermal fields use the double-distribution LBM (D3Q27 + BGK). Particle-fluid coupling is handled with IBB (interpolated bounce-back). Particle-particle and particle-wall interactions use a soft-sphere repulsion model.

## Physical Model
- **Case**: 3D closed cavity RB convection (hot plate at x=0, cold plate at x=LX) with optional spherical thermal particles.
- **Flow**: weakly compressible LBM (BGK), Boussinesq buoyancy; buoyancy direction is **x**.
- **Temperature**: passive scalar LBM (g distribution), coupled with flow.
- **Boundary conditions**:
  - Velocity: no-slip via bounce-back on all walls.
  - Temperature: x=0 hot wall (Dirichlet), x=LX cold wall (Dirichlet), y/z adiabatic (Neumann).
- **Particles**:
  - Rigid finite-size spheres (diameter set by `PARTICLE_DIAM_FRAC`).
  - IBB linear interpolation for momentum and thermal bounce-back.
  - Soft-sphere repulsion for particle-particle and particle-wall contacts.

## Numerical Methods
- **Velocity set**: D3Q27.
- **Collision**: BGK for both f and g.
- **Forcing**: Guo-type forcing, explicitly uses Fx (buoyancy direction).
- **IBB momentum exchange**: link-based interpolation using q.
- **New-fluid refill**: first/second-order extrapolation if neighbors are available; otherwise reconstructed from particle surface macro state.

## Code Structure
- `main.cu`: time loop and timing.
- `parameters.h`: case and physical parameters (compile-time macros).
- `include/`
  - `lbm.h`: flow/thermal interfaces and constants.
  - `particle.h`: particle interfaces and data structures.
- `src_lbm/`
  - `initial.cu`: initialization and restart loading.
  - `macrovar.cu`: macroscopic fields and buoyancy.
  - `collision.cu`: BGK collision and dissipation fields.
  - `streaming.cu`: streaming and boundary conditions.
  - `globals.cu`: global variables and physical constants.
- `src_part/`
  - `pinit.cu`: particle init and initial IB nodes.
  - `links.cu`: IBB link build and new-fluid detection.
  - `ibb.cu`: IBB bounce-back and particle force/heat.
  - `pforces.cu`: repulsion and force history.
  - `move.cu`: particle translation/rotation/temperature update.
  - `refill.cu`: refill for new fluid nodes.
- `output/`
  - `output_flow.cu`: field output and f/g output.
  - `output_statis.cu`: diagnostics and Nu.
  - `output_part.cu`: particle output.

## Build and Run
1. Edit `parameters.h` (grid size, physical parameters, particle settings, output intervals).
2. Set GPU architecture in `Makefile` (e.g., `-arch=sm_80`).
3. Build and run:
```bash
make
./rb3d
```

## Restart
- Set `CONTINUOUS_STEPS` in `parameters.h` to the restart step.
- The code reads from `Ra%.1ePr%.2f/`:
  - `fg%09d.dat` (distribution functions f/g)
  - `%09d.dat` (macroscopic fields and diagnostics)
  - `pos/pos_%09d.txt` (particle states)

## Outputs
Output directory: `Ra%.1ePr%.2f/`
- `statistics.dat`: `umean/vmean/wmean/tmean` every `NDIAG` steps.
- `Nu_*.txt`: Nusselt output every `NNUOUT` steps.
- `%09d.dat`: binary fields in order  
  `rho, ux, uy, uz, phi, ted, Ked` (each `LXYZ` doubles).
- `fg%09d.dat`: binary distributions `f` and `g` (`NPOP*LXYZ` doubles each).
- `pos/pos_%09d.txt`: particle states (one particle per line).

## Performance (single GPU)
Test platform: RTX A800 (80GB)  
Grid: `LX*LY*LZ=320*320*160`, time steps: `NEND=1,000,000`
- Single-phase: **18.26 h**
- Particle-laden (multiphase): **20.95 h**

> Performance depends on GPU, compiler flags, and output frequency.
