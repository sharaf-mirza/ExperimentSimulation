# Simulation of the Physics Experiment

This repository contains the implementation and analysis for **Experiment 108: Simulation of the Physics Experiment** from the Advanced Physics Lab Course at Ruhr-University Bochum.

The project focuses on numerical and statistical methods widely used in modern experimental and particle physics. The implementation is written in **Julia** using **Pluto notebooks** and combines numerical differential equation solving, Monte Carlo simulations, and particle decay reconstruction.

---

## Project Overview

The repository contains simulations and analysis for three main tasks:

### 1. Charged Particle Motion in Electromagnetic Fields
Simulation of electron trajectories in electric and magnetic fields using:
- Euler Method
- Predictor-Corrector Method
- Runge-Kutta 4 (RK4)

The project compares the stability and accuracy of different numerical ODE solvers for various time steps.

---

### 2. Monte Carlo Simulation of Resonance Masses
Implementation of Monte Carlo sampling techniques for resonance mass distributions:
- Acceptance-Rejection (Hit-and-Miss) Sampling
- Inverse CDF Sampling

The simulated distributions are compared with experimental resonance data using efficiency calculations and x² analysis.

---

### 3. $J/\psi \rightarrow \mu^+ \mu^-$ Decay Simulation
Simulation of the decay of a $J/\psi$ particle into two muons, including:
- generation of decay kinematics,
- propagation of muons in a magnetic field,
- reconstruction of muon momenta from circular trajectories,
- invariant mass reconstruction.

The project demonstrates basic concepts used in high-energy particle detectors and tracking systems.

---

Here is the link to the [overleaf project](https://www.overleaf.com/read/kmvbvnkrkbdn#b84e23) for the complete report with our results.

---

# Repository Structure

```text
.
├── Project.toml                    # Package and Pluto launcher dependencies
├── Manifest.toml                   # Locked root environment
├── README.md
├── src/
│   ├── ExperimentSimulation.jl     # Module entry point and shared constants
│   ├── ode_solvers.jl              # Equations of motion and three ODE solvers
│   ├── resonance_sampling.jl       # Discrete Monte Carlo sampling and comparisons
│   ├── resonance_fit.jl            # Fit model and continuous hit-and-miss sampling
│   └── decay_reconstruction.jl     # Decay kinematics and momentum reconstruction
├── notebooks/
│   ├── labcourse.jl                # Main interactive lab workflow, exercises 1–3
│   └── Exercise2_fit.jl            # Supplementary curve-fitting workflow (in progress)
├── data/
│   └── resonance.dat              # Experimental input dataset
├── figures/
│   └── 3D_traj.png                 # Saved trajectory figure
└── test/
    └── runtests.jl                 # Numerical and sampling tests
```

---

# Running the Project

Use **Julia 1.12**; the checked-in environments were generated with Julia 1.12.6.
Clone the repository and install the root environment from the repository folder:

```sh
git clone https://github.com/sharaf-mirza/ExperimentSimulation.git
cd ExperimentSimulation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pluto; Pluto.run()'
```

In Pluto, open either notebook:

```text
notebooks/labcourse.jl
notebooks/Exercise2_fit.jl
```

### How dependencies and paths work

The root `Project.toml` and `Manifest.toml` provide the shared package environment
and launch Pluto. Each notebook keeps its own embedded Pluto project and manifest;
Pluto installs and manages that notebook's analysis and plotting dependencies
automatically. Additional dependencies such as CSV, DataFrames, PlutoUI, and LsqFit
therefore belong to the notebook environments. Keep Pluto's generated environment
cells and cell-order metadata intact when editing the exported `.jl` files.

Both notebooks load the shared module from `src/` and read the dataset from `data/`
using paths relative to the notebook file. Launching Pluto from another working
directory does not change those paths. Keep the repository folders together when
sharing a notebook. Re-run the notebook's first setup cell after editing `src/`
to load the updated functions.

### Use the shared functions without Pluto

From the repository folder, start `julia --project=.` and run:

```julia
using ExperimentSimulation
using Random
using Unitful

# Generate and reconstruct a J/psi decay using a repeatable random stream.
rng = MersenneTwister(108)
p_minus, p_plus = jpsi_to_mumu(3.097u"GeV/c^2"; rng)
reconstructed_mass = jpsi_from_mumu(p_minus, p_plus)
```

The ODE functions use the lab's state layout
`[t, x, y, z, energy, px, py, pz]` and parameters `[charge, mass, E, B]` with
Unitful quantities. Sampling and decay functions accept an optional `rng` keyword;
omitting it uses Julia's default random stream, as in the interactive notebooks.

### Run tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests check ODE convergence against an analytic solution, uniform motion in
zero fields, sampling frequencies and the final histogram bin, decay momentum
balance and invariant mass, and a known circular-track reconstruction case.
They also check invalid time steps and repeatability with a seeded RNG.

### Scientific scope

The shared module retains the lab notebook's equations and reconstruction
approximations. Extracting them into `src/` does not constitute a full validation
of the physics. The existing `chi2` helper computes a sum of squared residuals;
its name is retained for compatibility, but this score is not a statistical
chi-square. The fitting notebook remains a supplementary workflow in progress.

Input data belongs in `data/`; saved figures belong in `figures/`. Temporary
figure exports can go in `figures/generated/`, which is ignored by Git.

---

# Repository Purpose

This repository serves as:
- a complete record of the experiment workflow,
- an implementation of the required numerical methods,
- documentation of simulations and analysis,
- a collaboration and review space for the lab experiment.

---
