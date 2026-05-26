# Simulation of the Physics Experimet — Experiment 108

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

### 3. \(J/\psi \rightarrow \mu^+\mu^-\) Decay Simulation
Simulation of the decay of a \(J/\psi\) particle into two muons, including:
- generation of decay kinematics,
- propagation of muons in a magnetic field,
- reconstruction of muon momenta from circular trajectories,
- invariant mass reconstruction.

The project demonstrates basic concepts used in high-energy particle detectors and tracking systems.

---

# Repository Structure

```text
.
├── labcourse.jl        # Main Pluto notebook
├── Exercise2_fit.jl    # MC Simulation from curve fitting (not complete yet)
├── resonance.dat       # Experimental resonance dataset
├── README.md           # Repository documentation
```

---

# Running the Project

## Install Pluto
```julia
using Pkg
Pkg.add("Pluto")
```

## Start Pluto
```julia
using Pluto
Pluto.run()
```

Then open:
```text
labcourse.jl
```

---

# Repository Purpose

This repository serves as:
- a complete record of the experiment workflow,
- an implementation of the required numerical methods,
- documentation of simulations and analysis,
- a collaboration and review space for the lab experiment.

---
