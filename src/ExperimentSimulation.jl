"""Shared, unit-aware numerical routines for the Experiment 108 lab notebooks."""
module ExperimentSimulation

using Distributions: Bernoulli, Uniform
using LinearAlgebra: dot, norm
using Random
using Unitful

export eom, dimensions, euler, predictor_corrector, runge_kutta_4
export hit_and_miss_sampling, inverse_cdf_sampling, sample_values, chi2, efficiency
export four_momentum, angles, momentum_from_circle, radius_of_circle
export momentum_from_radius, jpsi_to_mumu, jpsi_from_mumu
export model, hit_and_miss_continuous

# Conversion constants used by the lab's existing unit-aware algorithms.
const GeV_to_J = 1.60218e-10u"J/GeV"
const c = 299792458u"m/s"
const c_1 = 1.0u"c"
const m_mu = 0.1057u"GeV/c^2"

include("ode_solvers.jl")
include("resonance_sampling.jl")
include("decay_reconstruction.jl")
include("resonance_fit.jl")

end # module ExperimentSimulation
