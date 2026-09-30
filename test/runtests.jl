using Test
using ExperimentSimulation
using LinearAlgebra
using Random
using Unitful

@testset "ODE integration" begin
    r0 = [0.0u"s", 1.0u"m", 0.0u"m", 0.0u"m", 1.0u"GeV",
          0.0u"GeV/c", 0.0u"GeV/c", 0.0u"GeV/c"]
    # x' = x / 1 s has the known solution x(1 s) = exp(1) m.
    growth(r, par) = [1.0u"s/s", r[2] / 1.0u"s", 0.0u"m/s", 0.0u"m/s",
                      0.0u"GeV/s", 0.0u"GeV/c/s", 0.0u"GeV/c/s", 0.0u"GeV/c/s"]
    for (solver, order) in [(euler, 1), (predictor_corrector, 2), (runge_kutta_4, 4)]
        coarse = solver(growth, r0, nothing, 1.0u"s", 0.1u"s")
        fine = solver(growth, r0, nothing, 1.0u"s", 0.05u"s")
        coarse_error = abs(ustrip(u"m", last(coarse)[2]) - exp(1))
        fine_error = abs(ustrip(u"m", last(fine)[2]) - exp(1))
        @test coarse_error / fine_error > 0.8 * 2^order
        @test length(coarse) == 11
        @test last(coarse)[1] ≈ 1.0u"s"
        @test first(coarse) == r0
        @test r0[2] == 1.0u"m" # Solvers must not mutate the input state.
        @test only(solver(growth, r0, nothing, r0[1], 0.1u"s")) == r0
        @test_throws ArgumentError solver(growth, r0, nothing, 1.0u"s", 0.0u"s")
        @test_throws ArgumentError solver(growth, r0, nothing, -1.0u"s", 0.1u"s")
    end

    # With no fields, motion is uniform and energy and momentum stay fixed.
    initial = copy(r0)
    initial[6] = 0.01u"GeV/c"
    par = [1.60217663e-19u"C", 1.0u"GeV/c^2",
           fill(0.0u"V/m", 3), fill(0.0u"T", 3)]
    derivative = eom(initial, par)
    for solver in (euler, predictor_corrector, runge_kutta_4)
        trajectory = solver(eom, initial, par, 1.0u"ns", 0.1u"ns")
        @test length(trajectory) == 11
        @test last(trajectory)[1] ≈ 1.0u"ns"
        @test last(trajectory)[2] ≈ initial[2] + derivative[2] * 1.0u"ns"
        @test last(trajectory)[5:8] == initial[5:8]
    end
end

@testset "Resonance sampling" begin
    positions = [1.0, 2.0, 3.0]
    weights = [1.0, 2.0, 7.0]
    cdf = cumsum(weights) / sum(weights)
    n = 50_000
    inverse = inverse_cdf_sampling(cdf, positions, n; rng=MersenneTwister(108))
    @test sum(inverse) == n
    @test maximum(abs.(inverse / n - weights / sum(weights))) < 0.015
    @test inverse == inverse_cdf_sampling(cdf, positions, n; rng=MersenneTwister(108))

    hits = hit_and_miss_sampling(weights, positions, n; rng=MersenneTwister(108))
    @test 0 < sum(hits) <= n
    @test maximum(abs.(hits / sum(hits) - weights / sum(weights))) < 0.015
    @test efficiency(hits, n) ≈ sum(weights) / (length(weights) * maximum(weights)) atol=0.015
    # Regression: the final bin and a single-bin distribution must be sampled.
    @test hits[end] > 0
    @test hit_and_miss_sampling([1.0], [3.0], 20; rng=MersenneTwister(1)) == [20.0]
    @test inverse_cdf_sampling([1.0], [3.0], 20; rng=MersenneTwister(1)) == [20.0]
    values = sample_values(cdf, positions, 100; rng=MersenneTwister(2))
    @test length(values) == 100
    @test all(in(positions), values)
    @test chi2(weights, weights) == 0
    @test_throws ArgumentError chi2([1], [1, 2])
end

@testset "Decay kinematics and reconstruction" begin
    mass = 3.097u"GeV/c^2"
    rng = MersenneTwister(108)
    for _ in 1:20
        minus, plus = jpsi_to_mumu(mass; rng)
        @test all(isapprox.(minus[2:4], -plus[2:4]))
        @test minus[1] + plus[1] ≈ mass * (1.0u"c")^2
        @test jpsi_from_mumu(minus, plus) ≈ mass
    end
    @test jpsi_to_mumu(mass; rng=MersenneTwister(1)) == jpsi_to_mumu(mass; rng=MersenneTwister(1))
    energy, px, py, pz = four_momentum(0.1057u"GeV/c^2", 1.0u"GeV/c", 0.3, 0.7)
    cosine, phi = angles(px, py, pz)
    @test cosine ≈ 0.3
    @test phi ≈ 0.7
    @test norm([px, py, pz]) ≈ 1.0u"GeV/c"
    @test energy^2 ≈ (1.0u"c")^2 * (px^2 + py^2 + pz^2) + (1.0u"c")^4 * (0.1057u"GeV/c^2")^2

    # A quarter-circle in a 1 T field has chord sqrt(2) R and p = |q| B R.
    q = 1.602176634e-19u"C"
    field = [0.0u"T", 0.0u"T", 1.0u"T"]
    p = momentum_from_radius(1.0u"m", q, field)
    state1 = [0.0u"ns", 1.0u"m", 0.0u"m", 0.0u"m", 1.0u"GeV", 0.0u"GeV/c", p, 0.0u"GeV/c"]
    state2 = [1.0u"ns", 0.0u"m", 1.0u"m", 0.0u"m", 1.0u"GeV", -p, 0.0u"GeV/c", 0.0u"GeV/c"]
    radius, reconstructed = momentum_from_circle(state1, state2, q, field)
    @test radius ≈ 1.0u"m"
    @test reconstructed ≈ p
end

@testset "Continuous resonance model" begin
    parameters = [10.0, 3.0, 2.0, 1.0, 0.0]
    @test model(3.0, parameters) ≈ 11.0
    @test model([2.0, 4.0], parameters) ≈ [6.0, 6.0]
    grid = range(0.0, 1.0; length=11)
    draws = hit_and_miss_continuous(x -> 1.0, grid, ones(11), 100; rng=MersenneTwister(108))
    @test length(draws) == 100
    @test all(x -> 0.0 <= x <= 1.0, draws)
end
