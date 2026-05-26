### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ac552774-8faa-11ef-257f-ad04db824377
begin
	using Pkg
	using Plots
	using CSV
	using DataFrames
	using Random
	using PlutoUI
	using Unitful
	using LinearAlgebra
	using Distributions
	using LsqFit
end

# ╔═╡ efdd6d2d-2f5b-49a6-910b-098899bc59e3
md"""
# Exersice 1: Methods of solving ODE
"""

# ╔═╡ 33699dc6-4d43-480f-905e-2674db611b22
md"""
### Simulation Parameters

#### Charge and Mass
**Charge**: ``q`` = $(@bind q_e Slider(-5.0:0.1:5.0, default=1.0, show_value=true)) elementary charges

**Mass**: ``m`` = $(@bind m Slider(0.0u"GeV/c^2":0.1u"GeV/c^2":20.0u"GeV/c^2", default=1.0u"MeV/c^2", show_value=true))

#### Electric Field

``E_x`` = $(@bind Ex Slider(0.0u"V/m":0.1u"V/m":10.0u"V/m", default=0.0u"V/m", show_value=true))

``E_y`` = $(@bind Ey Slider(0.0u"V/m":0.1u"V/m":10.0u"V/m", default=0.0u"V/m", show_value=true))

``E_z`` = $(@bind Ez Slider(0.0u"V/m":0.1u"V/m":10.0u"V/m", default=0.0u"V/m", show_value=true))

#### Magnetic Field

``B_x`` = $(@bind Bx Slider(0.0u"T":0.1u"T":10.0u"T", default=0.0u"T", show_value=true))

``B_y`` = $(@bind By Slider(0.0u"T":0.1u"T":10.0u"T", default=0.0u"T", show_value=true))

``B_z`` = $(@bind Bz Slider(0.0u"T":0.1u"T":10.0u"T", default=1.0u"T", show_value=true))

#### Initial Conditions

``t_0`` = $(@bind t0 Slider(0.0u"s":0.1u"s":10.0u"s", default=0.0u"s", show_value=true))

``x_0`` = $(@bind x0 Slider(0.0u"m":0.1u"m":10.0u"m", default=0.0u"m", show_value=true))

``y_0`` = $(@bind y0 Slider(0.0u"m":0.1u"m":10.0u"m", default=0.0u"m", show_value=true))

``z_0`` = $(@bind z0 Slider(0.0u"m":0.1u"m":10.0u"m", default=0.0u"m", show_value=true))

``p_{x_0}`` = $(@bind px0 Slider(0.0u"GeV/c":0.1u"GeV/c":10.0u"GeV/c", default=1.0u"GeV/c", show_value=true))

``p_{y_0}`` = $(@bind py0 Slider(0.0u"GeV/c":0.1u"GeV/c":10.0u"GeV/c", default=0.0u"GeV/c", show_value=true))

``p_{z_0}`` = $(@bind pz0 Slider(0.0u"GeV/c":0.1u"GeV/c":10.0u"GeV/c", default=0.0u"GeV/c", show_value=true))

#### Simulation Time and Time Step
``t_{\text{max}}`` = $(@bind tmax Slider(0.0u"ns":1.0u"ns":50.0u"ns", default=10.0u"ns", show_value=true))

``Δt`` = $(@bind dt Slider(0.0u"ns":0.01u"ns":1.0u"ns", default=0.01u"ns", show_value=true))
"""

# ╔═╡ cc9d7ea9-91ce-472c-abfe-a67d8d55403e
begin
	const GeV_to_J = 1.60218e-10u"J/GeV"
	const c = 299792458u"m/s"
	const c_1 = 1.0u"c"
end

# ╔═╡ 0aee2d4a-42fb-4569-b199-d7535d811baa
# Equations of motion

function eom(r, par)
    # Unpack the variables from the state vector u
    # r = [t, x, y, z, E, px, py, pz]
    t, x, y, z = r[1], r[2], r[3], r[4]
    En, px, py, pz = r[5], r[6], r[7], r[8]

    # Unpack parameters
    q, m_GeV, E, B = par

    m = m_GeV * GeV_to_J * c_1^2 / c^2
    px_si = px * GeV_to_J * c_1 / c
    py_si = py * GeV_to_J * c_1 / c
    pz_si = pz * GeV_to_J * c_1 / c

    vx = px_si / m
    vy = py_si / m
    vz = pz_si / m
    Ex, Ey, Ez = E[1], E[2], E[3]
    Bx, By, Bz = B[1], B[2], B[3]

    # Lorentz force in SI units (momentum derivatives) using F = q * (E + v × B)
    dpx = q * (Ex + vy * Bz - vz * By)
    dpy = q * (Ey + vz * Bx - vx * Bz)
    dpz = q * (Ez + vx * By - vy * Bx)

    # Convert the force back to GeV/c for updating momentum
    dpx_GeV = dpx / (GeV_to_J * c_1 / c)
    dpy_GeV = dpy / (GeV_to_J * c_1 / c)
    dpz_GeV = dpz / (GeV_to_J * c_1 / c)

	# Energy derivative: dE/dt = q * v ⋅ E
	dEn = q * ((vx * Ex) + (vy * Ey) + (vz * Ez))

	# Convert J/s to GeV/s
	dEn_GeV = dEn / GeV_to_J


    # Return derivatives: [dt/dt, dx/dt, dy/dt, dz/dt, dEn/dt, dpx/dt, dpy/dt, dpz/dt]
    return [1.0u"s/s", vx, vy, vz, dEn_GeV, dpx_GeV, dpy_GeV, dpz_GeV]
end

# ╔═╡ 096a3b73-bdce-4a38-9ce0-bb186601c2e4
# To keep state vector in the right dimension
function dimensions(r)
	r[1] = uconvert(u"ns",    r[1])
	r[2] = uconvert(u"m",     r[2])
	r[3] = uconvert(u"m",     r[3])
	r[4] = uconvert(u"m",     r[4])
	r[5] = uconvert(u"GeV",   r[5])
	r[6] = uconvert(u"GeV/c", r[6])
	r[7] = uconvert(u"GeV/c", r[7])
	r[8] = uconvert(u"GeV/c", r[8])
    return r
end

# ╔═╡ 884fc9f1-139a-49c4-bb0e-735715db0e84
# Euler method
function euler(f, r0, par, tmax, dt)
    ts = (r0[1]+dt):dt:tmax  # Time steps
    n = length(ts)   # Number of steps
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory
	m = par[2]

    # Iterate through each time step
    for t in ts
        dimensions(r)
    	push!(rs, copy(r))

        ### Calculate all parameters of the state array (Hint: use broadcasting)
		r = r .+ dt .* f(r, par)
    end

	dimensions(r)
    push!(rs, copy(r))
    return rs
end

# ╔═╡ 4173869b-55f8-458f-8789-af44d022ebfd
# Predictor-corrector method
function predictor_corrector(f, r0, par, tmax, dt)
    ts = (r0[1]+dt):dt:tmax  # Time steps
    n = length(ts)   # Number of steps
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory
	m = par[2]

    # Iterate through each time step
    for t in ts
        dimensions(r)
   		push!(rs, copy(r))

		r_pred = r .+ dt * f(r, par)

        ### Calculate all parameters of the state array (Hint: use broadcasting)
		r = r .+ 0.5 .* dt .* (f(r, par) .+ f(r_pred, par))
    end

	dimensions(r)
    push!(rs, copy(r))
    return rs
end

# ╔═╡ acf9c865-02a4-488c-b0b2-11c9adbe4f83
# Runge-Kutta (RK4) method
function runge_kutta_4(f, r0, par, tmax, dt)
    ts = (r0[1]+dt):dt:tmax  # Time steps
    n = length(ts)   # Number of steps
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory
	m = par[2]

    # Iterate through each time step
    for t in ts
		dimensions(r)
        push!(rs, copy(r))

        k1 = dt * f(r, par)
        k2 = dt * f(r .+ 0.5 .* k1, par)
        k3 = dt * f(r .+ 0.5 .* k2, par)
        k4 = dt * f(r .+ k3, par)

       ### Calculate all parameters of the state array (Hint: use broadcasting, examplle of it is already in this function)
		r = r .+ (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4) ./ 6
    end

	dimensions(r)
    push!(rs, copy(r))
    return rs
end

# ╔═╡ 01ffa2e0-d25d-4694-aed7-537b4535e1ab
begin
	E = [Ex, Ey, Ez]
	B = [Bx, By, Bz]
	e = 1.60217663e-19u"C"
	q = q_e*e
	par = [q, m, E, B]
	
	# Define the initial conditions
	r0 = [t0, x0, y0, z0, 0.0u"GeV", px0, py0, pz0] # (t,x,y,z,En,px,py,pz)
	r0[5] = sqrt(c_1^2*(r0[6]^2 + r0[7]^2 + r0[8]^2) + c_1^4*m^2)
end

# ╔═╡ 224af2f9-e334-4289-8890-46393cc193bc
#rs = runge_kutta_4(eom, r0, par, tmax, dt)
 #rs = euler(eom, r0, par, tmax, dt)
 #rs = predictor_corrector(eom, r0, par, tmax, dt)
begin
	rs_euler = euler(eom, r0, par, tmax, dt)
    rs_pc = predictor_corrector(eom, r0, par, tmax, dt)
    rs_rk4 = runge_kutta_4(eom, r0, par, tmax, dt)
end

# ╔═╡ f99daddf-bc57-427d-a87c-3b9da7d116f9
rs_euler

# ╔═╡ 96002c68-50b6-409b-ad31-c2a7967414b0
### E.g. to access the array of time steps 
begin
	t_euler = [r[1] for r in rs_euler]
    x_euler = [r[2] for r in rs_euler]
    y_euler = [r[3] for r in rs_euler]
	z_euler = [r[4] for r in rs_euler]
	En_euler = [r[5] for r in rs_euler]

    t_pc = [r[1] for r in rs_pc]
    x_pc = [r[2] for r in rs_pc]
    y_pc = [r[3] for r in rs_pc]
	z_pc = [r[4] for r in rs_pc]
	En_pc = [r[5] for r in rs_pc]

    t_rk4 = [r[1] for r in rs_rk4]
    x_rk4 = [r[2] for r in rs_rk4]
    y_rk4 = [r[3] for r in rs_rk4]
	z_rk4 = [r[4] for r in rs_rk4]
	En_rk4 = [r[5] for r in rs_rk4]


end

# ╔═╡ 5a3cdea5-5140-43cb-a609-ae96161967ce
### Plot trajectory of the particle: x(t), y(t), z(t), (x,y)(t), x,y(z)
### Check that energy conserves, plot En(t)
### Repeat this for all three methods and compare trajectories.
begin

	#plot x(t)
    px = plot(t_euler, x_euler, label="Euler", xlabel="t", ylabel="x", title="x(t)", dpi = 300)
    plot!(px, t_pc, x_pc, label="Predictor-Corrector")
    plot!(px, t_rk4, x_rk4, label="RK4")

	#plot y(t)
    py = plot(t_euler, y_euler, label="Euler", xlabel="t", ylabel="y", title="y(t)")
    plot!(py, t_pc, y_pc, label="Predictor-Corrector")
    plot!(py, t_rk4, y_rk4, label="RK4")

	#plot z(t)
	pz = plot(t_euler, z_euler, label="Euler", xlabel="t", ylabel="z", title="z(t)")
    plot!(pz, t_pc, z_pc, label="Predictor-Corrector")
    plot!(pz, t_rk4, z_rk4, label="RK4")

	#plot (x,y)(t)
	    ptraj = plot(x_euler, y_euler, label="Euler", xlabel="x", ylabel="y", title="Particle trajectory in (x,y) plane")
	plot!(ptraj, x_pc, y_pc, label="Predictor-Corrector")
	plot!(ptraj, x_rk4, y_rk4, label="RK4")

	plot(px, py, pz, ptraj, layout=(2,2), size=(900,900))

	end

# ╔═╡ 0e9d6cb1-e037-46e1-908d-a67850b34139
begin
	# plot x(z)
	pxz = plot(
	    z_euler,
	    x_euler,
	    label="Euler",
	    xlabel="z",
	    ylabel="x",
	    title="x(z)"
	)
	
	plot!(pxz, z_pc, x_pc, label="Predictor-Corrector")
	plot!(pxz, z_rk4, x_rk4, label="RK4")
	
	
	# plot y(z)
	pyz = plot(
	    z_euler,
	    y_euler,
	    label="Euler",
	    xlabel="z",
	    ylabel="y",
	    title="y(z)"
	)
	
	plot!(pyz, z_pc, y_pc, label="Predictor-Corrector")
	plot!(pyz, z_rk4, y_rk4, label="RK4")

	plot(pxz, pyz, layout=(1,2), size=(950,500))
end

# ╔═╡ f4c4ceb6-a68f-47e0-930e-96dc91232567
begin
	p3d = plot(
	    x_euler,
	    y_euler,
	    z_euler,
	    label="Euler",
	    xlabel="x",
	    ylabel="y",
	    zlabel="z",
	    title="3D trajectory",
	    lw=2,
		size =(600,600)
	)
	
	plot!(p3d, x_pc, y_pc, z_pc, label="Predictor-Corrector")
	plot!(p3d, x_rk4, y_rk4, z_rk4, label="RK4")
end

# ╔═╡ 63bfaaf3-645d-4b33-95bf-9c2f4c65b2a0
begin
	
	pEn = plot(
	    t_euler,
	    En_euler,
	    label="Euler",
	    xlabel="t",
	    ylabel="Energy",
	    title="Energy conservation"
	)
	
	plot!(pEn, t_pc, En_pc, label="Predictor-Corrector")
	plot!(pEn, t_rk4, En_rk4, label="RK4")
	
	pEn
end

# ╔═╡ 042212eb-684d-446b-9785-3c127ad9f732
md"""
# Exersice 2: MC Simulation of the resonance mass
"""

# ╔═╡ 39481270-26dc-4eb3-9fd1-d83ac1f4bf03
begin
	file_path = "./resonance.dat"
	data = CSV.read(file_path, DataFrame, delim='\t')
end

# ╔═╡ ddefd433-a462-4fd9-807f-8e3a45cf07f5
begin
	bin_positions = data.x
	bin_contents = data.y
end

# ╔═╡ 01e2eca6-ae5c-4bf1-a22a-cea64b869d9b
num_samples = 10000

# ╔═╡ 5d352f88-8d33-4d0d-ad57-9bbb99773831
begin
	
		pdf = histogram(bin_positions, weights=bin_contents, bins=62, xlabel="Mass (MeV/c²)", ylabel="Candidates", title="PDF of Resonance Data", legend=false)
end

# ╔═╡ c4813601-5633-46f0-9a7a-bc0edc62e364
md"""
* The cumulative distribution function (CDF) is obtained by calculating the normalized cumulative sum of the histogram bin contents.

```math
CDF_k =
\frac{
\sum_{i=1}^{k} p_i
}{
\sum_{i=1}^{N} p_i
} 
```
"""

# ╔═╡ 7df5fae0-614e-4c36-a073-680e7aaec332
begin
	cdf_values = cumsum(bin_contents) ./ sum(bin_contents)### Calculate CDF based on PDF
	plot(
    bin_positions,
    cdf_values,
    seriestype = :steppost,
    fillrange = 0,
    fillcolor = :lightblue,
    xlabel = "Mass (MeV/c²)",
    ylabel = "CDF",
    title = "CDF of Resonance Data",
    legend = false
	)

end

# ╔═╡ 459b7567-d5fc-464d-9561-3c5141469df9
# Hit and miss method

function hit_and_miss_sampling(bin_contents, bin_positions, num_samples)	
	max_bin_content = maximum(bin_contents)
	accepted_samples_hm = zeros(lastindex(bin_positions))
	for i in 1:num_samples
		random_x = trunc(Int, rand() * (lastindex(bin_positions)-1) + 1)
		random_y = rand() * max_bin_content
		
    	if random_y <= bin_contents[random_x]
    		accepted_samples_hm[random_x] += 1
		end ### Position of hit is already set by (random_x,random_y), implement check whether it ahould be accepted or not
	end
	return accepted_samples_hm
end

# ╔═╡ 3567a205-65fb-4a28-a4f1-4334587e510d
begin
	accepted_samples_hm = hit_and_miss_sampling(bin_contents, bin_positions, num_samples)


	### Plot histogram with the simulated sample
	histogram(
        bin_positions,
        weights=accepted_samples_hm,
        bins=length(bin_positions),
        xlabel="Mass (MeV/c²)",
        ylabel="Counts",
        title="Hit-or-Miss Monte Carlo",
        label="Simulated sample"
    )
	
end

# ╔═╡ 93a82374-9227-401e-b875-c453e08ddafa
# Inverse CDF method

function inverse_cdf_sampling(cdf_values, bin_positions, num_samples)
	accepted_samples_inv = zeros(lastindex(bin_positions))
	for i in 1:num_samples

	
		u= rand()### Implement sampling of random variable u which is needed for inverse CDF method
        accepted_samples_inv[findfirst(>=(u), cdf_values)] = accepted_samples_inv[findfirst(>=(u), cdf_values)] + 1
	end
	return accepted_samples_inv
end

# ╔═╡ d339e66c-6c43-46fd-af99-dc7a8eb6520d
begin
	accepted_samples_inv = inverse_cdf_sampling(cdf_values, bin_positions, num_samples)

	### Plot histogram with the simulated sample
	histogram(
        bin_positions,
        weights=accepted_samples_inv,
        bins=length(bin_positions),
        xlabel="Mass (MeV/c²)",
        ylabel="Counts",
        title="Inverse CDF Monte Carlo",
        label="Simulated sample"
    )
end

# ╔═╡ b85856c4-b722-4f0c-a3f3-7566938173da
# Function to simulate mass of the resonance based on inverse CDF

function sample_values(cdf_values, bin_positions, events)
	values = []
	for i in 1:events
		u = rand()
        push!(values, copy(bin_positions[findfirst(>=(u), cdf_values)]))
	end
	return values
end

# ╔═╡ 1b9921eb-746b-4fdf-b5e2-f83c87c989e5
function chi2(sample, data)
	
    if length(sample) != length(data)
        throw(ArgumentError("Histograms must have the same number of bins"))
    end
	
    chi2_values = (data .- sample).^2### Implement calculation of chi2 values (Hint: use broadcasting or implement loop)
    sum(chi2_values)
end

# ╔═╡ b0b544b1-e26c-49d8-87cf-1002142534c7
begin
	### Calculate chi2 of Hit and miss method
	data_norm = bin_contents ./ sum(bin_contents)
	hm_norm = accepted_samples_hm ./ sum(accepted_samples_hm)
	
	chi2_hm = chi2(hm_norm, data_norm)
end

# ╔═╡ 7c274fb6-ccfe-484e-9c5b-321225186cb6
begin
	inv_norm = accepted_samples_inv ./ sum(accepted_samples_inv)
	chi2_inv = chi2(inv_norm, data_norm)### Calculate chi2 of Inverse CDF method
end

# ╔═╡ d2c1d638-f2c1-4834-9235-a9d55dbaa866
function efficiency(sample, num_samples)
	return sum(sample)/num_samples ### Implement calculation of efficiency
end

# ╔═╡ 6bb3a02f-4882-4626-9417-f3aa07521eb6
eff_hm = efficiency(accepted_samples_hm, num_samples)### Calculate efficiency of Hit and miss method

# ╔═╡ 9431493e-8a10-4af6-904b-c1a03625932c
eff_inv = efficiency(accepted_samples_inv, num_samples) ### Calculate efficiency of Inverse CDF method

# ╔═╡ 9d1ebe41-42bc-40c3-91f8-10a5d9707b66
md"""
## Comparison of Monte Carlo methods

The Hit-or-Miss method generates random points in a rectangular region and accepts only points lying below the probability density function. Therefore, many generated points are rejected, which reduces the efficiency.

The inverse CDF method directly generates samples according to the target probability distribution using the cumulative distribution function. This avoids rejected samples and is therefore more efficient.

The quality of both methods can be evaluated using the ``\chi^2`` value between the generated and original distributions.

In general:
- the inverse CDF method has higher efficiency,
- the inverse CDF method converges faster,
- the Hit-or-Miss method is conceptually simpler but computationally slower.
"""

# ╔═╡ 33074b6a-c172-4544-a575-0649e7664632
### Compare performances of the two methods
let
	sample_sizes = [10, 1000, 10000]
	
	for N in sample_sizes
	
	    hm = hit_and_miss_sampling(bin_contents, bin_positions, N)
	    inv = inverse_cdf_sampling(cdf_values, bin_positions, N)
	
	    hm_norm = hm ./ sum(hm)
	    inv_norm = inv ./ sum(inv)
	    data_norm = bin_contents ./ sum(bin_contents)
	
	    chi2_hm = chi2(hm_norm, data_norm)
	    chi2_inv = chi2(inv_norm, data_norm)
	
	    eff_hm = sum(hm) / N
	    eff_inv = sum(inv) / N
	
	    println("N = ", N)
	
	    println("Hit-or-Miss:")
	    println("   efficiency = ", eff_hm)
	    println("   chi2 = ", chi2_hm)
	
	    println("Inverse CDF:")
	    println("   efficiency = ", eff_inv)
	    println("   chi2 = ", chi2_inv)
	
	end
end

# ╔═╡ a3ea2044-932e-44c4-b5c6-77fdf355333f
md"""
## Discussion of the Monte Carlo methods

The efficiency of the hit-or-miss method remains approximately constant for increasing sample size. This is expected because the efficiency is determined mainly by the ratio between the area under the probability density function and the total sampling region.

The inverse CDF method has efficiency equal to 1 because every generated random number is accepted.

For increasing sample size, the ``\chi^2`` values decrease for both methods, indicating that the sampled distributions converge toward the original resonance distribution.

The inverse CDF method produces consistently smaller ``\chi^2`` values than the hit-or-miss method, demonstrating faster convergence and better computational performance.
"""

# ╔═╡ c13d949e-ebff-42bf-9d10-751b79e2a764
md"""
## Estimation of the resonance parameters

"""

# ╔═╡ c5ce4cb8-36be-42cd-87e4-d371f1220075
begin
    max_index = argmax(bin_contents)
    resonance_mass = bin_positions[max_index]

    half_max = maximum(bin_contents) / 2
    indices_half_max = findall(bin_contents .>= half_max)

    left_index = first(indices_half_max)
    right_index = last(indices_half_max)

    resonance_width = bin_positions[right_index] - bin_positions[left_index]

    println("Estimated resonance mass = ", resonance_mass, " MeV/c²")
    println("Estimated width FWHM = ", resonance_width, " MeV/c²")
end

# ╔═╡ 9a3f63ca-6dd8-410d-84fc-33f9a88f011b
md"""
he resonance mass was estimated from the position of the maximum of the experimental distribution. The resonance width was estimated using the full width at half maximum (FWHM) of the peak.

The obtained resonance mass is close to the known mass of the ``J/\psi``  particle reported by the Particle Data Group, approximately ``3.097GeV/c^2``. The estimated width is also consistent with the narrow resonance structure expected for the ``J/\psi``  particle.

Small deviations between the estimated values and the PDG values are expected due to finite binning of the histogram and statistical fluctuations in the sampled dataset.
"""

# ╔═╡ a86ea674-31a0-4285-9d33-84ffbdedeabd
md"""
# Exersice 3: Simulation of $J/\psi\rightarrow\mu^{+}\mu^{-}$ decay and reconstruction of muons momenta
"""

# ╔═╡ 2e8ccf97-d9f8-4683-8ad6-c9e2cdbbf80c
# Function to compute 4-momentum vector with angles, momentum and mass
function four_momentum(m, p, cos_theta, phi)
    theta = acos(cos_theta)
    
    px = p * sin(theta) * cos(phi)
    py = p * sin(theta) * sin(phi)
    pz = p * cos(theta)
    E = sqrt(c_1^2*p^2 + c_1^4*m^2)
    
    # Return the 4-momentum vector
    return E, px, py, pz
end

# ╔═╡ a69583de-164d-40c3-9562-134cdd19f6c5
# Function to get  angles based on the momentum components
function angles(px, py, pz)
    # Step 1: Compute the total momentum magnitude p
    p = sqrt(px^2 + py^2 + pz^2)
    
    # Step 2: Compute cos(theta) = pz / p
    cos_theta = pz / p
    
    # Step 3: Compute phi = atan2(py, px) (azimuthal angle in radians)
    phi = atan(py, px)
    
    # Return cos(theta) and phi
    return cos_theta, phi
end

# ╔═╡ 2467d8b6-0619-490f-baa2-da6a0fc8e99e
# Function to calculate the momentum of the particle based on the circular trajectory in magnetic field given by two state vectors
function momentum_from_circle(point1, point2, q, B)
    t1, x1, y1, z1, E1, px1, py1, pz1 = point1
    t2, x2, y2, z2, E2, px2, py2, pz2 = point2
	
    r1 = [x1, y1, z1]
    r2 = [x2, y2, z2]
    p1 = [px1, py1, pz1]
    p2 = [px2, py2, pz2]

	# Projection of the momenta on the plane that is perpendicular to the magnetic field
    B_e = B / norm(B)
	p1_par = dot(p1, B_e) * B_e
    p2_par = dot(p2, B_e) * B_e
    p1_perp = p1 - p1_par
    p2_perp = p2 - p2_par

	# Calculation of the distance between points in the plane that is perpendicular to the magnetic field
	d = r2 - r1
	d_perp = d - dot(d, B_e) * B_e

    cos_theta = dot(p1_perp, p2_perp) / (norm(p1_perp) * norm(p2_perp))

    R = norm(d_perp) / (2 * sqrt(abs(1 - abs(cos_theta)) / 2))
	p = sqrt(norm(p2_par)^2 + (q * norm(B) * R)^2)
	p = uconvert(u"GeV/c", p)


    return R, p
end

# ╔═╡ fff49acf-28ba-4ca4-8f47-e07dd32f446e
# Function to calculate the radius of the circle given two state vectors
function radius_of_circle(point1, point2)
    t1, x1, y1, z1, E1, px1, py1, pz1 = point1
    t2, x2, y2, z2, E2, px2, py2, pz2 = point2

    d = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

    p1 = [px1, py1, pz1]
    p2 = [px2, py2, pz2]

    p1_magnitude = norm(p1)
    p2_magnitude = norm(p2)

    cos_theta = dot(p1, p2) / (p1_magnitude * p2_magnitude)

    R = d / (2 * abs(cos_theta))

    return R
end

# ╔═╡ 66832ec3-d47c-4ac6-b5ad-f8625a03d5b5
# Function to compute momentum based on the radius of the circle through two points 
function momentum_from_radius(R, q, B)
	Bx, By, Bz = B
    B = sqrt(Bx^2 + By^2 + Bz^2)

    # Calculate the momentum p using p = qBr
    p = q * B * R
	p = uconvert(u"GeV/c", p)
	
    return p
end

# ╔═╡ 57ce7bb6-ad24-44e3-8a02-460078a3325e
const m_mu = 0.1057u"GeV/c^2"

# ╔═╡ dc286863-bf42-40e0-974b-e3f229dcf2d5
# Function to compute the momenta of two muons from the J/psi decay
function jpsi_to_mumu(m_jpsi)
    E_mu = m_jpsi*c_1^2 / 2  # Energy is half the invariant mass of J/psi
    p_mu = sqrt(E_mu^2 - c_1^4*m_mu^2) / c_1
    cos_theta = rand(Uniform(-1.0, 1.0))  # cos(θ) is uniformly distributed between -1 and 1
    theta = acos(cos_theta)
    phi = rand(Uniform(-π, π))  # φ is uniformly sampled between -π and π
	
    px = p_mu * sin(theta) * cos(phi)
    py = p_mu * sin(theta) * sin(phi)
    pz = p_mu * cos(theta)
    
    p1 = [uconvert(u"GeV", E_mu), uconvert(u"GeV/c", px), uconvert(u"GeV/c", py), uconvert(u"GeV/c", pz)]
    p2 = [uconvert(u"GeV", E_mu), uconvert(u"GeV/c", -px), uconvert(u"GeV/c", -py), uconvert(u"GeV/c", -pz)]
    
    charge_muon_1 = rand(Bernoulli(0.5)) == 1 ? 1 : -1 # Randomly assign sign of muon
    charge_muon_2 = -charge_muon_1  

    if charge_muon_1 == -1 # Always return nagatively charged muon first
        return p1, p2  
    else
        return p2, p1
    end
end

# ╔═╡ 3ba42e77-8b8e-49d7-9def-ad35a56bea19
# Function to compute the invariant mass of J/psi from two muon 4-momenta
function jpsi_from_mumu(p1, p2)
    
    E = p1[1] + p2[1]
    px = p1[2] + p2[2]
    py = p1[3] + p2[3]
    pz = p1[4] + p2[4]

    M_mumu = sqrt(E^2 - c_1^2*(px^2 + py^2 + pz^2)) / c_1^2
	
    return M_mumu
end

# ╔═╡ 230f7f9a-d2b8-4e56-ae57-5e1aafa41678
# Simulation of the J/psi decay to two muons and reconstraction of J/psi based on tracks of muons in the magnetic field
begin
	Nevents = 10
	tmu0, xmu0, ymu0, zmu0 = 0.0u"ns", 0.0u"m", 0.0u"m", 0.0u"m"
	tmumax, dtmu = 1.0u"ns", 1.0u"ns"
	E_exp = [0.0u"V/m", 0.0u"V/m", 0.0u"V/m"]
	B_exp = [0.0u"T", 1.0u"T", 0.0u"T"]
	par_minus = [-e, m_mu, E_exp, B_exp]
	par_plus = [e, m_mu, E_exp, B_exp]
	
	M_mumu = sample_values(cdf_values, bin_positions, Nevents) ### Simulate invariant mass of the J/psi using function from the Exersice 2
	M_mumu .= M_mumu .* 1.0u"GeV/c^2" ./ 1000.0
	M_mumu_sim = []
	pmu_minus = []
	pmu_plus = []
	pmu_minus_sim = []
	pmu_plus_sim = []
	
	
	for M in M_mumu

	### Using functions provided in this exersice, implement simulation of the J/psi decay into two muons, simulate trajectory of the muons with Runge-Kutta method and reconstruct muons momenta. Fill all the arrays, initialized above. Compare reconstructed momenta with initial.
	p_minus, p_plus = jpsi_to_mumu(M)

	push!(M_mumu_sim, jpsi_from_mumu(p_minus, p_plus))

	push!(pmu_minus, norm(p_minus[2:4]))
	push!(pmu_plus, norm(p_plus[2:4]))

	r0_minus = [tmu0, xmu0, ymu0, zmu0, p_minus[1], p_minus[2], p_minus[3], p_minus[4]]
	r0_plus  = [tmu0, xmu0, ymu0, zmu0, p_plus[1],  p_plus[2],  p_plus[3],  p_plus[4]]

	rs_minus = runge_kutta_4(eom, r0_minus, par_minus, tmumax, dtmu)
	rs_plus  = runge_kutta_4(eom, r0_plus,  par_plus,  tmumax, dtmu)
		
	
		
	R_minus, p_minus_rec = momentum_from_circle(rs_minus[1], rs_minus[2], e, B_exp)
	R_plus,  p_plus_rec  = momentum_from_circle(rs_plus[1],  rs_plus[2],  e, B_exp)

	push!(pmu_minus_sim, p_minus_rec)
	push!(pmu_plus_sim, p_plus_rec)
	
	end
end

# ╔═╡ c380b32e-a8c9-470c-952d-4ecb5ed5dd09
begin
	rel_err_minus = abs.(pmu_minus .- pmu_minus_sim) ./ pmu_minus
	rel_err_plus  = abs.(pmu_plus  .- pmu_plus_sim)  ./ pmu_plus

	println("Muon minus:")
	println("initial p = ", pmu_minus)
	println("reconstructed p = ", pmu_minus_sim)
	println("relative error = ", rel_err_minus)

	println("\nMuon plus:")
	println("initial p = ", pmu_plus)
	println("reconstructed p = ", pmu_plus_sim)
	println("relative error = ", rel_err_plus)
end

# ╔═╡ d9b49708-85ab-4a84-a1bb-11de29785309
begin
	M_mumu_reco = []

	for i in 1:Nevents
		p_minus = pmu_minus_sim[i]
		p_plus  = pmu_plus_sim[i]

		E_minus = sqrt(c_1^2 * p_minus^2 + c_1^4 * m_mu^2)
		E_plus  = sqrt(c_1^2 * p_plus^2  + c_1^4 * m_mu^2)

		M_reco = (E_minus + E_plus) / c_1^2
		push!(M_mumu_reco, uconvert(u"GeV/c^2", M_reco))
	end

	M_mumu_reco
end

# ╔═╡ 901582fb-35a1-492c-8132-c9c809d351e6
begin
	mass_rel_err = abs.(M_mumu .- M_mumu_reco) ./ M_mumu

	println("Generated J/ψ masses:")
	println(M_mumu)

	println("\nReconstructed J/ψ masses:")
	println(M_mumu_reco)

	println("\nRelative mass error:")
	println(mass_rel_err)
end

# ╔═╡ 0b0d3e30-2995-4f51-985c-740f2b36030f
begin
	println("mean relative error = ", mean(mass_rel_err))
	println("max relative error = ", maximum(mass_rel_err))
end

# ╔═╡ aaa535c9-55ee-432e-b41d-63a8a59e47f6
begin
	M_gen = [ustrip(u"GeV/c^2", m) for m in M_mumu]
	M_rec = [ustrip(u"GeV/c^2", m) for m in M_mumu_reco]

	plt= plot(
		1:Nevents,
		M_gen,
		label = "generated",
		xlabel = "Event",
		ylabel = "M(J/ψ) [GeV/c²]",
		title = "Generated vs reconstructed J/ψ mass",
		marker = :circle
	)

	plot!(plt,
		1:Nevents,
		M_rec,
		label = "reconstructed",
		marker = :diamond
	)


end

# ╔═╡ 655a0eb4-840e-4e51-936d-8204c6946e85
begin
	
	u=histogram(
		M_gen,
		bins = 10,
		alpha = 0.5,
		label = "generated",
		xlabel = "M(J/ψ) [GeV/c²]",
		ylabel = "Counts",
		title = "Generated vs reconstructed J/ψ mass"
	)

	histogram!(
		M_rec,
		bins = 10,
		alpha = 0.5,
		label = "reconstructed"
	)
end

# ╔═╡ 52cccc36-56c3-4784-b9bb-763cc5cbfaba
begin
	tmumax_plot = 9.0u"ns"
	dtmu_plot = 0.05u"ns"

	M = mean(M_mumu)
	p_minus, p_plus = jpsi_to_mumu(M)

	r0_minus = [tmu0, xmu0, ymu0, zmu0, p_minus[1], p_minus[2], p_minus[3], p_minus[4]]
	r0_plus  = [tmu0, xmu0, ymu0, zmu0, p_plus[1],  p_plus[2],  p_plus[3],  p_plus[4]]

	rs_minus = runge_kutta_4(eom, r0_minus, par_minus, tmumax_plot, dtmu_plot)
	rs_plus  = runge_kutta_4(eom, r0_plus,  par_plus,  tmumax_plot, dtmu_plot)

	x_minus = [r[2] for r in rs_minus]
	y_minus = [r[3] for r in rs_minus]
	z_minus = [r[4] for r in rs_minus]

	x_plus = [r[2] for r in rs_plus]
	y_plus = [r[3] for r in rs_plus]
	z_plus = [r[4] for r in rs_plus]

	mu= plot(
		x_minus,
		y_minus,
		z_minus,
		label = "μ⁻",
		xlabel = "x",
		ylabel = "y",
		zlabel = "z",
		title = "Muon trajectories in magnetic field",
		lw = 2
	)

	plot!(
		x_plus,
		y_plus,
		z_plus,
		label = "μ⁺",
		lw = 2
	)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LsqFit = "2fda8390-95c7-5789-9bda-21331edee243"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
CSV = "~0.10.14"
DataFrames = "~1.7.0"
Distributions = "~0.25.112"
LsqFit = "~0.16.0"
Plots = "~1.40.8"
PlutoUI = "~0.7.59"
Unitful = "~1.21.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "35d101232238ff3b96f3a4f1569e73d7b10017de"

[[deps.ADTypes]]
git-tree-sha1 = "bbc22a9a08a0ef6460041086d8a7b27940ed4ffd"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.22.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "54f895554d05c83e3dd59f6a396671dae8999573"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.24.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "d0250552e42bf7cc36479fd38a6e30004c9e8c2b"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.17"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d7477ecdafb813ddee2ae727afa94e9dcb5f3fb0"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.112"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f7017a4f337f8df189fcce98e32b67a1298a2115"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.31.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "629693584cef594c3f6f99e76e7a7ad17e60e8d5"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a8863b69c2a0859f2c2c87ebdc4c6712e88bdf0d"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "7c4195be1649ae622304031ed46a2f4df989f1eb"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.24"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LsqFit]]
deps = ["ADTypes", "Distributions", "ForwardDiff", "LinearAlgebra", "NLSolversBase", "Printf", "StatsAPI"]
git-tree-sha1 = "938aaa27db65e619e19aadd58fbae44fbb0d83e7"
uuid = "2fda8390-95c7-5789-9bda-21331edee243"
version = "0.16.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0eef589dd1c26a3ac9d753fe1a8bcad63f956fa6"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.16.8+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "b3f76b463c7998473062992b246045e6961a074e"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═ac552774-8faa-11ef-257f-ad04db824377
# ╟─efdd6d2d-2f5b-49a6-910b-098899bc59e3
# ╠═33699dc6-4d43-480f-905e-2674db611b22
# ╠═cc9d7ea9-91ce-472c-abfe-a67d8d55403e
# ╠═0aee2d4a-42fb-4569-b199-d7535d811baa
# ╠═884fc9f1-139a-49c4-bb0e-735715db0e84
# ╠═4173869b-55f8-458f-8789-af44d022ebfd
# ╠═acf9c865-02a4-488c-b0b2-11c9adbe4f83
# ╠═096a3b73-bdce-4a38-9ce0-bb186601c2e4
# ╠═01ffa2e0-d25d-4694-aed7-537b4535e1ab
# ╠═224af2f9-e334-4289-8890-46393cc193bc
# ╠═f99daddf-bc57-427d-a87c-3b9da7d116f9
# ╠═96002c68-50b6-409b-ad31-c2a7967414b0
# ╠═5a3cdea5-5140-43cb-a609-ae96161967ce
# ╠═0e9d6cb1-e037-46e1-908d-a67850b34139
# ╠═f4c4ceb6-a68f-47e0-930e-96dc91232567
# ╠═63bfaaf3-645d-4b33-95bf-9c2f4c65b2a0
# ╠═042212eb-684d-446b-9785-3c127ad9f732
# ╠═39481270-26dc-4eb3-9fd1-d83ac1f4bf03
# ╠═ddefd433-a462-4fd9-807f-8e3a45cf07f5
# ╠═01e2eca6-ae5c-4bf1-a22a-cea64b869d9b
# ╠═5d352f88-8d33-4d0d-ad57-9bbb99773831
# ╠═c4813601-5633-46f0-9a7a-bc0edc62e364
# ╠═7df5fae0-614e-4c36-a073-680e7aaec332
# ╠═459b7567-d5fc-464d-9561-3c5141469df9
# ╠═3567a205-65fb-4a28-a4f1-4334587e510d
# ╠═93a82374-9227-401e-b875-c453e08ddafa
# ╠═d339e66c-6c43-46fd-af99-dc7a8eb6520d
# ╠═b85856c4-b722-4f0c-a3f3-7566938173da
# ╠═1b9921eb-746b-4fdf-b5e2-f83c87c989e5
# ╠═b0b544b1-e26c-49d8-87cf-1002142534c7
# ╠═7c274fb6-ccfe-484e-9c5b-321225186cb6
# ╠═d2c1d638-f2c1-4834-9235-a9d55dbaa866
# ╠═6bb3a02f-4882-4626-9417-f3aa07521eb6
# ╠═9431493e-8a10-4af6-904b-c1a03625932c
# ╠═9d1ebe41-42bc-40c3-91f8-10a5d9707b66
# ╠═33074b6a-c172-4544-a575-0649e7664632
# ╟─a3ea2044-932e-44c4-b5c6-77fdf355333f
# ╟─c13d949e-ebff-42bf-9d10-751b79e2a764
# ╠═c5ce4cb8-36be-42cd-87e4-d371f1220075
# ╟─9a3f63ca-6dd8-410d-84fc-33f9a88f011b
# ╟─a86ea674-31a0-4285-9d33-84ffbdedeabd
# ╠═2e8ccf97-d9f8-4683-8ad6-c9e2cdbbf80c
# ╠═a69583de-164d-40c3-9562-134cdd19f6c5
# ╠═2467d8b6-0619-490f-baa2-da6a0fc8e99e
# ╠═fff49acf-28ba-4ca4-8f47-e07dd32f446e
# ╠═66832ec3-d47c-4ac6-b5ad-f8625a03d5b5
# ╠═57ce7bb6-ad24-44e3-8a02-460078a3325e
# ╠═dc286863-bf42-40e0-974b-e3f229dcf2d5
# ╠═3ba42e77-8b8e-49d7-9def-ad35a56bea19
# ╠═230f7f9a-d2b8-4e56-ae57-5e1aafa41678
# ╠═c380b32e-a8c9-470c-952d-4ecb5ed5dd09
# ╠═d9b49708-85ab-4a84-a1bb-11de29785309
# ╠═901582fb-35a1-492c-8132-c9c809d351e6
# ╠═0b0d3e30-2995-4f51-985c-740f2b36030f
# ╠═aaa535c9-55ee-432e-b41d-63a8a59e47f6
# ╠═655a0eb4-840e-4e51-936d-8204c6946e85
# ╠═52cccc36-56c3-4784-b9bb-763cc5cbfaba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
