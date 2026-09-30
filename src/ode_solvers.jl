"""
    eom(r, par)

Return derivatives for the lab's state `[t, x, y, z, energy, px, py, pz]`,
with parameters `[charge, mass, electric_field, magnetic_field]`.
State and field values use Unitful quantities. The lab's velocity model is retained.
"""
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

# Construct a numeric range in one time unit. Unitful ranges with mixed units
# can round down and silently omit a step even for an exact multiple of dt.
function _time_steps(r0, tmax, dt)
    dt > zero(dt) || throw(ArgumentError("dt must be positive"))
    tmax >= r0[1] || throw(ArgumentError("tmax must not precede the initial time"))
    times = range(ustrip(unit(dt), r0[1]); step=ustrip(dt), stop=ustrip(unit(dt), tmax))
    return Iterators.drop(times, 1)
end

function euler(f, r0, par, tmax, dt)
    ts = _time_steps(r0, tmax, dt)
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory

    # Iterate through each time step
    for _ in ts
        dimensions(r)
        push!(rs, copy(r))

        r = r .+ dt .* f(r, par)
    end

    dimensions(r)
    push!(rs, copy(r))
    return rs
end

function predictor_corrector(f, r0, par, tmax, dt)
    ts = _time_steps(r0, tmax, dt)
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory

    # Iterate through each time step
    for _ in ts
        dimensions(r)
        push!(rs, copy(r))

        r_pred = r .+ dt * f(r, par)

        r = r .+ 0.5 .* dt .* (f(r, par) .+ f(r_pred, par))
    end

    dimensions(r)
    push!(rs, copy(r))
    return rs
end

function runge_kutta_4(f, r0, par, tmax, dt)
    ts = _time_steps(r0, tmax, dt)
    r = copy(r0)     # Initial condition
    rs = []          # To store the trajectory

    # Iterate through each time step
    for _ in ts
        dimensions(r)
        push!(rs, copy(r))

        k1 = dt * f(r, par)
        k2 = dt * f(r .+ 0.5 .* k1, par)
        k3 = dt * f(r .+ 0.5 .* k2, par)
        k4 = dt * f(r .+ k3, par)

        r = r .+ (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4) ./ 6
    end

    dimensions(r)
    push!(rs, copy(r))
    return rs
end
