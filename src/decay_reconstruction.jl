function four_momentum(m, p, cos_theta, phi)
    theta = acos(cos_theta)

    px = p * sin(theta) * cos(phi)
    py = p * sin(theta) * sin(phi)
    pz = p * cos(theta)
    E = sqrt(c_1^2*p^2 + c_1^4*m^2)

    # Return the 4-momentum vector
    return E, px, py, pz
end

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

function momentum_from_radius(R, q, B)
    Bx, By, Bz = B
    B = sqrt(Bx^2 + By^2 + Bz^2)

    # Calculate the momentum p using p = qBr
    p = q * B * R
    p = uconvert(u"GeV/c", p)

    return p
end

"""Generate back-to-back muon four-momenta for a J/psi at rest, with optional `rng`."""
function jpsi_to_mumu(m_jpsi; rng=Random.default_rng())
    E_mu = m_jpsi*c_1^2 / 2  # Energy is half the invariant mass of J/psi
    p_mu = sqrt(E_mu^2 - c_1^4*m_mu^2) / c_1
    cos_theta = rand(rng, Uniform(-1.0, 1.0))  # cos(θ) is uniformly distributed between -1 and 1
    theta = acos(cos_theta)
    phi = rand(rng, Uniform(-π, π))  # φ is uniformly sampled between -π and π

    px = p_mu * sin(theta) * cos(phi)
    py = p_mu * sin(theta) * sin(phi)
    pz = p_mu * cos(theta)

    p1 = [uconvert(u"GeV", E_mu), uconvert(u"GeV/c", px), uconvert(u"GeV/c", py), uconvert(u"GeV/c", pz)]
    p2 = [uconvert(u"GeV", E_mu), uconvert(u"GeV/c", -px), uconvert(u"GeV/c", -py), uconvert(u"GeV/c", -pz)]

    charge_muon_1 = rand(rng, Bernoulli(0.5)) == 1 ? 1 : -1 # Randomly assign sign of muon
    charge_muon_2 = -charge_muon_1

    if charge_muon_1 == -1 # Always return nagatively charged muon first
        return p1, p2
    else
        return p2, p1
    end
end

function jpsi_from_mumu(p1, p2)

    E = p1[1] + p2[1]
    px = p1[2] + p2[2]
    py = p1[3] + p2[3]
    pz = p1[4] + p2[4]

    M_mumu = sqrt(E^2 - c_1^2*(px^2 + py^2 + pz^2)) / c_1^2

    return M_mumu
end
