"""Breit-Wigner signal plus linear background, with `p = [A, m0, Γ, a, b]`."""
function model(x, p)
    A, m0, Γ, a, b = p

    signal =
        A .* ((Γ^2 / 4) ./ ((x .- m0).^2 .+ Γ^2 / 4))

    background =
        a .+ b .* x

    return signal .+ background
end

function hit_and_miss_continuous(fit_function, xfit, yfit, num_samples=100000; rng=Random.default_rng())

    x_min = minimum(xfit)
    x_max = maximum(xfit)
    ymax  = maximum(yfit)

    accepted_samples = Float64[]

    for i in 1:num_samples

        # sample continuous x
        x = rand(rng) * (x_max - x_min) + x_min

        # sample y
        y = rand(rng) * ymax

        # acceptance-rejection
        if y <= fit_function(x)
            push!(accepted_samples, x)
        end
    end

    return accepted_samples
end
