"""Return bin counts from `num_samples` hit-and-miss proposals; accepts an optional `rng`."""
function hit_and_miss_sampling(bin_contents, bin_positions, num_samples; rng=Random.default_rng())
    max_bin_content = maximum(bin_contents)
    accepted_samples_hm = zeros(lastindex(bin_positions))
    for i in 1:num_samples
        random_x = rand(rng, eachindex(bin_positions))
        random_y = rand(rng) * max_bin_content

        if random_y <= bin_contents[random_x]
            accepted_samples_hm[random_x] += 1
        end
    end
    return accepted_samples_hm
end

"""Sample bin counts from a normalized, increasing CDF ending at one."""
function inverse_cdf_sampling(cdf_values, bin_positions, num_samples; rng=Random.default_rng())
    accepted_samples_inv = zeros(lastindex(bin_positions))
    for i in 1:num_samples

        u= rand(rng)
        accepted_samples_inv[findfirst(>=(u), cdf_values)] = accepted_samples_inv[findfirst(>=(u), cdf_values)] + 1
    end
    return accepted_samples_inv
end

"""Draw `events` bin-position values from a normalized CDF; accepts an optional `rng`."""
function sample_values(cdf_values, bin_positions, events; rng=Random.default_rng())
    values = []
    for i in 1:events
        u = rand(rng)
        push!(values, copy(bin_positions[findfirst(>=(u), cdf_values)]))
    end
    return values
end

"""Legacy sum-of-squared-residuals comparison; this is not a statistical chi-square."""
function chi2(sample, data)

    if length(sample) != length(data)
        throw(ArgumentError("Histograms must have the same number of bins"))
    end

    chi2_values = (data .- sample).^2
    sum(chi2_values)
end

function efficiency(sample, num_samples)
    return sum(sample)/num_samples
end
