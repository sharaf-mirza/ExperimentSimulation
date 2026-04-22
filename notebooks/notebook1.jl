### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ e42debc4-3e34-11f1-8733-4d76af361c73
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))

	Pkg.instantiate()
end

# ╔═╡ 74d91fdb-e6c1-4829-ada1-0f7a67d74097
using Plots, Random

# ╔═╡ 2447f0ea-3eb7-4e51-b454-11fd79873cc7
using Test

# ╔═╡ 970789d4-0a1d-44a9-b9ce-072656a1ca61
@testset "Fake test" begin
	@test 1 == 1
end

# ╔═╡ 21393fb7-90d2-4a09-9b44-ae3a8deda599


# ╔═╡ Cell order:
# ╠═e42debc4-3e34-11f1-8733-4d76af361c73
# ╠═74d91fdb-e6c1-4829-ada1-0f7a67d74097
# ╠═2447f0ea-3eb7-4e51-b454-11fd79873cc7
# ╠═970789d4-0a1d-44a9-b9ce-072656a1ca61
# ╠═21393fb7-90d2-4a09-9b44-ae3a8deda599
