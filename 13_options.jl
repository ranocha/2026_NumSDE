### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 3e85ee8d-8a09-43d2-965a-846e4e27606c
using Random

# ╔═╡ c7895bb8-c1e5-4135-8223-729c4acaff14
begin
	using OnlineStats
	using OnlineStats: Series
end

# ╔═╡ 60f3b695-dd13-442a-8a15-b57cac2e3d93
using SpecialFunctions

# ╔═╡ ce2411d2-81fb-11ee-0441-7d4bb926681f
md"""
# 13 Options

"""

# ╔═╡ e5481b55-c8b7-4271-bec2-5afabd3da3f5
md"""
#### Initializing packages

_When running this notebook for the first time, this could take up several minutes. Hang in there!_
"""

# ╔═╡ f59d7e98-a4ee-4a36-b899-e4e9faa520fd
md"""
## Expected discounted payoff for a European call option
"""

# ╔═╡ 816c2a70-7f37-4cdb-a6bd-baaf1a106493
f(X, parameters, t) = parameters.r * X

# ╔═╡ a6095158-68a2-490f-8eda-d43ba54217a1
g(X, parameters, t) = parameters.σ * X

# ╔═╡ 162e27a1-08bb-46ce-818d-9ec0c3abd91e
h(X, parameters, t) = exp(-parameters.r * t) * max(X - parameters.E, 0)

# ╔═╡ f4aad700-4c94-4766-97e8-f61ef4b3163e
function euler_maruyama(f, g, h, early_exit, X0, parameters, tspan; N, M)
	Δt = (last(tspan) - first(tspan)) / N
	sqrt_Δt = sqrt(Δt)

	stats = Series(Mean(), Variance())
	for _ in 1:M
		exited_early = false
		t = first(tspan)
		X = X0
		for _ in 1:N
			ΔW = sqrt_Δt * randn()
			X += f(X, parameters, t) * Δt + g(X, parameters, t) * ΔW
			t += Δt
			if early_exit(X, parameters, t)
				exited_early = true
				break
			end
		end

		if exited_early
			fit!(stats, 0)
		else
			@assert t ≈ last(tspan)
			fit!(stats, h(X, parameters, t))
		end
	end

	μ, σ = value(stats)
	c = 1.96 * σ / sqrt(M)
	confidence = (μ - c, μ + c)
	return (; confidence, mean = μ, variance = σ)
end

# ╔═╡ 1cd6ff23-cf74-47c2-819c-f1e552364c41
md"""
## Expected discounted payoff for a European up-and-out call option
"""

# ╔═╡ b4473678-7b0d-4fc6-893d-daf728ddf7db
early_exit_up_and_out(X, parameters, t) = X > parameters.B

# ╔═╡ 25ee9b29-f263-4cef-aedb-02108600f471
function european_up_and_out_call(X0, parameters, tspan)
	(; r, σ, E, B) = parameters
	T = last(tspan)
	
	power1 = -1 + 2 * r / σ^2
	power2 = 1 + 2 * r / σ^2
	d1 = (log(X0 / E) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
	d2 = d1 - σ * sqrt(T)
  	e1 = (log(X0 / B) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
	e2 = (log(X0 / B) + (r - 0.5 * σ^2) * T) / (σ * sqrt(T))
	f1 = (log(X0 / B) - (r - 0.5 * σ^2) * T) / (σ * sqrt(T))
	f2 = (log(X0 / B) - (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
	g1 = (log(X0 * E / B^2) - (r - 0.5 * σ^2) * T) / (σ * sqrt(T))
	g2 = (log(X0 * E / B^2) - (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
	Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
	Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
	Ne1 = 0.5 * (1 + erf(e1 / sqrt(2)))
	Ne2 = 0.5 * (1 + erf(e2 / sqrt(2)))
	Nf1 = 0.5 * (1 + erf(f1 / sqrt(2)))
	Nf2 = 0.5 * (1 + erf(f2 / sqrt(2)))
	Ng1 = 0.5 * (1 + erf(g1 / sqrt(2)))
	Ng2 = 0.5 * (1 + erf(g2 / sqrt(2)))
	a = (B / X0)^power1
	b = (B / X0)^power2
	return X0 * (Nd1 - Ne1 - b * (Nf2 - Ng2)) - E * exp(-r * T) * (Nd2 - Ne2 - a * (Nf1 - Ng1))
end

# ╔═╡ 173bedfe-f17e-48e9-aaf0-73792eeca35f
let
	Random.seed!(42)
	parameters = (r = 0.05, σ = 0.25, E = 6.0, B = 1.0e16)
	X0 = 5.0
	tspan = (0.0, 1.0)

	results = euler_maruyama(f, g, h, Returns(false),
							 X0, parameters, tspan; 
				   			 N = 10^3, M = 10^5)
	value = european_up_and_out_call(X0, parameters, tspan)
	(; value, results...)
end

# ╔═╡ 32857d64-eb06-4c89-aee8-4718b92e3271
let
	Random.seed!(42)
	parameters = (r = 0.05, σ = 0.25, E = 6.0, B = 9.0)
	X0 = 5.0
	tspan = (0.0, 1.0)

	results = euler_maruyama(f, g, h, early_exit_up_and_out,
							 X0, parameters, tspan; 
				   			 N = 10^3, M = 10^5)
	value = european_up_and_out_call(X0, parameters, tspan)
	(; value, results...)
end

# ╔═╡ 53cb9639-3e9d-45b2-8ef5-781c244a541a
md"""
## Appendix
"""

# ╔═╡ 27d0751b-539a-4547-8fa2-f78565afe8e2
space = html"<br><br><br>";

# ╔═╡ 87faa1d6-4a30-4905-9ca6-07f115003407
space

# ╔═╡ dac5abab-5699-44ee-bc2e-5e236d800b71
space

# ╔═╡ cceeeef3-e772-481d-843f-f8a7febca971
space

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
OnlineStats = "a15396b6-48d5-5d58-9928-6d29437db91e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
OnlineStats = "~1.7.3"
SpecialFunctions = "~2.7.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.11"
manifest_format = "2.0"
project_hash = "001e3b07f76d4b4cfbce09518e07abd5fb6f44c2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e421c1938fafab0165b04dc1a9dbe2a26272952c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.125"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.OnlineStats]]
deps = ["AbstractTrees", "Dates", "Distributions", "LinearAlgebra", "OnlineStatsBase", "OrderedCollections", "Random", "RecipesBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a19ba00b3968afcde81471033fd22976e62f5fea"
uuid = "a15396b6-48d5-5d58-9928-6d29437db91e"
version = "1.7.3"

[[deps.OnlineStatsBase]]
deps = ["AbstractTrees", "Dates", "LinearAlgebra", "OrderedCollections", "Statistics", "StatsBase"]
git-tree-sha1 = "a5a5a68d079ce531b0220e99789e0c1c8c5ed215"
uuid = "925886fa-5bf2-5e8e-b522-a9147a512338"
version = "1.7.1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+5"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─ce2411d2-81fb-11ee-0441-7d4bb926681f
# ╟─e5481b55-c8b7-4271-bec2-5afabd3da3f5
# ╠═3e85ee8d-8a09-43d2-965a-846e4e27606c
# ╠═c7895bb8-c1e5-4135-8223-729c4acaff14
# ╠═60f3b695-dd13-442a-8a15-b57cac2e3d93
# ╟─f59d7e98-a4ee-4a36-b899-e4e9faa520fd
# ╠═816c2a70-7f37-4cdb-a6bd-baaf1a106493
# ╠═a6095158-68a2-490f-8eda-d43ba54217a1
# ╠═162e27a1-08bb-46ce-818d-9ec0c3abd91e
# ╠═f4aad700-4c94-4766-97e8-f61ef4b3163e
# ╠═173bedfe-f17e-48e9-aaf0-73792eeca35f
# ╟─1cd6ff23-cf74-47c2-819c-f1e552364c41
# ╠═b4473678-7b0d-4fc6-893d-daf728ddf7db
# ╠═32857d64-eb06-4c89-aee8-4718b92e3271
# ╠═25ee9b29-f263-4cef-aedb-02108600f471
# ╟─87faa1d6-4a30-4905-9ca6-07f115003407
# ╟─dac5abab-5699-44ee-bc2e-5e236d800b71
# ╟─cceeeef3-e772-481d-843f-f8a7febca971
# ╟─53cb9639-3e9d-45b2-8ef5-781c244a541a
# ╠═27d0751b-539a-4547-8fa2-f78565afe8e2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
