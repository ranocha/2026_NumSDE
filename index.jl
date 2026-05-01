### A Pluto.jl notebook ###
# v0.20.23 

using Markdown
using InteractiveUtils

# ╔═╡ e6c64c80-773b-11ef-2379-bf6609137e69
md"""
# Numerik stochastischer Differentialgleichungen

### Sommersemester 2026
### Johannes Gutenberg-Universität Mainz
### Prof. Dr. Hendrik Ranocha
"""

# ╔═╡ 3bf1c95f-4a14-4a23-ba93-cf9bb26cb41e
let
	repo = "2026_NumSDE"
	url = "https://ranocha.de/" * repo * "/"

	notebooks = String[]
	for name in readdir(@__DIR__)
		full_name = joinpath(@__DIR__, name)
		if isfile(full_name) && endswith(name, ".jl") &&
								startswith(name, r"\d")
			push!(notebooks, name)
		end
	end

	text = """Hier finden Sie eine Liste der statischen Notebooks zur Vorlesung.
			  Um die Notebooks dynamisch verwenden zu können müssen Sie Julia
			  lokal installieren wie in der README.md des
			  [Repositories](https://github.com/ranocha/$(repo)) beschrieben."""
	for name in notebooks
		file = read(name, String)
		# pattern = r"md\"\"\"\n# (\d+\.\d+ [^\n]+)"
		pattern = r"md\"\"\"\n# (\d+ [^\n]+)"
		m = match(pattern, file)
		m === nothing && continue
		title = m.captures[1]

		text = text * "\n- [`" * title * "`](" *
				url * name[begin:end-3] * ".html)"
	end

	Markdown.parse(text)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.10"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─e6c64c80-773b-11ef-2379-bf6609137e69
# ╟─3bf1c95f-4a14-4a23-ba93-cf9bb26cb41e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
