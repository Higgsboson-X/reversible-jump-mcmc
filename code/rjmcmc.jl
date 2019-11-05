using Distributions
using StatsBase

include("./utils.jl")

function rjmcmc(y, hyperparams, kmax, epochs, p_split = 0.5, p_birth = 0.5)

	n = length(y);
	params = initialize_params(y, hyperparams);
	println("[params] ", params.k, ", ", params.μ, params.σ²);

	for epoch in 1:epochs
		println("epoch - ", epoch, ", k = ", params.k);
		δ_vec = collect(range(hyperparams.δ, hyperparams.δ, length = params.k));
		c_n = get_comp_counts(params.z, params.k);

		# gibbs sampling;
		println("\tgibbs sampling ...");
		# w;
		f_w = Dirichlet(δ_vec .+ c_n);
		params.w = rand(f_w);
		# μⱼ;
		for j in 1:params.k
			y_j = y[params.z .== j];

			m = (sum(y_j)/params.σ²[j] + hyperparams.κ*hyperparams.ξ) / (c_n[j]/params.σ²[j] + hyperparams.κ);
			s = 1 / (c_n[j]/params.σ²[j] + hyperparams.κ);
			f_μ = Normal(m, s);
			params.μ[j] = rand(f_μ);

			a = hyperparams.α + c_n[j]/2;
			b = params.β + sum((y_j .- params.μ[j]).^2) / 2;
			f_σ² = Gamma(a, 1/b);
			params.σ²[j] = 1 / rand(f_σ²);
		end
		# zᵢ;
		params.z, _ = get_allocation(params.μ, params.σ², params.w, y);
		# β;
		a = hyperparams.g + params.k * hyperparams.α;
		b = hyperparams.h + sum(1 ./ params.σ²);
		f_β = Gamma(a, 1/b);
		params.β = rand(f_β);

		# splitting/combining;
		p = rand();
		if p < p_split || params.k == 1
			# split;
			params = split_comp(hyperparams, params, y, kmax);
			# println(params.k, ", ", length(params.μ), ", ", length(params.σ²), ", ", length(params.μ));
		else
			# combine;
			params = combine_comp(hyperparams, params, y, kmax);
			# println(params.k, ", ", length(params.μ), ", ", length(params.σ²), ", ", length(params.μ));
		end
		# birth/death;
		p = rand();
		if p < p_birth || params.k == 1
			# birth;
			params = birth_comp(hyperparams, params, y, kmax);
			# println(params.k, ", ", length(params.μ), ", ", length(params.σ²), ", ", length(params.μ));
		else
			# death;
			params = death_comp(hyperparams, params, y, kmax);
			# println(params.k, ", ", length(params.μ), ", ", length(params.σ²), ", ", length(params.μ));
		end

	end

	return params;

end
