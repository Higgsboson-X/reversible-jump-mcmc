using Distributions
using SpecialFunctions

include("./types.jl")

function get_default_hyperparams(y)

	ξ = mean(y);
	κ = var(y);
	α = 2;
	g = 0.5;
	h = 1;
	λ = 3;
	δ = 1;
	hyperparams = Types.HyperParams(ξ, κ, α, g, h, λ, δ);

	return hyperparams;

end


function initialize_params(y, hyperparams)

	# y: data;
	n = length(y);

	# 1;
	f_k = Poisson(hyperparams.λ);
	k = rand(f_k);
	while k == 0
		k = rand(f_k);
	end
	# k = n;

	# 1;
	f_β = Gamma(hyperparams.g, 1/hyperparams.h);
	β = rand(f_β);

	# k;
	f_σ² = Gamma(hyperparams.α, 1/β);
	σ² = 1 ./ rand(f_σ², k);

	# k;
	f_μ = Normal(hyperparams.ξ, 1 / hyperparams.κ);
	μ = rand(f_μ, k);

	# k;
	f_w = Dirichlet(collect(range(hyperparams.δ, hyperparams.δ, length = k)));
	w = rand(f_w);

	z, _ = get_allocation(μ, σ², w, y);

	params = Types.Params(μ, σ², β, k, z, w);

	return params;

end

function get_allocation(μ, σ², w, y)

	k = length(μ);
	n = length(y);

	z = zeros(Int64, n);
	p_alloc = zeros(Float64, n);
	for i in 1:n
		p = w ./ sqrt.(σ²) .* exp.(-(y[i] .- μ).^2 ./ (2σ²));
		z[i] = argmax(p);
		p_alloc[i] = z[i] / sum(p);
	end

	return z, prod(p_alloc);

end


function calc_p_alloc(μ, σ², w, y, z)

	k = length(μ);
	n = length(y);

	p_alloc = zeros(Float64, n);
	for i in 1:n
		p = w ./ sqrt.(σ²) .* exp.(-(y[i] .- μ).^2 ./ (2σ²));
		p_alloc[i] = z[i] / sum(p);
	end

	return prod(p_alloc);

end


function get_comp_counts(z, k)

	c_n = zeros(Int64, k);
	for j = 1:k
		c_n[j] = length(filter(x -> x == j, z));
	end

	return c_n;

end


function gaussian_likelihood(μ, σ², w, y)

	n = length(y);
	k = length(μ);

	f = Normal.(μ, σ²);
	l = zeros(Float64, n);
	@assert(length(μ) == length(σ²) == length(w), "[error] unequal lengths");
	for i in 1:n
		l[i] = sum(pdf.(f, y[i]) .* w);
	end

	return prod(l);

end


function calc_gm_likelihood_ratio(μ, σ², w, y, ε = 1e-6)

	μ₁, μ₂ = μ;
	σ₁², σ₂² = σ²;
	w₁, w₂ = w;

	n = length(y);

	r = zeros(Float64, n);
	for i in 1:n
		l1 = gaussian_likelihood(μ₁, σ₁², w₁, y[i]);
		l2 = gaussian_likelihood(μ₂, σ₂², w₂, y[i]);
		r[i] = l1 / (l2 + ε) + ε;
	end

	return prod(r);

end


function get_dk1_bk(k, kmax)

	if params.k + 1 == kmax
		dₖ₊₁ = 1.;
		bₖ = 0.5;
	elseif params.k == 1
		dₖ₊₁ = 0.5;
		bₖ = 1.;
	else
		dₖ₊₁ = 0.5;
		bₖ = 0.5;
	end

	return dₖ₊₁, bₖ;

end


function calc_split_acceptance_rate(l_ratio, k, dₖ₊₁, bₖ, λ, ws, σs, μs, us, δ, α, β, ξ, κ, n₁, n₂, p_alloc, ε = 1e-6)

	A = l_ratio + ε;

	w, w₁, w₂ = ws;
	σ, σ₁, σ₂ = σs;
	μ, μ₁, μ₂ = μs;

	u₁, u₂, u₃ = us;

	f_u = Beta.([2, 2, 1], [2, 2, 1]);

	f_k = Poisson(λ);
	A = A * pdf(f_k, k)/(pdf(f_k, k + 1) + ε) * (k + 1) + ε;
	println("\t\t\tA1 = ", A);
	A = A * w₁^(δ-1+n₁) * w₂^(δ-1+n₂) / (w^(δ-1+n₁+n₂)*beta(δ, k*δ) + ε) + ε;
	println("\t\t\tA2 = ", A);
	A = A * sqrt(κ/2π) * exp(-1/2*κ * ((μ₁-ξ)^2 + (μ₂-ξ)^2 - (μ - ξ)^2)) + ε;
	println("\t\t\tA3 = ", A);
	A = A * β^α/gamma(α) * (σ₁^2*σ₂^2/σ^2)^(-α-1) * exp(-β*(1/σ₁^2+1/σ₂^2-1/σ^2)) + ε;
	println("\t\t\tA4 = ", A);
	A = A * dₖ₊₁/(bₖ*p_alloc*prod(pdf.(f_u, [u₁, u₂, u₃])) + ε) + ε;
	println("\t\t\tA5 = ", A);
	A = A * w * abs(μ₁-μ₂) * σ₁^2*σ₂^2 / (u₂ * (1-u₂^2) * u₃ * (1-u₃) * σ^2 + ε) + ε;

	return A;

end


function calc_birth_acceptance_rate(p, n, k, k₀, δ, λ, w, bₖ, dₖ₊₁, ε = 1e-6)

	A = 1.;
	f_k = Poisson(λ);
	A = A * pdf(f_k, k)/(pdf(f_k, k + 1) + ε);
	println("\t\t\tA1 = ", A);
	A = A/beta(k*δ, δ) * w^(δ-1) * (1-w)^(n+k*δ-k) * (k+1) + ε;
	println("\t\t\tA2 = ", A);
	A = A * dₖ₊₁/((k₀+1)*bₖ) * 1/p * (1-w)^k + ε;

	return A;

end


function is_adjacent(μs, μ₁, μ₂)

	μs = sort(μs);

	j₁ = argmin(abs.(μs .- μ₁));
	j₂ = argmin(abs.(μs .- μ₂));

	return abs(j₁ - j₂) == 1

end


# =========================================================================================


function split_comp(hyperparams, params, y, kmax, ε = 1e-6)

	n = length(y);

	jₛ = sample(1:params.k);

	f_u = Beta.([2, 2, 1], [2, 2, 1]);

	u₁, u₂, u₃ = rand.(f_u);

	μ = params.μ[jₛ];
	σ = sqrt(params.σ²[jₛ]);
	w = params.w[jₛ];

	# proposed new parameters;
	w₁ = w * u₁;
	w₂ = w * (1 - u₁);
	μ₁ = μ - u₂ * σ * sqrt(w₂ / w₁);
	μ₂ = μ + u₂ * σ * sqrt(w₁ / w₂);
	σ₁ = sqrt(u₃ * (1-u₂^2) * σ^2 * w/w₁);
	σ₂ = sqrt((1-u₃) * (1-u₂^2) * σ^2 * w/w₂);

	inds = collect(1:n);
	inds = inds[params.z .== jₛ];
	# propose reallocation;
	zₛ, p_alloc = get_allocation([μ₁, μ₂], [σ₁^2, σ₂^2], [w₁, w₂], y[inds]);
	n₁ = sum(zₛ .== 1);
	n₂ = sum(zₛ .== 2);

	#=
	println("\t\tn₁ = ", n₁, ", n₂ = ", n₂);
	println("\t\tμ₁ = ", μ₁, ", μ₂ = ", μ₂);
	println("\t\tσ₁ = ", σ₁, ", σ₂ = ", σ₂);
	println("\t\tw₁ = ", w₁, ", w₂ = ", w₂);
	println("\t\tinds = ", inds);
	=#

	println("\t\tn₁ = ", n₁, ", n₂ = ", n₂);

	zₛ = zₛ .+ params.k .- 1;

	if jₛ == 1
		μ_new = params.μ[2:end];
		σ²_new = params.σ²[2:end];
		w_new = params.w[2:end];
	elseif jₛ == params.k
		μ_new = params.μ[1:end-1];
		σ²_new = params.σ²[1:end-1];
		w_new = params.w[1:end-1]
	else
		μ_new = vcat(params.μ[1:jₛ-1], params.μ[jₛ+1:end]);
		σ²_new = vcat(params.σ²[1:jₛ-1], params.σ²[jₛ+1:end]);
		w_new = vcat(params.w[1:jₛ-1], params.w[jₛ+1:end]);
	end

	μ_new = vcat(μ_new, [μ₁, μ₂]);
	σ²_new = vcat(σ²_new, [σ₁^2, σ₂^2]);
	w_new = vcat(w_new, [w₁, w₂]);

	z_new = copy(params.z);
	z_new[inds] = zₛ;

	# check adjacent;
	if !is_adjacent(μ_new, μ₁, μ₂)
		println("\tsplit not adjacent❎");
		return params;
	end

	# l_old = gaussian_likelihood(params.μ, params.σ², params.w, y[inds]);
	# l_new = gaussian_likelihood(μ_new, σ²_new, w_new, y[inds]);

	# calculate A;

	dₖ₊₁, bₖ = get_dk1_bk(params.k, kmax);
	r = calc_gm_likelihood_ratio([μ_new, params.μ], [σ²_new, params.σ²], [w_new, params.w], y[inds]);
	A = calc_split_acceptance_rate(
		r, params.k, dₖ₊₁, bₖ, hyperparams.λ,
		[w, w₁, w₂], [σ, σ₁, σ₂], [μ, μ₁, μ₂], [u₁, u₂, u₃],
		hyperparams.δ, hyperparams.α, params.β,
		hyperparams.ξ, hyperparams.κ, n₁, n₂, p_alloc, ε
	);

	#=
	A = calc_gm_likelihood_ratio([μ_new, params.μ], [σ²_new, params.σ²], [w_new, params.w], y[inds]);

	dₖ₊₁, bₖ = get_dk1_bk(params.k, kmax);

	f_k = Poisson(hyperparams.λ);
	println("\t\t\tratio = ", A);
	A = A * pdf(f_k, params.k)/pdf(f_k, params.k + 1) * (params.k + 1) + ε;
	println("\t\t\tA1 = ", A);
	A = A * w₁^(hyperparams.δ-1+n₁) * w₂^(hyperparams.δ-1+n₂) / (w^(hyperparams.δ-1+n₁+n₂)*beta(hyperparams.δ, params.k*hyperparams.δ)) + ε;
	println("\t\t\tA2 = ", A);
	A = A * sqrt(hyperparams.κ/2π) * exp(-1/2*hyperparams.κ * ((μ₁-hyperparams.ξ)^2 + (μ₂-hyperparams.ξ)^2 - (μ - hyperparams.ξ)^2)) + ε;
	println("\t\t\tA3 = ", A);
	A = A * params.β^hyperparams.α/gamma(hyperparams.α) * (σ₁^2*σ₂^2/σ^2)^(-hyperparams.α-1) * exp(-params.β*(1/σ₁^2+1/σ₂^2-1/σ^2)) + ε;
	println("\t\t\tA4 = ", A);
	A = A * dₖ₊₁/(bₖ*p_alloc*prod(pdf.(f_u, [u₁, u₂, u₃]))) + ε;
	println("\t\t\tA5 = ", A);
	A = A * w * abs(μ₁-μ₂) * σ₁^2*σ₂^2 / (u₂ * (1-u₂^2) * u₃ * (1-u₃) * σ^2) + ε;
	=#

	print("\tsplit with A = ", A);

	if rand() < min(1, A)
		println("😃");
		new_params = Types.Params(μ_new, σ²_new, params.β, params.k+1, z_new, w_new);
		return new_params;
	else
		println("❎");
		return params;
	end

end


function combine_comp(hyperparams, params, y, kmax, ε = 1e-6)

	n = length(y);

	# sample the adjacent μ;
	j₁ = sample(1:params.k);
	tmp_μ = copy(params.μ);
	tmp_μ[j₁] = Inf;
	j₂ = argmin(abs.(tmp_μ .- params.μ[j₁]));
	w₁, w₂ = params.w[[j₁, j₂]];
	μ₁, μ₂ = params.μ[[j₁, j₂]];
	σ₁, σ₂ = sqrt.(params.σ²[[j₁, j₂]]);

	w = w₁ + w₂;
	μ = (w₁*μ₁ + w₂*μ₂) / w;
	σ² = (w₁*(μ₁^2+σ₁^2) + w₂*(μ₂^2+σ₂^2)) / w - μ^2;

	μ_new = copy(params.μ);
	σ²_new = copy(params.σ²);
	w_new = copy(params.w);
	z_new = copy(params.z);

	μ_new[j₁] = μ;
	σ²_new[j₁] = σ²;
	w_new[j₁] = w;

	inds = filter(i -> params.z[i] == j₂, 1:n);
	z_new[inds] = collect(range(j₁, j₁, length = length(inds)));

	if j₂ == 1
		μ_new = μ_new[2:end];
		σ²_new = σ²_new[2:end];
		w_new = w_new[2:end];
		z_new = z_new .- 1;
	elseif j₂ == params.k
		μ_new = μ_new[1:end-1];
		σ²_new = σ²_new[1:end-1];
		w_new = w_new[1:end-1];
	else
		μ_new = vcat(μ_new[1:j₂-1], μ_new[j₂+1:end]);
		σ²_new = vcat(σ²_new[1:j₂-1], σ²_new[j₂+1:end]);
		w_new = vcat(w_new[1:j₂-1], w_new[j₂+1:end]);
		inds = filter(i -> z_new[i] >= j₂, 1:n);
		z_new[inds] = z_new[inds] .- 1;
	end

	u₁ = w₁ / w;
	u₂ = (μ - μ₁) / sqrt(σ² * w₂/w₁);
	u₃ = σ₁^2 / ((1-u₂^2) * σ² * w/w₁);
	n₁ = sum(params.z .== j₁);
	n₂ = sum(params.z .== j₂);

	inds = filter(i -> params.z[i] == j₁ || params.z[i] == j₂, 1:n);
	z_tmp = params.z[inds];
	z_tmp[z_tmp .== j₁] = collect(range(1, 1, length = sum(z_tmp .== j₁)));
	z_tmp[z_tmp .== j₂] = collect(range(2, 2, length = sum(z_tmp .== j₂)));
	p_alloc = calc_p_alloc([μ₁, μ₂], [σ₁^2, σ₂^2], [w₁, w₂], y[inds], z_tmp);

	dₖ₊₁, bₖ = get_dk1_bk(params.k, kmax);
	r = calc_gm_likelihood_ratio([params.μ, μ_new], [params.σ², σ²_new], [params.w, w_new], y[inds]);
	A = calc_split_acceptance_rate(
		r, params.k-1, dₖ₊₁, bₖ, hyperparams.λ,
		[w, w₁, w₂], [sqrt(σ²), σ₁, σ₂], [μ, μ₁, μ₂], [u₁, u₂, u₃],
		hyperparams.δ, hyperparams.α, params.β,
		hyperparams.ξ, hyperparams.κ-1, n₁, n₂, p_alloc, ε
	);

	print("\tcombine with 1/A = ", 1/A);

	if rand() < min(1, 1/A)
		println("😃");
		new_params = Types.Params(μ_new, σ²_new, params.β, params.k-1, z_new, w_new);
		return new_params;
	else
		println("❎");
		return params;
	end

end


function birth_comp(hyperparams, params, y, kmax, ε = 1e-6)

	n = length(y);

	f_w = Beta(1, params.k);
	w = rand(f_w);

	f_μ = Normal(hyperparams.ξ, 1/hyperparams.κ);
	μ = rand(f_μ);

	f_σ² = Gamma(hyperparams.α, 1/params.β);
	σ² = 1/rand(f_σ²);

	w_new = vcat(params.w .* (1 - w), [w]);
	μ_new = vcat(params.μ, [μ]);
	σ²_new = vcat(params.σ², [σ²]);

	dₖ₊₁, bₖ = get_dk1_bk(params.k, kmax);

	c_n = get_comp_counts(params.z, params.k);
	# k₀ = length(filter(j -> c_n[j] == 0, 1:params.k));
	k₀ = sum(c_n == 0);

	A = calc_birth_acceptance_rate(pdf(f_w, w), n, params.k, k₀, hyperparams.δ, hyperparams.λ, w, bₖ, dₖ₊₁, ε);

	#=
	A = 1.;
	f_k = Poisson(hyperparams.λ);
	A *= pdf(f_k, params.k)/pdf(f_k, params.k + 1);
	A *= 1/beta(params.k*hyperparams.δ, hyperparams.δ) * w^(hyperparams.δ-1) * (1-w)^(n+params.k*hyperparams.δ-params.k) * (params.k+1);
	A *= dₖ₊₁/((k₀+1)*bₖ) * 1/pdf(f_w, w) * (1-w)^params.k;
	=#

	print("\tbirth with A = ", A);

	if rand() < min(1, A)
		println("😃");
		new_params = Types.Params(μ_new, σ²_new, params.β, params.k+1, params.z, w_new);
		return new_params;
	else
		println("❎");
		return params;
	end

end


function death_comp(hyperparams, params, y, kmax, ε = 1e-6)

	n = length(y);

	c_n = get_comp_counts(params.z, params.k);
	j₀s = filter(j -> c_n[j] == 0, 1:params.k);
	if isempty(j₀s)
		return params
	end
	j₀ = sample(j₀s);

	μ_new = copy(params.μ);
	σ²_new = copy(params.σ²);
	w_new = copy(params.w);

	z_new = copy(params.z);

	if j₀ == 1
		μ_new = μ_new[2:end];
		σ²_new = σ²_new[2:end];
		w_new = w_new[2:end];
		z_new = z_new .- 1;
	elseif j₀ == params.k
		μ_new = μ_new[1:end-1];
		σ²_new = σ²_new[1:end-1];
		w_new = w_new[1:end-1];
		# no need to change allocation;
	else
		μ_new = vcat(μ_new[1:j₀-1], μ_new[j₀+1:end]);
		σ²_new = vcat(σ²_new[1:j₀-1], σ²_new[j₀+1:end]);
		w_new = vcat(w_new[1:j₀-1], w_new[j₀+1:end]);
		inds = filter(i -> z_new[i] >= j₀, 1:n);
		z_new[inds] = z_new[inds] .- 1;
	end

	# μ_new = vcat(params.μ[1:max(1, j₀-1)], params.μ[min(params.k, j₀+1):end]);
	# w_new = vcat(params.w[1:max(1, j₀-1)], params.w[min(params.k, j₀+1):end]);
	# σ²_new = vcat(params.σ²[1:max(1, j₀-1)], params.σ²[min(params.k, j₀+1):end]);

	f_w = Beta(1, params.k-1);
	k₀ = length(filter(j -> c_n[j] == 0, 1:params.k)) - 1;
	dₖ₊₁, bₖ = get_dk1_bk(params.k-1, kmax);
	A = calc_birth_acceptance_rate(pdf(f_w, params.w[j₀]), n, params.k-1, k₀, hyperparams.δ, hyperparams.λ, params.w[j₀], bₖ, dₖ₊₁, ε);

	print("\tdeath with 1/A = ", 1/A);
	if rand() < min(1, 1/A)
		println("😃");
		# println(params.k-1, ", ", length(μ_new), ", ", length(σ²_new), ", ", length(w_new));
		new_params = Types.Params(μ_new, σ²_new, params.β, params.k-1, z_new, w_new);
		return new_params;
	else
		println("❎");
		return params;
	end

end
