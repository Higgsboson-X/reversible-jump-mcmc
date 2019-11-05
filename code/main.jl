using Plots

include("./utils.jl")
include("./types.jl")
include("./rjmcmc.jl")
include("./data_generator.jl")

function simulate(epochs = 100, p_split = 0.9, p_birth = 0.8)

	μₜ = [1, 2, 5];
	σₜ = [1, 1, 1];
	wₜ = [0.2, 0.3, 0.5];

	n = 1000;
	y = gaussian_mixture_pts(μₜ, σₜ.^2, wₜ, n);

	hyperparams = get_default_hyperparams(y);
	params = rjmcmc(y, hyperparams, 6, epochs, p_split, p_birth);

	fₜ = Normal.(μₜ, σₜ.^2);
	fₚ = Normal.(params.μ, params.σ²);

	a = minimum(μₜ .- σₜ);
	b = maximum(μₜ .+ σₜ);

	p = collect(range(a, b, step = 0.01));

	vₚ = zeros(Float64, length(p));
	vₜ = zeros(Float64, length(p));

	for i = 1:length(p)
		vₜ[i] = sum(pdf.(fₜ, p[i]) .* wₜ);
		vₚ[i] = sum(pdf.(fₚ, p[i]) .* params.w);
	end

	println(">>>>>>> Summary <<<<<<<");
	println("\tk = ", params.k);
	println("\tμ = ", params.μ);
	println("\tσ = ", sqrt.(params.σ²));
	println("\tw = ", params.w);

	plot(p, [vₜ, vₚ], label = ["truth", "fitted"], fontfamily = "Courier");

end
