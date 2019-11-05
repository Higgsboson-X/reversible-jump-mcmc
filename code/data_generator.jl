using Distributions
using StatsBase

# simulate gaussian mixtures;
function gaussian_mixture_pts(μ, σ², w, n_pts)

	f_list = Normal.(μ, σ²);

	# l = minimum(μ .- 3 * sqrt.(σ²));
	# u = maximum(μ .+ 3 * sqrt.(σ²));

	pts = zeros(Float64, n_pts);
	n = ceil.(Int64, n_pts .* w);
	k = 1;
	for i in 1:length(μ)
		pts[k:(k+n[i]-1)] = rand(f_list[i], n[i]);
		k += n[i];
	end

	return pts;

end


function gm_likelihood(μ, σ², w)

	f_list = Normal.(μ, σ²);
	f(x) = sum(pdf.(f_list, x));

	return f;

end

function visualize_gm(pts, l, u, n)

	histogram(pts, bin = range(l, u, length = n), fillcolor = [:black], fillalpha = 0);

end
