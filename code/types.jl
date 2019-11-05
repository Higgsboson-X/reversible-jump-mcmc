module Types

mutable struct HyperParams
	ξ # mean(y);
	κ # var(y);
	α
	g
	h
	λ
	δ
end

mutable struct Params
	μ  # μ ~ Normal(ξ, 1 / κ);
	σ² # 1/σ² ~ Gamma(α, β);
	β  # β ~ Gamma(g, h);
	k  # k ~ Poisson(λ);
	z  # p(zᵢ=j|⋅) ∝ wⱼ/σⱼ * exp(-(yⱼ - μⱼ)² / 2σⱼ);
	w  # β|⋅ ~ Gamma(g + kα, h + ∑1/σⱼ²);
end

end
