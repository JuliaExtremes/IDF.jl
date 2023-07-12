struct FittedMLE{T} <: Fitted{T}
    model::T
    θ̂::Vector{Float64}
    I_Fisher::Matrix{Float64}
end

function cint(fitted_mle::FittedMLE, g::Function;
                p::Real = 0.95)
    """Returns an asymptotic confidence interval for g(θ̂), at confidence level p, based on the asymptotic normality of the MLE.
    """

    Δg(θ::DenseVector) = ForwardDiff.gradient(g, θ)
    G = Δg(fitted_mle.θ̂)
    var = G'/fitted_mle.I_Fisher*G

    asymptotic_distrib = Normal(g(fitted_mle.θ̂), sqrt(var))

    return [quantile(asymptotic_distrib, (1-p)/2), quantile(asymptotic_distrib, (1+p)/2)]
end

function estimIDFRelationship(fitted_no_scaling::FittedMLE{NoScalingGumbelModel};
                                d_ref::Union{Real, Nothing} = nothing)
    """Return an estimation of the general IDF relationship parameters (α, δ, C_μ (intercept for log(μ)), C_σ (intercept for log(σ)))
    based on a regression of μ_d and σ_d depending on d. 

    Only relevant with fitted NoScalingGumbelModel, for which 
    μ_d and σ_d have been estimated independently from one duration d to an other.
    """

    D_values = fitted_no_scaling.model.D_values
    if isnothing(d_ref)
        d_ref = maximum(D_values)
    end

    # error function
    function error(x)
        """x = [logistic(α), log(δ), C_μ, C_σ]"""

        α, δ, C_μ, C_σ = logistic_inverse(x[1]), exp(x[2]), x[3], x[4]

        error = 0.0
        for i in eachindex(D_values)
            d=D_values[i]
            log_μ̂_d = fitted_no_scaling.θ̂[2*i-1]
            log_σ̂_d = fitted_no_scaling.θ̂[2*i]
            error = error + ( log_μ̂_d - (C_μ - α*(log(d + δ)-log(d_ref + δ))) )^2
            error = error + ( log_σ̂_d - (C_σ - α*(log(d + δ)-log(d_ref + δ))) )^2
        end

        return error
    end

    # initialization
    α_init = - GLM.coef(GLM.lm(
                    @formula(y ~ x), 
                    DataFrame(x = log.(D_values), y = [fitted_no_scaling.θ̂[2*i-1] for i in eachindex(D_values)])
                    )
                )[2] # initialization of α via linear regression of log(μ_d) depending on log(d)
    index_d_ref = argmax(D_values.== d_ref)
    x_init = [logistic(α_init), 0.0, fitted_no_scaling.θ̂[2*index_d_ref-1], fitted_no_scaling.θ̂[2*index_d_ref]]
    # optimization
    result = optimize(error, x_init, BFGS())

    return result.minimizer
end
