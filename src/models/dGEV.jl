struct dGEVModel <: IDFModel

    params_names::Vector{<:String}
    D_values::Vector{<:Real} # en minutes
    d_ref::Real # en minutes

    function dGEVModel(D_values::Vector{<:Real};
                        d_ref::Union{Real, Nothing} = nothing)
        if isnothing(d_ref)
            d_ref = maximum(D_values)
        end
        return new(["μ_"*string(d_ref), "σ_"*string(d_ref), "ξ_"*string(d_ref), "α", "δ"], D_values, d_ref)
    end

end

function transformParams(model::dGEVModel, params::Vector{<:Real})
    """Transforms a vector of parameters for the dGEVModel to put it in the real space"""
    return [log(params[1]), log(params[2]), logistic(params[3]+0.5), logistic(params[4]), log(params[5])]
end

function getParams(model::dGEVModel, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""
    return [exp(θ[1]), exp(θ[2]), logistic_inverse(θ[3])-0.5, logistic_inverse(θ[4]), exp(θ[5])]
end

function logLikelihood(model::dGEVModel, data::DataFrame, θ::Vector{<:Real})

    D_values = model.D_values
    d_ref = model.d_ref

    observations = select(data, Symbol.(to_french_name.(D_values)))

    # GEV parameters
    μ = exp(θ[1])
    σ = exp(θ[2])
    ξ = logistic_inverse(θ[3]) - 0.5

    # scale relationship parameters
    α = logistic_inverse(θ[4])
    δ = exp(θ[5])

    #Calculus of the log-likelihood
    log_likelihood = 0.0
    for d in D_values

        μ_d = μ * ( (d + δ) / (d_ref + δ) ) ^ (-α)
        σ_d = σ * ( (d + δ) / (d_ref + δ) ) ^ (-α)
        ξ_d = ξ

        distrib_extreme_d = GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
        log_likelihood = log_likelihood + sum( logpdf.(Ref(distrib_extreme_d), 
                                                observations[:,Symbol(to_french_name(d))]) )

    end

    return log_likelihood
end

function initializeParams(model::dGEVModel, data::DataFrame)

    # estimated IDF based on duration-by-duration Gumbel parameters estimation
    no_scaling_model = NoScalingGumbelModel(model.D_values)
    fitted_no_scaling_model = fitMLE(no_scaling_model, data)
    estimated_IDF = estimIDFRelationship(fitted_no_scaling_model, d_ref = model.d_ref)

    return [estimated_IDF[3], estimated_IDF[4], 0.0, estimated_IDF[1], estimated_IDF[2]] #params must lie in the ensemble of reals (ie. be transformed)
end