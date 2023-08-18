struct dGEVModel <: IDFModel

    params_names::Vector{<:String}
    d_ref::Real # en minutes
    params::Vector{<:Real}

    function dGEVModel(d_ref::Union{Real, Nothing}, params::Vector{<:Real})
        
        return new(["μ_"*string(d_ref), "σ_"*string(d_ref), "ξ_"*string(d_ref), "α", "δ"], d_ref, params)
    end

end

function getDistribution(model::dGEVModel, d::Real) 
    """Returns the distribution of the maximal intensity for duration d, according to the model"""

    μ, σ, ξ, α, δ = model.params

    μ_d = μ * ( (d + δ) / (model.d_ref + δ) ) ^ (-α)
    σ_d = σ * ( (d + δ) / (model.d_ref + δ) ) ^ (-α)
    ξ_d = ξ

    return GeneralizedExtremeValue(μ_d, σ_d, ξ_d)

end


function transformParams(model::dGEVModel)
    """Transforms the vector of parameters of the model to put it in the real space"""

    μ, σ, ξ, α, δ = model.params
    return [log(μ), log(σ), logistic(ξ+0.5), logistic(α), log(δ)]

end


function getParams(model_type::Type{<:dGEVModel}, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""
    return [exp(θ[1]), exp(θ[2]), logistic_inverse(θ[3])-0.5, logistic_inverse(θ[4]), exp(θ[5])]
end


function setParams(model::dGEVModel, new_θ::Vector{<:Real})
    """Returns a new dGEVModelwith the updated set of param values. The argument is θ, ie. the transformed param values"""

    return dGEVModel(model.d_ref, getParams(dGEVModel, new_θ))
    
end


function initializeModel(model_type::Type{<:dGEVModel}, data::DataFrame;
    d_ref::Union{Real, Nothing} = nothing)
"""Returns a dGEVModel based on a regression of the Gumbel parameters estimated at each duration independently"""

no_scaling_gumbel_model = getEstimatedModel(fitMLE(NoScalingGumbelModel, data))

return estimdGEVModel(no_scaling_gumbel_model, d_ref = d_ref)

end


