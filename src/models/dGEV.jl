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

function setParams(model::dGEVModel, new::Vector{<:Real}; 
                    is_transformed::Bool = true)
    """Returns a new dGEVModel with the updated set of param values. 
    If is_transformed==true, then "new" contains a set of transformed param values
    If is_transformed==false, then "new" contains a set of param values
    """

    if is_transformed 
        new_θ = new
        return dGEVModel(model.d_ref, getParams(dGEVModel, new_θ))
    else
        new_params = new
        return dGEVModel(model.d_ref, new_params)
    end
end


function initializeModel(model_type::Type{<:dGEVModel}, data::DataFrame;
                            d_ref::Union{Real, Nothing} = nothing)
    """Returns a dGEVModel based on a regression of the Gumbel parameters estimated at each duration independently
    As this model is just an initialization and will be used at the beginning of an optimization process,
    we make sure that the parameter values are not at their boundaries"""

    no_scaling_gumbel_model = modelEstimation(fitMLE(NoScalingGumbelModel, data))
    dGEV_model = estimdGEVModel(no_scaling_gumbel_model, d_ref = d_ref)

    μ, σ, ξ, α, δ = dGEV_model.params
    params_new = [ maximum([μ, 0.01]),
                maximum([σ, 0.01]),
                maximum( [ minimum([ξ, 0.49]), -0.49 ] ),
                maximum( [ minimum([α, 0.99]), 0.01 ] ),
                maximum([δ, 0.01])
    ]

    return setParams(dGEV_model, params_new, is_transformed=false)

end


