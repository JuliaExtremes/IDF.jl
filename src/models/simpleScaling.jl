struct SimpleScalingModel <: IDFModel

    params_names::Vector{<:String}
    d_ref::Real # en minutes
    params::Vector{<:Real}

    function SimpleScalingModel(d_ref::Real, params::Vector{<:Real})
        
        return new(["μ_"*string(d_ref), "σ_"*string(d_ref), "ξ_"*string(d_ref), "α"], d_ref, params)
    end

end


function getDistribution(model::SimpleScalingModel, d::Real) 
    """Returns the distribution of the maximal intensity for duration d, according to the model"""

    μ, σ, ξ, α = model.params

    μ_d = μ * ( d / model.d_ref ) ^ (-α)
    σ_d = σ * ( d / model.d_ref ) ^ (-α)
    ξ_d = ξ

    return GeneralizedExtremeValue(μ_d, σ_d, ξ_d)

end


function transformParams(model::SimpleScalingModel)
    """Transforms the vector of parameters of the model to put it in the real space"""

    μ, σ, ξ, α = model.params
    return [log(μ), log(σ), logistic(ξ+0.5), logistic(α)]

end


function getParams(model_type::Type{<:SimpleScalingModel}, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""

    return [exp(θ[1]), exp(θ[2]), logistic_inverse(θ[3])-0.5, logistic_inverse(θ[4])]
end


function setParams(model::SimpleScalingModel, new::Vector{<:Real}; 
                    is_transformed::Bool = true)
    """Returns a new SimpleScalingModel with the updated set of param values. 
    If is_transformed==true, then "new" contains a set of transformed param values
    If is_transformed==false, then "new" contains a set of param values
    """

    if is_transformed 
        new_θ = new
        return SimpleScalingModel(model.d_ref, getParams(SimpleScalingModel, new_θ))
    else
        new_params = new
        return SimpleScalingModel(model.d_ref, new_params)
    end
end


function initializeModel(model_type::Type{<:SimpleScalingModel}, data::DataFrame;
                            d_ref::Union{Real, Nothing} = nothing)
    """Returns a SimpleScalingModel based on a regression of the Gumbel parameters estimated at each duration independently"""

    no_scaling_gumbel_model = modelEstimation(fitMLE(NoScalingGumbelModel, data))

    return estimSimpleScalingModel(no_scaling_gumbel_model, d_ref = d_ref)

end

