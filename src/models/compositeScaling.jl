struct CompositeScalingModel <: IDFModel

    params_names::Vector{<:String}
    d_ref::Real # en minutes
    params::Vector{<:Real}

    function CompositeScalingModel(d_ref::Real, params::Vector{<:Real})
        
        return new(["μ_"*string(d_ref), "σ_"*string(d_ref), "ξ_"*string(d_ref), "α_μ", "α_σ"], d_ref, params)
    end

end


function getDistribution(model::CompositeScalingModel, d::Real) 
    """Returns the distribution of the maximal intensity for duration d, according to the model"""

    μ, σ, ξ, α_μ, α_σ = model.params

    μ_d = μ * ( d / model.d_ref ) ^ (-α_μ)
    σ_d = σ * ( d / model.d_ref ) ^ (-α_σ)
    ξ_d = ξ 

    return GeneralizedExtremeValue(μ_d, σ_d, ξ_d)

end


function transformParams(model::CompositeScalingModel)
    """Transforms the vector of parameters of the model to put it in the real space"""

    μ, σ, ξ, α_μ, α_σ = model.params
    return [log(μ), log(σ), logistic(ξ+0.5), logistic(α_μ), logistic(α_σ)]

end


function getParams(model_type::Type{<:CompositeScalingModel}, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""

    return [exp(θ[1]), exp(θ[2]), logistic_inverse(θ[3])-0.5, logistic_inverse(θ[4]), logistic_inverse(θ[5])]
end


function setParams(model::CompositeScalingModel, new::Vector{<:Real}; 
                    is_transformed::Bool = true)
    """Returns a new CompositeScalingModel with the updated set of param values. 
    If is_transformed==true, then "new" contains a set of transformed param values
    If is_transformed==false, then "new" contains a set of param values
    """

    if is_transformed 
        new_θ = new
        return CompositeScalingModel(model.d_ref, getParams(CompositeScalingModel, new_θ))
    else
        new_params = new
        return CompositeScalingModel(model.d_ref, new_params)
    end
end


function initializeModel(model_type::Type{<:CompositeScalingModel}, data::DataFrame;
                            d_ref::Union{Real, Nothing} = nothing)
    """Returns a CompositeScalingModel based on a regression of the Gumbel parameters estimated at each duration independently
    As this model is just an initialization and will be used at the beginning of an optimization process,
    we make sure that the parameter values are not at their boundaries"""

    # TODO

end

function gradF_dref(model_type::Type{<:CompositeScalingModel}, x, θ) 
    """Returns the value of ∇_θ F_dref(x,θ), where :
        - θ is a transformed vector of global parameters for the Composite Scaling model.
        - F_dref is the cdf of the marginal variable Y_dref (rain maximas for duration d_ref)
        - x is in the support of the variable Y_dref.
    """
    
    # TODO
    
end

