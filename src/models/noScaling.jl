struct NoScalingGEVModel <: IDFModel

    params_names::Vector{<:String}
    D_values::Vector{<:Real} # en minutes
    params::Vector{<:Real}

    function NoScalingGEVModel(D_values::Vector{<:Real}, params::Vector{<:Real})
        return new( reduce(vcat, (["μ_"*string(d), "σ_"*string(d), "ξ_"*string(d)] for d in D_values)), D_values, params)
    end

end


function getDistribution(model::NoScalingGEVModel, d::Real) 
    """Returns the distribution of the maximal intensity for duration d, according to the model"""

    index_d = (model.D_values .== d)
    @assert sum(index_d) == 1 "The duration d must be one of the reference durations of the model. " *
                            "If you want to study intermediate durations, start by using " *
                            "'estimSimpleScalingRelationship()' or 'estimIDFRelationship() in order to create a " *
                            "SimpleScalingModel or a dGEVModel."

    i = argmax(index_d)

    μ_d, σ_d, ξ_d = model.params[3*i-2], model.params[3*i-1], model.params[3*i]

    return GeneralizedExtremeValue(μ_d, σ_d, ξ_d)

end


function transformParams(model::NoScalingGEVModel)
    """Transforms the vector of parameters of the model to put it in the real space"""

    n_params = length(model.params)

    θ = [log(model.params[1])]
    for i in 2:n_params
        if i%3 == 0
            θ = [ θ ; [logistic(model.params[i] + 0.5)] ]
        else
            θ = [ θ ; [log(model.params[i])] ]
        end
    end

    return θ

end


function getParams(model_type::Type{<:NoScalingGEVModel}, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""

    n_params = length(θ)

    params = [exp(θ[1])]
    for i in 2:n_params
        if i%3 == 0
            params = [params ; [logistic_inverse(θ[i]) - 0.5] ]
        else
            params = [params ; [exp(θ[i])] ]
        end
    end

    return params

end

function setParams(model::NoScalingGEVModel, new::Vector{<:Real}; 
                    is_transformed::Bool = true)
    """Returns a new NoScalingGEVModel with the updated set of param values. 
    If is_transformed==true, then "new" contains a set of transformed param values
    If is_transformed==false, then "new" contains a set of param values
    """

    if is_transformed 
        new_θ = new
        return NoScalingGEVModel(model.D_values, getParams(NoScalingGEVModel, new_θ))
    else
        new_params = new
        return NoScalingGEVModel(model.D_values, new_params)
    end
end


function initializeModel(model_type::Type{<:NoScalingGEVModel}, data::DataFrame;
    d_ref::Union{Real, Nothing} = nothing)
    """As for the NoScalingGumbelModel, we use Extremes.jl to optimize (with the moments) the parameters values"""

    no_scaling_gumbel_model = initializeModel(NoScalingGumbelModel, data)

    D_values = no_scaling_gumbel_model.D_values

    params_init = []
    for i in eachindex(D_values)
        params_init = [params_init; [no_scaling_gumbel_model.params[2*i-1], no_scaling_gumbel_model.params[2*i], 0.0]]
    end

    params_init = Float64.(params_init)
    return NoScalingGEVModel(D_values, params_init)

end

function estimSimpleScalingModel(model::NoScalingGEVModel;
                                    d_ref::Union{Real, Nothing} = nothing)
    """Returns a simple scaling model parametrized by an estimation of the simple scaling relationship parameters  (μ_d_ref, σ_d_ref, α)
    based on a linear regression of μ_d and σ_d depending on d. 
    We use the corresponding method for the NoScalingGumbelModel, and then set ξ as the mean ξ for all durations.
    """
    
    n_params = length(model.params)
    gumbel_params = model.params[((1:n_params) .% 3) .!= 0]
    gumbel_model = NoScalingGumbelModel(model.D_values, gumbel_params)
    estim_SS_via_gumbel = estimSimpleScalingModel(gumbel_model, d_ref = d_ref)

    params_value = estim_SS_via_gumbel.params
    params_value[3] = mean(model.params[((1:n_params) .% 3) .== 0])

    return setParams(estim_SS_via_gumbel, params_value, is_transformed=false)

end

function estimdGEVModel(model::NoScalingGEVModel;
                            d_ref::Union{Real, Nothing} = nothing)
    """Returns a dGEV model parametrized by an estimation of the dGEV parameters  (μ_d_ref, σ_d_ref, α, δ)
    based on a linear regression of μ_d and σ_d depending on d. 
    We use the corresponding method for the NoScalingGumbelModel, and then set ξ as the mean ξ for all durations.
    """

    n_params = length(model.params)
    gumbel_params = model.params[((1:n_params) .% 3) .!= 0]
    gumbel_model = NoScalingGumbelModel(model.D_values, gumbel_params)
    estim_dGEV_via_gumbel = estimdGEVModel(gumbel_model, d_ref = d_ref)

    params_value = estim_dGEV_via_gumbel.params
    params_value[3] = mean(model.params[((1:n_params) .% 3) .== 0])

    return setParams(estim_dGEV_via_gumbel, params_value, is_transformed=false)

end