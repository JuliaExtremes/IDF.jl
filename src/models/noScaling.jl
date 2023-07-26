struct NoScalingGEVModel <: IDFModel

    params_names::Vector{<:String}
    D_values::Vector{<:Real} # en minutes

    function NoScalingGEVModel(D_values::Vector{<:Real})
        return new( reduce(vcat, (["μ_"*string(d), "σ_"*string(d), "ξ_"*string(d)] for d in D_values)), D_values )
    end

end

function transformParams(model::NoScalingGEVModel, params::Vector{<:Real})
    """Transforms a vector of parameters for the dGEVModel to put it in the real space"""

    n_params = length(model.params_names)
    θ = Vector{}(undef, n_params)

    for i in 1:n_params
        if i%3 == 0
            θ[i] = logistic(params[i] + 0.5)
        else
            θ[i] = log(params[i])
        end
    end

    return θ
end

function getParams(model::NoScalingGEVModel, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""

    n_params = length(model.params_names)
    params = Vector{}(undef, n_params)

    for i in 1:n_params
        if i%3 == 0
            params[i] = logistic_inverse(θ[i]) - 0.5
        else
            params[i] = exp(θ[i])
        end
    end

    return params
end

function logLikelihood(model::NoScalingGEVModel, data::DataFrame, θ::Vector{<:Real})
    """"""

    D_values = model.D_values

    observations = select(data, Symbol.(to_french_name.(D_values)))

    # parameters
    params = getParams(model, θ)

    #Calculus of the log-likelihood
    log_likelihood = 0.0
    for i in eachindex(D_values)
        d = D_values[i]

        μ_d = params[3*i-2]
        σ_d = params[3*i-1]
        ξ_d = params[3*i]

        distrib_extreme_d = GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
        log_likelihood = log_likelihood + sum( logpdf.(Ref(distrib_extreme_d), 
            observations[:,Symbol(to_french_name(d))]) )
    end
    
    return log_likelihood
end

function initializeParams(model::NoScalingGEVModel, data::DataFrame)
    """Everything lies in the initialization, as we use Extremes.jl to optimize (with the moments) the parameters values"""

    no_scaling_model = NoScalingGumbelModel(model.D_values)
    fitted_no_scaling_model = fitMLE(no_scaling_model, data)

    D_values = model.D_values
    n_params = length(model.params_names)

    θ_init = Vector{Float64}(undef, n_params)
    for i in eachindex(D_values)
        θ_init[3*i-2] = fitted_no_scaling_model.θ̂[2*i-1]
        θ_init[3*i-1] = fitted_no_scaling_model.θ̂[2*i]
        θ_init[3*i] = 0.0
    end

    return θ_init #params must lie in the ensemble of reals (ie. be transformed)
end

function estimSimpleScalingRelationship(model::NoScalingGEVModel, θ::Vector{<:Real};
    d_ref::Union{Real, Nothing} = nothing)
    """Return an estimation of the simple scaling relationship parameters (α, C_μ (intercept for log(μ)), C_σ (intercept for log(σ)))
    based on a linear regression of μ_d and σ_d depending on d. 
    """

    gumbel_model = NoScalingGumbelModel(model.D_values)
    n_params = length(model.params_names)
    gumbel_θ = θ[((1:n_params) .% 3) .!= 0]

    return estimSimpleScalingRelationship(gumbel_model, gumbel_θ, d_ref = d_ref)

end

function estimIDFRelationship(model::NoScalingGEVModel, θ::Vector{<:Real};
                                d_ref::Union{Real, Nothing} = nothing)
    """Return an estimation of the general IDF relationship parameters (α, δ, C_μ (intercept for log(μ)), C_σ (intercept for log(σ)))
    based on a regression of μ_d and σ_d depending on d. 
    """

    gumbel_model = NoScalingGumbelModel(model.D_values)
    n_params = length(model.params_names)
    gumbel_θ = θ[((1:n_params) .% 3) .!= 0]

    return estimIDFRelationship(gumbel_model, gumbel_θ, d_ref = d_ref)

end

function returnLevel(model::NoScalingGEVModel, θ::Vector{<:Real}, d::Real, T::Real)
    """Renvoie le niveau de retour associé à une durée d'accumulation d et à un temps de retour T,
    pour le modèle Gumbel sans invariance d'échelle paramétré par le vecteur (transformé) θ
    """

    @assert 1 < T "The return period must be bigger than 1 year."

    index_d = (model.D_values .== d)
    @assert sum(index_d) == 1 "The duration d must be one of the reference durations of the model. " *
                            "If you want to study intermediate durations, start by using " *
                            "'estimSimpleScalingRelationship()' or 'estimIDFRelationship() in order to create a " *
                            "SimpleScalingModel or a dGEVModel."
    i = argmax(index_d)

    params = getParams(model, θ)
    μ_d, σ_d, ξ_d = params[3*i-2], params[3*i-1], params[3*i]
    distrib_extreme_d = GeneralizedExtremeValue(μ_d, σ_d, ξ_d)

    return quantile(distrib_extreme_d, 1 - 1/T)

end