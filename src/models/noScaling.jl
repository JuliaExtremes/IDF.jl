struct NoScalingGumbelModel <: IDFModel

    params_names::Vector{<:String}
    D_values::Vector{<:Real} # en minutes

    function NoScalingGumbelModel(D_values::Vector{<:Real})
        return new( reduce(vcat, (["μ_"*string(d), "σ_"*string(d)] for d in D_values)), D_values )
    end

end

function transformParams(model::NoScalingGumbelModel, params::Vector{<:Real})
    """Transforms a vector of parameters for the dGEVModel to put it in the real space"""
    return log.(params)
end

function getParams(model::NoScalingGumbelModel, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""
    return exp.(θ)
end

function logLikelihood(model::NoScalingGumbelModel, data::DataFrame, θ::Vector{<:Real})
    """For now everything will lie in the initialization"""

    D_values = model.D_values

    observations = select(data, Symbol.(to_french_name.(D_values)))

    # parameters
    params = getParams(model, θ)

    #Calculus of the log-likelihood
    log_likelihood = 0.0
    for i in eachindex(D_values)
        d = D_values[i]

        μ_d = params[2*i-1]
        σ_d = params[2*i]

        distrib_extreme_d = Gumbel(μ_d, σ_d)
        log_likelihood = log_likelihood + sum( logpdf.(Ref(distrib_extreme_d), 
            observations[:,Symbol(to_french_name(d))]) )
    end
    
    return log_likelihood
end

function initializeParams(model::NoScalingGumbelModel, data::DataFrame)
    """Everything lies in the initialization, as we use Extremes.jl to optimize (with the moments) the parameters values"""

    D_values = model.D_values
    
    θ_init = []
    for d in D_values
        μ, ϕ = Extremes.gumbelfitpwm(data, Symbol(to_french_name(d))).θ̂
        θ_init = [θ_init; [log(μ), ϕ]]
    end
    return Vector{Float64}(θ_init) #params must lie in the ensemble of reals (ie. be transformed)
end

function estimSimpleScalingRelationship(model::NoScalingGumbelModel, θ::Vector{<:Real};
    d_ref::Union{Real, Nothing} = nothing)
    """Return an estimation of the simple scaling relationship parameters (α, C_μ (intercept for log(μ)), C_σ (intercept for log(σ)))
    based on a linear regression of μ_d and σ_d depending on d. 
    """

    D_values = model.D_values
    if isnothing(d_ref)
        d_ref = maximum(D_values)
    end

    # error function
    function error(x)
        """x = [logistic(α), C_μ, C_σ]"""

        α, C_μ, C_σ = logistic_inverse(x[1]), x[2], x[3]

        error = 0.0
        for i in eachindex(D_values)
            d=D_values[i]
            log_μ̂_d = θ[2*i-1]
            log_σ̂_d = θ[2*i]
            error = error + ( log_μ̂_d - (C_μ - α*(log(d)-log(d_ref))) )^2
            error = error + ( log_σ̂_d - (C_σ - α*(log(d)-log(d_ref))) )^2
        end

        return error
    end

    # initialization
    α_init = - GLM.coef(GLM.lm(
                    @formula(y ~ x), 
                    DataFrame(x = log.(D_values), y = [θ[2*i-1] for i in eachindex(D_values)])
                    )
                )[2] # initialization of α via linear regression of log(μ_d) depending on log(d)
    index_d_ref = argmax(D_values.== d_ref)
    x_init = [logistic(α_init), θ[2*index_d_ref-1], θ[2*index_d_ref]]

    # optimization
    result = optimize(error, x_init, BFGS())

    return result.minimizer

end

function estimIDFRelationship(model::NoScalingGumbelModel, θ::Vector{<:Real};
                                d_ref::Union{Real, Nothing} = nothing)
    """Return an estimation of the general IDF relationship parameters (α, δ, C_μ (intercept for log(μ)), C_σ (intercept for log(σ)))
    based on a regression of μ_d and σ_d depending on d. 
    """

    D_values = model.D_values
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
            log_μ̂_d = θ[2*i-1]
            log_σ̂_d = θ[2*i]
            error = error + ( log_μ̂_d - (C_μ - α*(log(d + δ)-log(d_ref + δ))) )^2
            error = error + ( log_σ̂_d - (C_σ - α*(log(d + δ)-log(d_ref + δ))) )^2
        end

        return error
    end

    # initialization
    simple_scaling_params = estimSimpleScalingRelationship(model, θ, d_ref = d_ref)
    x_init = [simple_scaling_params[1], 0.0, simple_scaling_params[2], simple_scaling_params[3]]

    # optimization
    result = optimize(error, x_init, BFGS())

    return result.minimizer
end

function returnLevel(model::NoScalingGumbelModel, θ::Vector{<:Real}, d::Real, T::Real)
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
    μ_d, σ_d = params[2*i-1], params[2*i]
    distrib_extreme_d = Gumbel(μ_d, σ_d)

    return quantile(distrib_extreme_d, 1 - 1/T)

end