struct NoScalingGumbelModel <: IDFModel

    params_names::Vector{<:String}
    D_values::Vector{<:Real} # en minutes
    params::Vector{<:Real}

    function NoScalingGumbelModel(D_values::Vector{<:Real}, params::Vector{<:Real})
        return new( reduce(vcat, (["μ_"*string(d), "σ_"*string(d)] for d in D_values)), D_values, params)
    end

end


function getDistribution(model::NoScalingGumbelModel, d::Real) 
    """Returns the distribution of the maximal intensity for duration d, according to the model"""

    index_d = (model.D_values .== d)
    @assert sum(index_d) == 1 "The duration d must be one of the reference durations of the model. " *
                            "If you want to study intermediate durations, start by using " *
                            "'estimSimpleScalingRelationship()' or 'estimIDFRelationship() in order to create a " *
                            "SimpleScalingModel or a dGEVModel."

    i = argmax(index_d)

    μ_d, σ_d = model.params[2*i-1], model.params[2*i]

    return Gumbel(μ_d, σ_d)

end


function transformParams(model::NoScalingGumbelModel)
    """Transforms the vector of parameters of the model to put it in the real space"""

    return log.(model.params)

end


function getParams(model_type::Type{<:NoScalingGumbelModel}, θ::Vector{<:Real})
    """Returns the vector of parameters associated to the transformed vector θ"""

    return exp.(θ)

end


function setParams(model::NoScalingGumbelModel, new::Vector{<:Real}; 
                    is_transformed::Bool = true)
    """Returns a new NoScalingGumbelModel with the updated set of param values. 
    If is_transformed==true, then "new" contains a set of transformed param values
    If is_transformed==false, then "new" contains a set of param values
    """

    if is_transformed 
        new_θ = new
        return NoScalingGumbelModel(model.D_values, getParams(NoScalingGumbelModel, new_θ))
    else
        new_params = new
        return NoScalingGumbelModel(model.D_values, new_params)
    end
end



function initializeModel(model_type::Type{<:NoScalingGumbelModel}, data::DataFrame;
    d_ref::Union{Real, Nothing} = nothing)
    """We use Extremes.jl to optimize (with the moments) the parameters values"""

    D_values = to_duration.(names(data))
    
    params_init = []
    for d in D_values
        
        μ, ϕ = Extremes.gumbelfitpwm(dropmissing(data, Symbol(to_french_name(d))), Symbol(to_french_name(d))).θ̂
        params_init = [params_init; [μ, exp(ϕ)]]
        
    end

    params_init = Float64.(params_init)
    return NoScalingGumbelModel(D_values, params_init)

end

function estimSimpleScalingModel(model::NoScalingGumbelModel;
                                    d_ref::Union{Real, Nothing} = nothing)
    """Returns an simple scaling model parametrized by an estimation of the simple scaling relationship parameters  (μ_d_ref, σ_d_ref, α)
    based on a linear regression of μ_d and σ_d depending on d. 
    """

    D_values = model.D_values
    if isnothing(d_ref)
        d_ref = maximum(D_values)
    end

    θ = transformParams(model)

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
    α_init = maximum( [ minimum([α_init, 0.99]), 0.01 ] ) # α must be between 0 and 1
    index_d_ref = argmax(D_values.== d_ref)
    x_init = [logistic(α_init), θ[2*index_d_ref-1], θ[2*index_d_ref]]

    # optimization
    result = optimize(error, x_init, BFGS())

    # model construction
    SS_model = SimpleScalingModel(d_ref, 
                        getParams(SimpleScalingModel, [result.minimizer[2], result.minimizer[3], 0.0, result.minimizer[1]])
                        )

    return SS_model

end

function estimdGEVModel(model::NoScalingGumbelModel;
                            d_ref::Union{Real, Nothing} = nothing)
    """Returns a DGEV model parametrized by an estimation of the dGEV parameters  (μ_d_ref, σ_d_ref, α, δ)
    based on a linear regression of μ_d and σ_d depending on d. 
    """

    D_values = model.D_values
    if isnothing(d_ref)
        d_ref = maximum(D_values)
    end

    θ = transformParams(model)

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
    SS_model = estimSimpleScalingModel(model, d_ref = d_ref)
    transformed_params = transformParams(SS_model)
    x_init = [transformed_params[4], 0.0, transformed_params[1], transformed_params[2]]

    # optimization
    result = optimize(error, x_init, BFGS())

    # model construction
    dGEV_model = dGEVModel(d_ref, 
                        getParams(dGEVModel, [result.minimizer[3], result.minimizer[4], 0.0, result.minimizer[1], result.minimizer[2]])
                        )

    return dGEV_model

end