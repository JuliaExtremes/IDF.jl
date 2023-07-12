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
    """For now evrything will lie in the initialization"""
    return 0.0
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