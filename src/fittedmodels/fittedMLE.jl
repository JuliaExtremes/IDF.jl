struct FittedMLE{T} <: Fitted{T}
    model::T
    θ̂::Vector{Float64}
    I_Fisher::Matrix{Float64}
end

function cint(fitted_mle::FittedMLE, g::Function;
                p::Real = 0.95)
    """Returns an asymptotic confidence interval for g(θ̂), at confidence level p, based on the asymptotic normality of the MLE.
    """

    Δg(θ::DenseVector) = ForwardDiff.gradient(g, θ)
    G = Δg(fitted_mle.θ̂)
    var = G'/fitted_mle.I_Fisher*G

    asymptotic_distrib = Normal(g(fitted_mle.θ̂), sqrt(var))

    return [quantile(asymptotic_distrib, (1-p)/2), quantile(asymptotic_distrib, (1+p)/2)]
end

function returnLevelEstimation(fitted_mle::FittedMLE, d::Real, T::Real)
    """Renvoie l'estimation ponctuelle du niveau de retour associé à une durée d'accumulation d et à un temps de retour T"""

    return returnLevel(fitted_mle.model, fitted_mle.θ̂, d, T)
end

function returnLevelCint(fitted_mle::FittedMLE, d::Real, T::Real;
                            p::Real = 0.95)
    """Renvoie un intervalle de confiance pour le niveau de retour associé à une durée d'accumulation d et à un temps de retour T"""

    g_return_level(θ) = returnLevel(fitted_mle.model, θ, d, T)

    return  cint(fitted_mle, g_return_level, p=p)
end
