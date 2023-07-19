function logPosterior(model::IDFModel, data::DataFrame, θ;
                        prior_distribs::Dict{String,ContinuousUnivariateDistribution} = Dict{String,ContinuousUnivariateDistribution}())

    log_posterior = logLikelihood(model, data, θ)

    for i in eachindex(model.params_names)

        name_param = model.params_names[i]
        if haskey(prior_distribs, name_param)
            prior_distrib = prior_distribs[name_param]
        else
            prior_distrib = Extremes.Flat()
        end

        @assert ( maximum(prior_distrib) == Inf && minimum(prior_distrib) == -Inf ) "The given prior distribution for" *
        " the parameter " * name_param *" doesn't fit. The prior must concern the transformation of the parameter." *
        " Hence, its support must be the set of all real numbers."

        log_posterior = log_posterior + logpdf(prior_distrib, θ[i])
    end

    return log_posterior

end

function fitBayesian(model::IDFModel, data::DataFrame;
                        prior_distribs::Dict{String,ContinuousUnivariateDistribution} = Dict{String,ContinuousUnivariateDistribution}(),
                        initialvalues::Union{Vector{<:Real}, Nothing} = nothing,
                        print_evolution::Bool = false,
                        niter = 500, warmup = 100)

    # Define the loglikelihood function and the gradient for the NUTS algorithm
    logf(θ::DenseVector) = logPosterior(model, data, θ, prior_distribs = prior_distribs)
    Δlogf(θ::DenseVector) = ForwardDiff.gradient(logf, θ)
    function logfgrad(θ::DenseVector)
        ll = logf(θ)
        g = Δlogf(θ)
        return ll, g
    end

    # initialization
    if isnothing(initialvalues)
        initialvalues = fitMLE(model, data).θ̂
    end

    if print_evolution
        print("Inital value for θ : ")
        println(initialvalues)

        print("Log-posterior density for the inital value of θ : ")
        println(logfgrad(initialvalues)[1])

        print("Grad of the log-posterior density for the inital value of θ : ")
        println(logfgrad(initialvalues)[2])
    end

    names_params = model.params_names

    # computing the chains
    sim = MambaLite.Chains(niter, length(initialvalues), start = (warmup + 1), names = names_params)
    θ = MambaLite.NUTSVariate(initialvalues, logfgrad)
    for i in 1:niter
        MambaLite.sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    return FittedBayesian(model, sim)

end