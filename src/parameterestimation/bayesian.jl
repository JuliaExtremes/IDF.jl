function logPosterior(model::IDFModel, data::DataFrame;
                        prior_distribs::Dict{String,ContinuousUnivariateDistribution} = Dict{String,ContinuousUnivariateDistribution}())

    log_posterior = logpdf(model, data)

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

        log_posterior = log_posterior + Extremes.logpdf(prior_distrib, transformParams(model)[i])
    end

    return log_posterior

end

function fitBayesian(model_type::Type{<:IDFModel}, data::DataFrame;
                        prior_distribs::Dict{String,ContinuousUnivariateDistribution} = Dict{String,ContinuousUnivariateDistribution}(),
                        initialmodel::Union{IDFModel, Nothing} = nothing,
                        d_ref::Union{Real, Nothing} = nothing,
                        print_evolution::Bool = false,
                        niter = 500, warmup = 100)

    # initialization
    if isnothing(initialmodel)
        initialmodel = modelEstimation( fitMLE(model_type, data, d_ref = d_ref) )
    end

    # Define the loglikelihood function and the gradient for the NUTS algorithm
    logf(θ::DenseVector) = logPosterior(setParams(initialmodel, θ), data, prior_distribs = prior_distribs)
    Δlogf(θ::DenseVector) = ForwardDiff.gradient(logf, θ)
    function logfgrad(θ::DenseVector)
        ll = logf(θ)
        g = Δlogf(θ)
        return ll, g
    end

    if print_evolution
        print("Inital value for the parameters : ")
        println(initialmodel.params)

        print("Log-posterior density for the inital value of the parameters : ")
        println(logfgrad(transformParams(initialmodel))[1])

        print("Grad of the log-posterior density for the inital value of the parameters : ")
        println(logfgrad(transformParams(initialmodel))[2])
    end

    names_params = initialmodel.params_names

    # computing the chains
    sim = MambaLite.Chains(niter, length(names_params), start = (warmup + 1), names = names_params)
    θ = MambaLite.NUTSVariate(transformParams(initialmodel), logfgrad)
    for i in 1:niter
        MambaLite.sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    return FittedBayesian(initialmodel, sim)

end