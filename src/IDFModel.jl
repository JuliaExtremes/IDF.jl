abstract type IDFModel <: ContinuousMultivariateDistribution end

include(joinpath("models", "noScalingGumbel.jl"))
include(joinpath("models", "noScaling.jl"))
include(joinpath("models", "simpleScaling.jl"))
include(joinpath("models", "dGEV.jl"))


function getDistribution(model::IDFModel, D_values::Vector{<:Real}) 
    """Returns the distribution of the maximal intensities for all durations in D_values, according to the specific model.
    For now independence btw durations is supposed true. Later we might add an argument 'copula'"""

    marginals = [getDistribution(model, d) for d in D_values]

    return Product(marginals)

end


function sample(model::IDFModel, D_values::Vector{<:Real}, n::Int64=1)
    """Returns a dataframe corresponding to a random sample of size n, of vectors of maximal intensities for all durations in D_values, 
    according to the specific model.
    """
    
    distrib = getDistribution(model, D_values)
    y = Distributions.rand(distrib, n)
    
    return DataFrame([to_french_name(D_values[i]) => y[i,:] for i in eachindex(D_values)])
    
end


function logpdf(model::IDFModel, data::DataFrame)
    """Returns the logpdf (or log-likelihood) of the model evaluated for the given dataframe"""

    D_values = to_duration.(names(data))
    multiv_distrib = getDistribution(model, D_values) 

    vectorized_data = [Vector(dropmissing(data)[i,:]) for i in axes(dropmissing(data),1)]

    return sum(Distributions.logpdf(multiv_distrib, vectorized_data))

end


function returnLevel(model::IDFModel, d::Real, T::Real)
    """Returns the return level associated to rain duration d and a return period T,
    according to the given model
    """

    @assert 1 < T "The return period must be bigger than 1 year."

    return quantile(getDistribution(model, d), 1 - 1/T)

end