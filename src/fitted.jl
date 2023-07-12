abstract type Fitted{T<:IDFModel} end

include(joinpath("fittedmodels", "fittedMLE.jl"))
include(joinpath("fittedmodels", "fittedBayesian.jl"))
