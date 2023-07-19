abstract type IDFModel end

include(joinpath("models", "noScalingGumbel.jl"))
include(joinpath("models", "noScaling.jl"))
include(joinpath("models", "simpleScaling.jl"))
include(joinpath("models", "dGEV.jl"))