module IDF

    using Distributions, Gadfly, DataFrames, Extremes, ForwardDiff, Optim, LinearAlgebra, Copulas, GLM, Random, MambaLite

    # include("MambaLite/src/MambaLite.jl") # for now. After it will be in the "using" line

    include("utils.jl")
    include("IDFModel.jl")
    include("fitted.jl")
    include("parameterestimation.jl")

    export

        # structures
        dGEVModel,
        NoScalingGumbelModel,
        SimpleScalingModel,
        FittedMLE,

        # methods
        fitMLE
        
end