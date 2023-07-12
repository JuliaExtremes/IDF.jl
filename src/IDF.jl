module IDF

    using Distributions, Gadfly, DataFrames, Extremes, ForwardDiff, Optim, LinearAlgebra, Copulas, MambaLite, GLM, Random

    include("utils.jl")
    include("IDFModel.jl")
    include("fitted.jl")
    include("parameterestimation.jl")

    export

        # structures
        dGEVModel,
        NoScalingGumbelModel,
        FittedMLE,

        # methods
        fitMLE

end