module IDF

    using Distributions, Gadfly, DataFrames, Extremes, ForwardDiff, Optim, LinearAlgebra, Copulas, GLM, Random, MambaLite

    # include("MambaLite/src/MambaLite.jl") # for now. After it will be in the "using" line

    include("utils.jl")
    include("IDFModel.jl")
    include("fitted.jl")
    include("parameterestimation.jl")
    include("testing.jl")

    export

        # structures
        IDFModel,
        dGEVModel,
        NoScalingGumbelModel,
        NoScalingGEVModel,
        SimpleScalingModel,
        FittedMLE,
        TestSimpleScalingModel,

        # methods
        to_french_name,
        to_duration,

        getDistribution, 
        sample,
        logpdf,

        estimSimpleScalingModel,
        estimdGEVModel,
        initializeModel,

        getEstimatedModel,
        cint,
        returnLevelEstimation,
        returnLevelCint,
        
        fitMLE

        
end