module IDF

    using Distributions, Gadfly, DataFrames, Extremes, ForwardDiff, Optim, LinearAlgebra, Copulas, GLM, Random, MambaLite, SpecialFunctions, Roots

    import Base.minimum
    import Base.maximum

    import Distributions.minimum
    import Distributions.maximum
    import Distributions.insupport
    import Distributions.cdf
    import Distributions.quantile


    # include("MambaLite/src/MambaLite.jl") # for now. After it will be in the "using" line

    include("utils.jl")
    include("IDFModel.jl")
    include("fitted.jl")
    include("parameterestimation.jl")
    include("testing.jl")

    export

        # structures
        ZolotarevDistrib,
        IDFModel,
        dGEVModel,
        NoScalingGumbelModel,
        NoScalingGEVModel,
        SimpleScalingModel,
        FittedMLE,
        TestGEVGEV,
        TestGEVGOF,

        # methods
        to_french_name,
        to_duration,
        cvmcriterion,
        modifiedADcriterion,
        approx_eigenvalues,

        getDistribution, 
        sample,
        logpdf,

        estimSimpleScalingModel,
        estimdGEVModel,
        initializeModel,

        modelEstimation,
        cint,
        returnLevelEstimation,
        returnLevelCint,

        getChainParam,
        getChainFunction,

        fitMLE,
        fitBayesian,

        computeGEVGEVStatistic,
        computeGEVGOFStatistic,
        computeGEVGOFNullDistrib,

        rejectH0,
        pvalue
        
end