@testset "parameterestimation.jl" begin

    include(joinpath("parameterestimation", "test_maxLikelihood.jl"))
    #include(joinpath("parameterestimation", "test_bayesian.jl"))

end