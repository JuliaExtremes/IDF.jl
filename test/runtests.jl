# using IDF
include(joinpath(dirname(@__DIR__), "src/IDF.jl"))
using Test, GLM

@testset "IDF.jl" begin

    include("test_utils.jl")
    include("test_IDFModel.jl")
    include("test_fitted.jl")
    include("test_parameterestimation.jl")
    #include("test_testing.jl")

end
