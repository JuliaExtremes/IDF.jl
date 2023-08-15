@testset "IDFModel.jl" begin

    include(joinpath("models", "test_noScalingGumbel.jl"))
    include(joinpath("models", "test_noScaling.jl"))
    include(joinpath("models", "test_simpleScaling.jl"))
    include(joinpath("models", "test_dGEV.jl"))

end