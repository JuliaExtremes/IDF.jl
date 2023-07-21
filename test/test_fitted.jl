@testset "fitted.jl" begin

    include(joinpath("fittedmodels", "test_fittedMLE.jl"))
    include(joinpath("fittedmodels", "test_fittedBayesian.jl"))

end
