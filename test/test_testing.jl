@testset "testing.jl" begin

    include(joinpath("testing", "test_testingsimplescaling.jl"))
    include(joinpath("testing", "test_testingdGEV.jl"))

    @testset "statistic()" begin
    end

    @testset "rejectH0()" begin
    end

    @testset "pvalue()" begin
    end

end