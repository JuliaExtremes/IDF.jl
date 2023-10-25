@testset "testing.jl" begin

    include(joinpath("testing", "test_testGEVGEV.jl"))
    include(joinpath("testing", "test_testGEVGOF.jl"))

    n= 30
    D_values = [15,60,360]
    d_ref = 360

    dGEV_params = [3,2,0.1,0.7,4]
    dGEV_model = IDF.dGEVModel(d_ref, dGEV_params)
    dGEV_data = IDF.sample(dGEV_model, D_values, n)

    test_obj = IDF.TestGEVGEV(IDF.SimpleScalingModel, dGEV_data)
    test_obj2 = IDF.TestGEVGEV(IDF.dGEVModel, dGEV_data)

    @testset "statistic()" begin

        @test IDF.statistic(test_obj) == test_obj.statistic
    end

    @testset "rejectH0()" begin

        @test_throws AssertionError IDF.rejectH0(test_obj, 0.0)

        reject = IDF.rejectH0(test_obj, 0.02)
        @test reject == (IDF.Distributions.cdf(test_obj.H0_distrib, test_obj.statistic) > 1 - 0.02)
    end

    @testset "pvalue()" begin

        pval = IDF.pvalue(test_obj2)

        @test IDF.Distributions.cdf(test_obj2.H0_distrib, test_obj2.statistic) ≈ 1 - pval
        @test IDF.rejectH0(test_obj2, 2*pval)
    end

end