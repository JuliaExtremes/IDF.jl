@testset "testGEVGEV.jl" begin

    n= 30
    D_values = [15,60,360]
    d_ref = 360

    SS_params = [3,2,0.1,0.7]
    SS_model = IDF.SimpleScalingModel(d_ref, SS_params)
    SS_data = IDF.sample(SS_model, D_values, n)

    @testset "computeGEVGEVStatistic()" begin

        stat = IDF.computeGEVGEVStatistic(IDF.SimpleScalingModel, SS_data, 60)

        d_out_data = SS_data[:,IDF.DataFrames.names(SS_data) .== IDF.to_french_name(60)]
        fitted_d_out = IDF.fitMLE(IDF.NoScalingGEVModel, d_out_data)
        one_out_data = IDF.DataFrames.select(SS_data, ["15 min", "6 h"])
        fitted_IDF_one_out = IDF.fitMLE(IDF.SimpleScalingModel, one_out_data, d_ref = 60)
        Σ = IDF.LinearAlgebra.I/fitted_d_out.I_Fisher + (IDF.LinearAlgebra.I/fitted_IDF_one_out.I_Fisher)[1:3, 1:3]
        statistic = IDF.LinearAlgebra.norm( sqrt(IDF.LinearAlgebra.I/Σ) * (fitted_d_out.θ̂ .- fitted_IDF_one_out.θ̂[1:3]) )^2
        
        @test stat ≈ statistic

    end

    dGEV_params = [3,2,0.1,0.7,4]
    dGEV_model = IDF.dGEVModel(d_ref, dGEV_params)
    dGEV_data = IDF.sample(dGEV_model, D_values, n)

    @testset "TestGEVGEV()" begin

        test_obj = IDF.TestGEVGEV(IDF.SimpleScalingModel, dGEV_data)

        @test test_obj.model_type == IDF.SimpleScalingModel
        @test test_obj.d_out == 15
        @test test_obj.H0_distrib == IDF.Distributions.Chi(3)

        test_obj = IDF.TestGEVGEV(IDF.dGEVModel, dGEV_data, d_out = 60)

        @test test_obj.model_type == IDF.dGEVModel
        @test test_obj.data == dGEV_data
        @test test_obj.statistic ≈ IDF.computeGEVGEVStatistic(IDF.dGEVModel, dGEV_data, 60)

    end

end