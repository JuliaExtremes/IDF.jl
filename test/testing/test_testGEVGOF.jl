@testset "testGEVGOF.jl" begin

    n= 30
    D_values = [15,60,360]
    d_ref = 360

    dGEV_params = [3,2,0.1,0.7,4]
    dGEV_model = IDF.dGEVModel(d_ref, dGEV_params)
    data = IDF.sample(dGEV_model, D_values, n)

    @testset "computeGEVGOFStatistic()" begin

        one_out_data = data[:,IDF.DataFrames.names(data) .!= IDF.to_french_name(60)]
        fitted_IDF_one_out = IDF.fitMLE(IDF.SimpleScalingModel, one_out_data, d_ref = 60)
        estim_distrib_d_out = IDF.getDistribution(IDF.modelEstimation(fitted_IDF_one_out), 60)
        empirical_distrib_d_out = IDF.DataFrames.select( data, Symbol(IDF.to_french_name(60)) )[:,Symbol(IDF.to_french_name(60))]

        result_CVM = IDF.computeGEVGOFStatistic(IDF.SimpleScalingModel, data, 60)
        result_AD = IDF.computeGEVGOFStatistic(IDF.SimpleScalingModel, data, 60, criterion = "modified AD")

        @test result_CVM[2] == result_AD[2] == IDF.modelEstimation(fitted_IDF_one_out)
        @test result_CVM[3] ≈ result_AD[3] ≈ fitted_IDF_one_out.I_Fisher
        @test result_CVM[1] ≈ IDF.cvmcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))
        @test result_AD[1] ≈ IDF.modifiedADcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))


    end

    @testset "TestGEVGOF()" begin

        test_obj = IDF.TestGEVGOF(IDF.dGEVModel, data)

        @test test_obj.model_type == IDF.dGEVModel
        @test test_obj.data == data
        @test test_obj.d_out == 15
        @test test_obj.criterion == "cvm"

        result_CVM = IDF.computeGEVGOFStatistic(IDF.dGEVModel, data, 15)

        @test test_obj.statistic == result_CVM[1]
        @test test_obj.estim_model == result_CVM[2]
        @test test_obj.I_Fisher == result_CVM[3]
        @test isnothing(test_obj.H0_distrib)

    end


end