@testset "testGEVGOF.jl" begin

    n= 30
    D_values = [15,60,360]
    d_ref = 360

    dGEV_params = [3,2,0.1,0.7,4]
    dGEV_model = IDF.dGEVModel(d_ref, dGEV_params)
    data = IDF.sample(dGEV_model, D_values, n)

    result_CVM = IDF.computeGEVGOFStatistic(IDF.SimpleScalingModel, data, 60)
    result_AD = IDF.computeGEVGOFStatistic(IDF.SimpleScalingModel, data, 60, criterion = "modified AD")

    @testset "computeGEVGOFStatistic()" begin

        one_out_data = data[:,IDF.DataFrames.names(data) .!= IDF.to_french_name(60)]
        fitted_IDF_one_out = IDF.fitMLE(IDF.SimpleScalingModel, one_out_data, d_ref = 60)
        estim_distrib_d_out = IDF.getDistribution(IDF.modelEstimation(fitted_IDF_one_out), 60)
        empirical_distrib_d_out = IDF.DataFrames.select( data, Symbol(IDF.to_french_name(60)) )[:,Symbol(IDF.to_french_name(60))]

        
        @test result_CVM[2] == result_AD[2] == IDF.modelEstimation(fitted_IDF_one_out)
        @test result_CVM[3] ≈ result_AD[3] ≈ fitted_IDF_one_out.I_Fisher
        @test result_CVM[1] ≈ IDF.cvmcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))
        @test result_AD[1] ≈ IDF.modifiedADcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))


    end

    @testset "get_g()" begin

        function g(u)

            θ̂ = IDF.transformParams(dGEV_model)
            x = IDF.Distributions.quantile(IDF.getDistribution(dGEV_model, 5), u)
    
            return IDF.gradF_dref(IDF.dGEVModel, x, θ̂) 

        end

        @test IDF.get_g(dGEV_model, 5)(0.37) ≈ g(0.37)

    end

    @testset "computeGEVGOFNullDistrib()" begin

        estim_model = result_CVM[2]
        I_cvm = result_CVM[3] / n
        I_ad = result_AD[3] / n
        g = IDF.get_g(estim_model, estim_model.d_ref)

        ρ1(u,v) = minimum([u,v]) - u*v + Float64(g(u)'/I_cvm*g(v))
        λs = IDF.approx_eigenvalues(ρ1, 50)
        @test IDF.computeGEVGOFNullDistrib(estim_model, I_cvm, "cvm", k=50).λs ≈ λs

        ρ2(u,v) = sqrt(1/((1-u)*(1-v))) * ( minimum([u,v]) - u*v + Float64(g(u)'/I_ad*g(v)) )
        λs = IDF.approx_eigenvalues(ρ2, 200)
        @test IDF.computeGEVGOFNullDistrib(estim_model, I_ad, "modified AD", k=200).λs ≈ λs

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

        H0_distrib = IDF.computeGEVGOFNullDistrib(test_obj.estim_model, test_obj.I_Fisher/n, "cvm", k=100)

        @test test_obj.H0_distrib == H0_distrib

    end


end