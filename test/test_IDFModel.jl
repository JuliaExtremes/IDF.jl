@testset "IDFModel.jl" begin

    include(joinpath("models", "test_noScalingGumbel.jl"))
    include(joinpath("models", "test_noScaling.jl"))
    include(joinpath("models", "test_simpleScaling.jl"))
    include(joinpath("models", "test_dGEV.jl"))
    include(joinpath("models", "test_compositeScaling.jl"))


    D_values = [5, 15, 360]
    d_ref = 60

    params_gumbel = [9,6,8.2,5.5,3,2]
    noscaling_gumbel_model = IDF.NoScalingGumbelModel(D_values, params_gumbel)

    params_gev = [9,6,0.1,8.3,5.5,0.12,3,2,0.09]
    noscaling_gev_model = IDF.NoScalingGEVModel(D_values, params_gev)
    
    params_SS = [3,2,0.1,0.7]
    SS_model = IDF.SimpleScalingModel(d_ref, params_SS)
    
    params_dGEV = [3,2,0.1,0.7,3]
    dGEV_model = IDF.dGEVModel(d_ref, params_dGEV)


    @testset "getDistribution()" begin

        @test_throws AssertionError IDF.getDistribution(noscaling_gumbel_model, [1,12,480]) 

        multiv_distrib = IDF.getDistribution(noscaling_gev_model, D_values)
        @test multiv_distrib.v[2] == IDF.getDistribution(noscaling_gev_model, 15)

        multiv_distrib = IDF.getDistribution(dGEV_model, [1,12,480])
        @test multiv_distrib.v[3] ≈ IDF.Distributions.GeneralizedExtremeValue(
                                                params_dGEV[1]*( (d_ref + params_dGEV[5]) / (480 + params_dGEV[5]) )^params_dGEV[4],
                                                params_dGEV[2]*( (d_ref + params_dGEV[5]) / (480 + params_dGEV[5]) )^params_dGEV[4], 
                                                params_dGEV[3])

    end

    data_gumbel = IDF.sample(noscaling_gumbel_model, D_values)
    data_SS = IDF.sample(SS_model, [1,12,480], 30)

    @testset "sample()" begin

        @test names(data_gumbel) == IDF.to_french_name.(D_values)
        
        @test size(data_SS, 1) == 30
        @test IDF.to_duration.(names(data_SS)) ≈ [1,12,480]

    end

    @testset "logpdf()" begin

        multiv_distrib_gev = IDF.getDistribution(noscaling_gev_model, D_values)
        vectorized_data_gumbel = [Vector(data_gumbel[i,:]) for i in axes(data_gumbel,1)]
        @test IDF.logpdf(noscaling_gev_model, data_gumbel) ≈ sum(IDF.Distributions.logpdf(multiv_distrib_gev, vectorized_data_gumbel))

        multiv_distrib_dGEV = IDF.getDistribution(dGEV_model, [1,12,480])
        vectorized_data_SS = [Vector(data_SS[i,:]) for i in axes(data_SS,1)]
        @test IDF.logpdf(dGEV_model, data_SS) ≈ sum(IDF.Distributions.logpdf(multiv_distrib_dGEV, vectorized_data_SS))
    
    end

    @testset "returnLevel()" begin

        @test_throws AssertionError IDF.returnLevel(noscaling_gumbel_model, 60, 50)
        
        r1 = IDF.returnLevel(noscaling_gev_model, 15, 50)
        @test IDF.Distributions.cdf(IDF.Distributions.GeneralizedExtremeValue(8.3,5.5,0.12), r1) ≈ 1 - 1/50
        
        r2 = IDF.returnLevel(SS_model, 170, 30)
        @test IDF.Distributions.cdf(IDF.getDistribution(SS_model, 170), r2) ≈ 1 - 1/30
        
    end

end