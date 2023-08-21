@testset "noScaling.jl" begin
    D_values = [5, 60, 360]
    params = [9,6,0.1,6,4,0.12,3,2,0.09]
    model = IDF.NoScalingGEVModel(D_values, params)
    
    @testset "NoScalingGEVModel()" begin
        @test model.D_values == D_values
        @test model.params == params
        @test length(model.params_names) == length(D_values)*3 == length(params)
        
        i = rand(1:length(model.params_names))
        @test model.params_names[i][1] == ( (i%3 == 1) ? 'μ' : (
                                            (i%3 == 2) ? 'σ' : 'ξ') 
                                            )
    end

    @testset "getDistribution()" begin

        @test_throws AssertionError IDF.getDistribution(model, 30)

        distrib_1h = IDF.getDistribution(model, 60)
        @test distrib_1h == IDF.Distributions.GeneralizedExtremeValue(params[4], params[5], params[6])

    end

    @testset "transformParams()" begin

        θ = IDF.transformParams(model)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(9)
        @test θ[5] ≈ log(4)
        @test θ[6] ≈ IDF.logistic(0.12+0.5)

        params2 = [3,2,0.1,6,4,0.1,8,-5,0.1]
        @test_throws DomainError IDF.transformParams(IDF.NoScalingGEVModel(D_values, params2))

    end

    @testset "getParams()" begin

        θ = IDF.transformParams(model)
        params3 = IDF.getParams(IDF.NoScalingGEVModel, θ)
        @test params3 ≈ params

    end

    @testset "setParams()" begin

        θ = IDF.transformParams(model)
        θ[4] = log(7)
        model2 = IDF.setParams(model, θ)

        @test model2.D_values == model.D_values
        @test model2.params_names == model.params_names
        @test model2.params[8] ≈model.params[8]
        @test model2.params[4] ≈ 7

    end

    n = 30
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(params[1], params[2], params[3]), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(params[4], params[5], params[6]), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(params[7], params[8], params[9]), n))


    

    init_model = IDF.initializeModel(IDF.NoScalingGEVModel,data)


    @testset "initializeModel()" begin

        @test length(init_model.params) == 3*length(D_values)
        
        no_scaling_gumbel_model = IDF.initializeModel(IDF.NoScalingGumbelModel, data)
        d = D_values[2]
        μ, ϕ = IDF.Extremes.gumbelfitpwm(data, Symbol(IDF.to_french_name(d))).θ̂
        @test init_model.params[1] ≈ no_scaling_gumbel_model.params[1]
        @test init_model.params[5] ≈ exp(ϕ)
        @test init_model.params[9] ≈ 0.0

    end

    d_ref = maximum(D_values)
    SS_model = IDF.estimSimpleScalingModel(init_model, d_ref=d_ref)

    @testset "estimSimpleScalingModel()" begin

        @test typeof(SS_model) == IDF.SimpleScalingModel

        gumbel_model = IDF.NoScalingGumbelModel(D_values, [9,6,6,4,3,2])
        SS_model_estim_gumbel = IDF.estimSimpleScalingModel(init_model, d_ref = d_ref)
        @test SS_model.params ≈ SS_model_estim_gumbel.params

    end

    dGEV_model = IDF.estimdGEVModel(init_model, d_ref=d_ref)

    @testset "estimdGEVModel()" begin

        @test typeof(dGEV_model) == IDF.dGEVModel

        gumbel_model = IDF.NoScalingGumbelModel(D_values, [9,6,6,4,3,2])
        dGEV_model_estim_gumbel = IDF.estimdGEVModel(init_model, d_ref = d_ref)
        @test dGEV_model.params ≈ dGEV_model_estim_gumbel.params
        
    end

end