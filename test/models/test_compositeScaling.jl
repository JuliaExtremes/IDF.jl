@testset "compsiteScaling.jl" begin
    d_ref = 360
    params = [3,2,0.1,0.7,0.9]
    model = IDF.CompositeScalingModel(d_ref, params)
    
    @testset "CompositeScalingModel()" begin

        @test model.d_ref == d_ref
        @test length(model.params_names) == 5
        @test model.params == params
        @test model.params_names == ["μ_360", "σ_360", "ξ_360", "α_μ", "α_σ"]

    end

    @testset "getDistribution()" begin

        μ, σ, ξ, α_μ, α_σ = model.params

        distrib_6h = IDF.getDistribution(model, 360)
        @test distrib_6h == IDF.Distributions.GeneralizedExtremeValue(μ, σ, ξ)

        distrib_1h = IDF.getDistribution(model, 60)
        @test distrib_1h ≈ IDF.Distributions.GeneralizedExtremeValue(μ*6^0.7, σ*6^0.9, ξ)

    end

    @testset "transformParams()" begin

        θ = IDF.transformParams(model)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[2] ≈ log(2)
        @test θ[3] ≈ IDF.logistic(0.6)
        @test θ[4] ≈ IDF.logistic(0.7)
        @test θ[5] ≈ IDF.logistic(0.9)

        params2 = [3,2,0.1,1.5,2.0]
        @test_throws DomainError IDF.transformParams(IDF.CompositeScalingModel(d_ref, params2))

    end


    @testset "getParams()" begin

        θ = IDF.transformParams(model)
        params3 = IDF.getParams(IDF.CompositeScalingModel, θ)
        @test params3 ≈ params

    end

    @testset "setParams()" begin

        θ = IDF.transformParams(model)
        θ[5] = IDF.logistic(0.8)
        model2 = IDF.setParams(model, θ)

        @test model2.d_ref == model.d_ref
        @test model2.params_names == model.params_names
        @test model2.params[2] ≈ model.params[2]
        @test model2.params[4] ≈ model.params[4]
        @test model2.params[5] ≈ 0.8

        params2 = params
        params2[5] = 0.8
        model3 = IDF.setParams(model, params2, is_transformed = false)
        @test model2.params ≈ model3.params

    end

    # n = 30
    # μ, σ, ξ, α = params
    # data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( 5 / 360 ) ^ (-α), σ * ( 5 / 360 ) ^ (-α), ξ), n),
    #                 Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 60 / 360 ) ^ (-α), σ * ( 60 / 360 ) ^ (-α), ξ), n),
    #                 Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 360 / 360 ) ^ (-α), σ * ( 360 / 360 ) ^ (-α), ξ), n))

    
    # init_model = IDF.initializeModel(IDF.SimpleScalingModel, data)

    @testset "initializeModel()" begin

        # TODO

    end

    @testset "gradF_dref()" begin

        # TODO

    end

end