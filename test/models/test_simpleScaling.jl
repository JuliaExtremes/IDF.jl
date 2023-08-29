@testset "simpleScaling.jl" begin
    d_ref = 360
    params = [3,2,0.1,0.7]
    model = IDF.SimpleScalingModel(d_ref, params)
    
    @testset "SimpleScalingModel()" begin

        @test model.d_ref == d_ref
        @test length(model.params_names) == 4
        @test model.params == params
        @test model.params_names == ["μ_360", "σ_360", "ξ_360", "α"]

    end

    @testset "getDistribution()" begin

        μ, σ, ξ, α = model.params

        distrib_6h = IDF.getDistribution(model, 360)
        @test distrib_6h == IDF.Distributions.GeneralizedExtremeValue(μ, σ, ξ)

        distrib_1h = IDF.getDistribution(model, 60)
        @test distrib_1h ≈ IDF.Distributions.GeneralizedExtremeValue(μ*6^α, σ*6^α, ξ)

    end

    @testset "transformParams()" begin

        θ = IDF.transformParams(model)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[2] ≈ log(2)
        @test θ[3] ≈ IDF.logistic(0.6)
        @test θ[4] ≈ IDF.logistic(0.7)

        params2 = [3,2,0.1,1.5]
        @test_throws DomainError IDF.transformParams(IDF.SimpleScalingModel(d_ref, params2))

    end


    @testset "getParams()" begin

        θ = IDF.transformParams(model)
        params3 = IDF.getParams(IDF.SimpleScalingModel, θ)
        @test params3 ≈ params

    end

    @testset "setParams()" begin

        θ = IDF.transformParams(model)
        θ[4] = IDF.logistic(0.8)
        model2 = IDF.setParams(model, θ)

        @test model2.d_ref == model.d_ref
        @test model2.params_names == model.params_names
        @test model2.params[2] ≈ model.params[2]
        @test model2.params[4] ≈ 0.8

    end

    n = 30
    μ, σ, ξ, α = params
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( 5 / 360 ) ^ (-α), σ * ( 5 / 360 ) ^ (-α), ξ), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 60 / 360 ) ^ (-α), σ * ( 60 / 360 ) ^ (-α), ξ), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 360 / 360 ) ^ (-α), σ * ( 360 / 360 ) ^ (-α), ξ), n))

    
    init_model = IDF.initializeModel(IDF.SimpleScalingModel, data)

    @testset "initializeModel()" begin

        @test init_model.d_ref == 360

        no_scaling_model = IDF.modelEstimation(IDF.fitMLE(IDF.NoScalingGumbelModel,data))
        estim_ss_model = IDF.estimSimpleScalingModel(no_scaling_model)

        @test init_model.params ≈ estim_ss_model.params

    end

end