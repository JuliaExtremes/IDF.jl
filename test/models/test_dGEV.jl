@testset "dGEV.jl" begin
    d_ref = 360
    params = [3,2,0.1,0.7,3]
    model = IDF.dGEVModel(d_ref, params)
    
    @testset "dGEVModel()" begin

        @test model.d_ref == d_ref
        @test length(model.params_names) == 5
        @test model.params == params
        @test model.params_names == ["μ_360", "σ_360", "ξ_360", "α", "δ"]

    end

    @testset "getDistribution()" begin

        μ, σ, ξ, α, δ = model.params

        distrib_6h = IDF.getDistribution(model, 360)
        @test distrib_6h == IDF.Distributions.GeneralizedExtremeValue(μ, σ, ξ)

        distrib_1h = IDF.getDistribution(model, 60)
        @test distrib_1h ≈ IDF.Distributions.GeneralizedExtremeValue(μ*( (360 + δ) / (60 + δ) )^α, σ*( (360 + δ) / (60 + δ) )^α, ξ)

    end

    @testset "transformParams()" begin

        θ = IDF.transformParams(model)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[2] ≈ log(2)
        @test θ[3] ≈ IDF.logistic(0.6)
        @test θ[4] ≈ IDF.logistic(0.7)
        @test θ[5] ≈ log(3)

        params2 = [3,2,0.1,0.7,-2]
        @test_throws DomainError IDF.transformParams(IDF.dGEVModel(d_ref, params2))

    end

    @testset "getParams()" begin

        θ = IDF.transformParams(model)
        params3 = IDF.getParams(IDF.dGEVModel, θ)
        @test params3 ≈ params

    end

    @testset "setParams()" begin

        θ = IDF.transformParams(model)
        θ[5] = log(4)
        model2 = IDF.setParams(model, θ)

        @test model2.d_ref == model.d_ref
        @test model2.params_names == model.params_names
        @test model2.params[4] ≈ model.params[4]
        @test model2.params[5] ≈ 4

        params2 = params
        params2[5] = 4
        model3 = IDF.setParams(model, params2, is_transformed = false)
        @test model2.params ≈ model3.params

    end

    n = 30
    μ, σ, ξ, α, δ = params
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( (5 + δ) / (360 + δ) ) ^ (-α), σ * ( (5 + δ) / (360 + δ) ) ^ (-α), ξ), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( (60 + δ) / (360 + δ) ) ^ (-α), σ * ( (60 + δ) / (360 + δ) ) ^ (-α), ξ), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( (360 + δ) / (360 + δ) ) ^ (-α), σ * ( (360 + δ) / (360 + δ) ) ^ (-α), ξ), n))
  
    
    init_model = IDF.initializeModel(IDF.dGEVModel, data)

    @testset "initializeModel()" begin

        @test init_model.d_ref == 360

        no_scaling_model = IDF.modelEstimation(IDF.fitMLE(IDF.NoScalingGumbelModel,data))
        estim_ss_model = IDF.estimdGEVModel(no_scaling_model)

        @test init_model.params ≈ estim_ss_model.params

    end

    @testset "gradF_dref()" begin

        x = 3.0
        fun(θ) = IDF.cdf(IDF.getDistribution(IDF.setParams(model, θ), d_ref), x)
        
        θ₀ = IDF.transformParams(model)
        @test IDF.gradF_dref(IDF.dGEVModel, x, θ₀) ≈ IDF.ForwardDiff.gradient(fun, θ₀)

    end

end