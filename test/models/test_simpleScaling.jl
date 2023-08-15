@testset "simpleScaling.jl" begin
    D_values = [5, 60, 360]
    model = IDF.SimpleScalingModel(D_values)
    
    @testset "SimpleScalingModel()" begin

        @test model.D_values == D_values
        @test model.d_ref == maximum(D_values)
        @test length(model.params_names) == 4

        model2 = IDF.SimpleScalingModel(D_values, d_ref = 60)
        @test model2.d_ref == 60
        @test model2.params_names == ["μ_60", "σ_60", "ξ_60", "α"]

    end

    params = [3,2,0.1,0.7]

    @testset "transformParams()" begin

        params2 = [3,2,0.1,1.5]
        @test_throws DomainError IDF.transformParams(model, params2)

        θ = IDF.transformParams(model, params)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[2] ≈ log(2)
        @test θ[3] ≈ IDF.logistic(0.6)
        @test θ[4] ≈ IDF.logistic(0.7)

    end

    @testset "getParams()" begin

        θ = IDF.transformParams(model, params)
        params3 = IDF.getParams(model, θ)
        @test params3 ≈ params

    end

    n = 30
    μ, σ, ξ, α = params
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( 5 / 360 ) ^ (-α), σ * ( 5 / 360 ) ^ (-α), ξ), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 60 / 360 ) ^ (-α), σ * ( 60 / 360 ) ^ (-α), ξ), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 360 / 360 ) ^ (-α), σ * ( 360 / 360 ) ^ (-α), ξ), n))

    @testset "logLikelihood()" begin

        d_ref = model.d_ref

        θ = IDF.transformParams(model, params)
        loglike = IDF.logLikelihood(model, data, θ)

        #Calculus of the log-likelihood
        log_likelihood = 0.0
        for i in eachindex(D_values)
            d = D_values[i]

            μ_d = μ * ( d / d_ref ) ^ (-α)
            σ_d = σ * ( d / d_ref ) ^ (-α)
            ξ_d = ξ

            distrib_extreme_d = IDF.GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
            log_likelihood = log_likelihood + sum( IDF.logpdf.(Ref(distrib_extreme_d), 
                                                    data[:,Symbol(IDF.to_french_name(d))]) )

        end

        @test loglike ≈ log_likelihood

    end

    θ_init = IDF.initializeParams(model,data)

    @testset "initializeParams()" begin

        @test length(θ_init) == length(model.params_names)

        no_scaling_model = IDF.NoScalingGumbelModel(D_values)
        fitted_no_scaling_model = IDF.fitMLE(no_scaling_model, data)
        estimated_simple_scaling = IDF.estimSimpleScalingRelationship(no_scaling_model, fitted_no_scaling_model.θ̂, d_ref = model.d_ref)

        @test θ_init[1] ≈ estimated_simple_scaling[2]
        @test θ_init[2] ≈ estimated_simple_scaling[3]
        @test θ_init[3] ≈ 0.0
        @test θ_init[4] ≈ estimated_simple_scaling[1]

    end

    @testset "returnLevel()" begin

        returnlevel = IDF.returnLevel(model, θ_init, 120, 50)
        params = IDF.getParams(model, θ_init)
        μ, σ, ξ, α = params
        μ_d, σ_d, ξ_d = μ * ( 120 / 360 ) ^ (-α), σ * ( 120 / 360 ) ^ (-α), ξ
        distrib_extreme_d = IDF.GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
        @test IDF.cdf(distrib_extreme_d, returnlevel) ≈ 1 - 1/50


    end

end