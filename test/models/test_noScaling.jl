@testset "noScaling.jl" begin
    D_values = [5, 60, 360]
    model = IDF.NoScalingGEVModel(D_values)
    
    @testset "NoScalingGEVModel()" begin
        @test model.D_values == D_values
        @test length(model.params_names) == length(D_values)*3
        
        i = rand(1:length(model.params_names))
        @test model.params_names[i][1] == ( (i%3 == 1) ? 'μ' : (
                                            (i%3 == 2) ? 'σ' : 'ξ') 
                                            )
    end

    @testset "transformParams()" begin

        params1 = [3,2,0.1,6,4,0.1,8,-5,0.1]
        @test_throws DomainError IDF.transformParams(model, params1)

        params2 = [3,2,0.1,6,4,0.1,8,5,0.1]
        θ = IDF.transformParams(model, params2)
        @test length(params2) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[5] ≈ log(4)
        @test θ[6] ≈ IDF.logistic(0.1+0.5)

    end

    @testset "getParams()" begin

        params2 = [3,2,0.1,6,4,0.1,8,5,0.1]
        θ = convert(Array{Float64,1},  IDF.transformParams(model, params2))
        params3 = IDF.getParams(model, θ)
        @test params3 ≈ params2

    end

    params_value = [9,6,0.1,6,4,0.12,3,2,0.09]
    θ = convert(Array{Float64,1},  IDF.transformParams(model, params_value))
    n = 30
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(params_value[1], params_value[2], params_value[3]), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(params_value[4], params_value[5], params_value[6]), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(params_value[7], params_value[8], params_value[9]), n))


    @testset "logLikelihood()" begin

        loglike = IDF.logLikelihood(model, data, θ)

        #Calculus of the log-likelihood
        log_likelihood = 0.0
        for i in eachindex(D_values)
            d = D_values[i]

            μ_d = params_value[3*i-2]
            σ_d = params_value[3*i-1]
            ξ_d = params_value[3*i]

            distrib_extreme_d = IDF.GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
            log_likelihood = log_likelihood + sum( IDF.logpdf.(Ref(distrib_extreme_d), 
                data[:,Symbol(IDF.to_french_name(d))]) )
        end

        @test loglike ≈ log_likelihood

    end

    θ_init = IDF.initializeParams(model,data)

    @testset "initializeParams()" begin

        @test length(θ_init) == 3*length(D_values)

        no_scaling_gumbel_model = IDF.NoScalingGumbelModel(model.D_values)
        fitted_no_scaling_gumbel_model = IDF.fitMLE(no_scaling_gumbel_model, data)
        @test θ_init[1] ≈ fitted_no_scaling_gumbel_model.θ̂[1]
        @test θ_init[5] ≈ fitted_no_scaling_gumbel_model.θ̂[4]
        @test θ_init[9] ≈ 0.0

    end

    @testset "estimSimpleScalingRelationship()" begin

        estim_GEV = IDF.estimSimpleScalingRelationship(model, θ)
        gumbel_model = IDF.NoScalingGumbelModel(model.D_values)
        gumbel_params = [9,6,6,4,3,2]
        estim_gumbel = IDF.estimSimpleScalingRelationship(gumbel_model, IDF.transformParams(gumbel_model, gumbel_params))
        @test estim_GEV == estim_gumbel

    end

    @testset "estimIDFRelationship()" begin

        estim_GEV = IDF.estimIDFRelationship(model, θ)
        gumbel_model = IDF.NoScalingGumbelModel(model.D_values)
        gumbel_params = [9,6,6,4,3,2]
        estim_gumbel = IDF.estimIDFRelationship(gumbel_model, IDF.transformParams(gumbel_model, gumbel_params))
        @test estim_GEV == estim_gumbel
        
    end

    @testset "returnLevel()" begin

        @test_throws AssertionError IDF.returnLevel(model, θ_init, 1.5, 50)

        returnlevel = IDF.returnLevel(model, θ_init, D_values[2], 50)
        params = IDF.getParams(model, θ_init)
        μ_d, σ_d, ξ_d = params[4], params[5], params[6]
        distrib_extreme_d = IDF.GeneralizedExtremeValue(μ_d, σ_d, ξ_d)
        @test IDF.cdf(distrib_extreme_d, returnlevel) ≈ 1 - 1/50

    end

end