@testset "noScalingGumbel.jl" begin
    D_values = [5, 60, 360]
    params = [9,6,6,4,3,2]
    model = IDF.NoScalingGumbelModel(D_values, params)

    @testset "NoScalingGumbelModel()" begin

        @test model.D_values == D_values
        @test model.params == params

        @test length(model.params_names) == 2*length(D_values) == length(params)
        i = rand(1:length(model.params_names))
        @test model.params_names[i][1] == ((i%2 == 1) ? 'μ' : 'σ')
    end

    @testset "getDistribution()" begin

        @test_throws AssertionError IDF.getDistribution(model, 30)

        distrib_1h = IDF.getDistribution(model, 60)
        @test distrib_1h == IDF.Distributions.Gumbel(params[3], params[4])

    end

    @testset "transformParams()" begin

        θ = IDF.transformParams(model)
        @test length(params) == length(θ)
        @test θ[1] ≈ log(9)
        @test θ[4] ≈ log(4)

        params2 = [3,2,6,4,8,-5]
        @test_throws DomainError IDF.transformParams(IDF.NoScalingGumbelModel(D_values, params2))

    end

    @testset "getParams()" begin

        θ = IDF.transformParams(model)
        params3 = IDF.getParams(IDF.NoScalingGumbelModel, θ)
        @test params3 ≈ params

    end

    @testset "setParams()" begin

        θ = IDF.transformParams(model)
        θ[4] = log(1)
        model2 = IDF.setParams(model, θ)

        @test model2.D_values == model.D_values
        @test model2.params_names == model.params_names
        @test model2.params[2] ≈ model.params[2]
        @test model2.params[4] ≈ 1

    end

    n = 30
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.Gumbel(params[1], params[2]), n),
                    Symbol("1 h") => rand(IDF.Gumbel(params[3], params[4]), n),
                    Symbol("6 h") => rand(IDF.Gumbel(params[5], params[6]), n))

    

    init_model = IDF.initializeModel(IDF.NoScalingGumbelModel,data)

    @testset "initializeModel()" begin

        @test length(init_model.params) == 2*length(D_values)

        d = D_values[1]
        μ, ϕ = IDF.Extremes.gumbelfitpwm(data, Symbol(IDF.to_french_name(d))).θ̂
        @test init_model.params[1] ≈ μ
        @test init_model.params[2] ≈ exp(ϕ)

    end

    d_ref = maximum(D_values)
    SS_model = IDF.estimSimpleScalingModel(init_model, d_ref=d_ref)

    @testset "estimSimpleScalingModel()" begin

        @test typeof(SS_model) == IDF.SimpleScalingModel
        @test length(SS_model.params) == 4

        θ = IDF.transformParams(init_model)

        # error function
        function get_error(x)
            """x = [logistic(α), C_μ, C_σ]"""

            α, C_μ, C_σ = IDF.logistic_inverse(x[1]), x[2], x[3]

            error = 0.0
            for i in eachindex(D_values)
                d=D_values[i]
                log_μ̂_d = θ[2*i-1]
                log_σ̂_d = θ[2*i]
                error = error + ( log_μ̂_d - (C_μ - α*(log(d)-log(d_ref))) )^2
                error = error + ( log_σ̂_d - (C_σ - α*(log(d)-log(d_ref))) )^2
            end

            return error
        end

        # initialization
        α_init = - GLM.coef(GLM.lm(
                            @formula(y ~ x), 
                            IDF.DataFrame(x = log.(D_values), y = [θ[2*i-1] for i in eachindex(D_values)])
                            )
                        )[2] # initialization of α via linear regression of log(μ_d) depending on log(d)
        α_init = maximum( [ minimum([α_init, 0.99]), 0.01 ] ) # α must be between 0 and 1
        index_d_ref = argmax(D_values.== d_ref)
        x_init = [IDF.logistic(α_init), θ[2*index_d_ref-1], θ[2*index_d_ref]]
        
        optim_result = IDF.transformParams(SS_model)[4], IDF.transformParams(SS_model)[1], IDF.transformParams(SS_model)[2]
        @test get_error(optim_result) < get_error(x_init)

    end

    dGEV_model = IDF.estimdGEVModel(init_model, d_ref=d_ref)

    @testset "estimdGEVModel()" begin
        
        @test typeof(dGEV_model) == IDF.dGEVModel
        @test length(dGEV_model.params) == 5

        θ = IDF.transformParams(init_model)

        # error function
        function get_error(x)
            """x = [logistic(α), log(δ), C_μ, C_σ]"""

            α, δ, C_μ, C_σ = IDF.logistic_inverse(x[1]), exp(x[2]), x[3], x[4]

            error = 0.0
            for i in eachindex(D_values)
                d=D_values[i]
                log_μ̂_d = θ[2*i-1]
                log_σ̂_d = θ[2*i]
                error = error + ( log_μ̂_d - (C_μ - α*(log(d + δ)-log(d_ref + δ))) )^2
                error = error + ( log_σ̂_d - (C_σ - α*(log(d + δ)-log(d_ref + δ))) )^2
            end

            return error
        end

        # initialization
        transformed_params = IDF.transformParams(SS_model)
        x_init = [transformed_params[4], 0.0, transformed_params[1], transformed_params[2]]

        optim_result = IDF.transformParams(dGEV_model)[4], IDF.transformParams(dGEV_model)[5], IDF.transformParams(dGEV_model)[1], IDF.transformParams(dGEV_model)[2]
        @test get_error(optim_result) < get_error(x_init)
        @test get_error(optim_result) < get_error(x_init)
    end

end