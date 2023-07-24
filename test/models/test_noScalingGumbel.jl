@testset "noScalingGumbel.jl" begin
    D_values = [5, 60, 360]
    model = IDF.NoScalingGumbelModel(D_values)

    @testset "NoScalingGumbelModel()" begin

        @test model.D_values == D_values
        @test length(model.params_names) == length(D_values)*2
        
        i = rand(1:length(model.params_names))
        @test model.params_names[i][1] == ((i%2 == 1) ? 'μ' : 'σ')
    end

    @testset "transformParams()" begin

        params1 = [3,2,6,4,8,-5]
        @test_throws DomainError IDF.transformParams(model, params1)

        params2 = [3,2,6,4,8,5]
        θ = IDF.transformParams(model, params2)
        @test length(params2) == length(θ)
        @test θ[1] ≈ log(3)
        @test θ[4] ≈ log(4)

    end

    @testset "getParams()" begin

        params2 = [3,2,6,4,8,5]
        θ = IDF.transformParams(model, params2)
        params3 = IDF.getParams(model, θ)
        @test params3 ≈ params2
    end

    params_value = [9,6,6,4,3,2]
    n = 30
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.Gumbel(params_value[1], params_value[2]), n),
                    Symbol("1 h") => rand(IDF.Gumbel(params_value[3], params_value[4]), n),
                    Symbol("6 h") => rand(IDF.Gumbel(params_value[5], params_value[6]), n))

    @testset "logLikelihood()" begin

        θ = IDF.transformParams(model, params_value)
        loglike = IDF.logLikelihood(model, data, θ)

        #Calculus of the log-likelihood
        log_likelihood = 0.0
        for i in eachindex(D_values)
            d = D_values[i]

            μ_d = params_value[2*i-1]
            σ_d = params_value[2*i]

            distrib_extreme_d = IDF.Gumbel(μ_d, σ_d)
            log_likelihood = log_likelihood + sum( IDF.logpdf.(Ref(distrib_extreme_d), 
                data[:,Symbol(IDF.to_french_name(d))]) )
        end

        @test loglike ≈ log_likelihood

    end

    θ_init = IDF.initializeParams(model,data)

    @testset "initializeParams()" begin

        @test length(θ_init) == 2*length(D_values)

        d = D_values[1]
        μ, ϕ = IDF.Extremes.gumbelfitpwm(data, Symbol(IDF.to_french_name(d))).θ̂
        @test θ_init[1] ≈ log(μ)
        @test θ_init[2] ≈ ϕ

    end

    θ = θ_init
    d_ref = maximum(D_values)
    SS_relationship = IDF.estimSimpleScalingRelationship(model, θ, d_ref=d_ref)

    @testset "estimSimpleScalingRelationship()" begin

        @test length(SS_relationship) == 3

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
        index_d_ref = argmax(D_values.== d_ref)
        x_init = [IDF.logistic(α_init), θ[2*index_d_ref-1], θ[2*index_d_ref]]

        @test get_error(SS_relationship) < get_error(x_init)
    end

    @testset "estimIDFRelationship()" begin
        θ = θ_init
        d_ref = maximum(D_values)
        dGEV_relationship = IDF.estimIDFRelationship(model, θ, d_ref=d_ref)

        @test length(dGEV_relationship) == 4

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
        x_init = [SS_relationship[1], 0.0,SS_relationship[2], SS_relationship[3]]

        @test get_error(dGEV_relationship) < get_error(x_init)
    end

    @testset "returnLevel()" begin

        @test_throws AssertionError IDF.returnLevel(model, θ_init, 1.5, 50)

        returnlevel = IDF.returnLevel(model, θ_init, D_values[1], 50)
        params = IDF.getParams(model, θ_init)
        μ_d, σ_d = params[1], params[2]
        distrib_extreme_d = IDF.Gumbel(μ_d, σ_d)
        @test IDF.cdf(distrib_extreme_d, returnlevel) ≈ 1 - 1/50
    end

end