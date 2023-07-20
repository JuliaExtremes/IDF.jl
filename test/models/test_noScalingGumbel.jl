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

    end

    @testset "getParams()" begin

        params2 = [3,2,6,4,8,5]
        θ = IDF.transformParams(model, params2)
        params3 = IDF.getParams(model, θ)
        @test params3 ≈ params2
    end

    params_value = [9,6,6,4,3,2]
    n = 30
    data = IDF.DataFrame(Symbol("5 min") => rand(Gumbel(params_value[1], params_value[2]), n),
                    Symbol("1 h") => rand(Gumbel(params_value[3], params_value[4]), n),
                    Symbol("6 h") => rand(Gumbel(params_value[5], params_value[6]), n))

    @testset "logLikelihood()" begin

        θ = IDF.transformParams(model, params_value)
        loglike = IDF.logLikelihood(model, data, θ)

        #Calculus of the log-likelihood
        log_likelihood = 0.0
        for i in eachindex(D_values)
            d = D_values[i]

            μ_d = params_value[2*i-1]
            σ_d = params_value[2*i]

            distrib_extreme_d = Gumbel(μ_d, σ_d)
            log_likelihood = log_likelihood + sum( logpdf.(Ref(distrib_extreme_d), 
                data[:,Symbol(IDF.to_french_name(d))]) )
        end

        @test loglike ≈ log_likelihood

    end

    @testset "initializeParams()" begin
    end

    @testset "estimSimpleScalingRelationship()" begin
    end

    @testset "estimIDFRelationship()" begin
    end

    @testset "returnLevel()" begin
    end

end