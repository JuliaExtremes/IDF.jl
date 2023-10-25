@testset "fittedMLE.jl" begin

    abstract_model = IDF.SimpleScalingModel(360, [1,1,0,0.5])
    params = [3,2,0.1,0.7]
    θ̂ = IDF.transformParams(IDF.SimpleScalingModel(360, params))
    I_Fisher = Matrix{Float64}(IDF.I, 4, 4)

    fitted_mle = IDF.FittedMLE(abstract_model, θ̂, I_Fisher)

    estimated_model = IDF.modelEstimation(fitted_mle)

    @testset "ModelEstimation()" begin

        @test typeof(estimated_model) == IDF.SimpleScalingModel
        @test estimated_model.params ≈ params
        @test estimated_model.d_ref == 360
        @test estimated_model.params_names == abstract_model.params_names

    end

    @testset "cint()" begin

        g(θ) = exp(θ[1])
        Δg(θ) = [exp(θ[1]), 0, 0, 0]
        G = Δg(fitted_mle.θ̂)
        var = G'*G

        distrib = IDF.Normal(g(θ̂), sqrt(var))

        cint = IDF.cint(fitted_mle, g)
        @test IDF.cdf(distrib, cint[2]) - IDF.cdf(distrib, cint[1]) ≈ 0.95

        cint = IDF.cint(fitted_mle, g, p=0.8)
        @test IDF.cdf(distrib, cint[2]) - IDF.cdf(distrib, cint[1]) ≈ 0.8

    end

    @testset "returnLevelEstimation()" begin

        d = 5
        T = 50
        @test IDF.returnLevelEstimation(fitted_mle, d, T) ≈ IDF.returnLevel(estimated_model, d, T) 

    end

    @testset "returnLevelCint()" begin

        d = 5
        T = 50
        g(θ) = IDF.returnLevel(IDF.setParams(fitted_mle.abstract_model, θ), d, T)
        @test IDF.returnLevelCint(fitted_mle, d, T, p=0.8) ≈ IDF.cint(fitted_mle, g, p=0.8) 

    end

end