@testset "fittedMLE.jl" begin

    D_values = [5,60,360]
    model = IDF.SimpleScalingModel(D_values)
    params = [3,2,0.1,0.7]
    θ̂ = IDF.transformParams(model, params)
    I_Fisher = Matrix{Float64}(I, 4, 4)

    fitted_mle = IDF.FittedMLE(model, θ̂, I_Fisher)

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
        @test IDF.returnLevelEstimation(fitted_mle, d, T) ≈ IDF.returnLevel(model, θ̂, d, T) 

    end

    @testset "returnLevelCint()" begin

        d = 5
        T = 50
        g(θ) = IDF.returnLevel(model, θ, d, T) 
        @test IDF.returnLevelCint(fitted_mle, d, T, p=0.8) ≈ IDF.cint(fitted_mle, g, p=0.8) 

    end

end