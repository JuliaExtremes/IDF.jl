@testset "maxLikelihood.jl" begin

    n = 60
    d_ref = 360
    params = [3,2,0.1,0.7]
    μ, σ, ξ, α = params
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( 5 / 360 ) ^ (-α), σ * ( 5 / 360 ) ^ (-α), ξ), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 60 / 360 ) ^ (-α), σ * ( 60 / 360 ) ^ (-α), ξ), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( 360 / 360 ) ^ (-α), σ * ( 360 / 360 ) ^ (-α), ξ), n))

    @testset "fitMLE()" begin

        @test_throws AssertionError IDF.fitMLE(IDF.NoScalingGEVModel, data,
            initialmodel = IDF.NoScalingGEVModel([5,60,360], [1,1,-0.5, 1,1,-0.5,1,1,-0.5])
        )

        initialmodel = IDF.SimpleScalingModel(d_ref, params)
        result = IDF.fitMLE(IDF.SimpleScalingModel, data, initialmodel = initialmodel)

        @test typeof(result.abstract_model) == IDF.SimpleScalingModel
        @test result.abstract_model.d_ref == 360

        fobj(θ) = - IDF.logpdf(IDF.setParams(initialmodel, θ), data)
        function grad_fobj(G, θ)
            grad = IDF.ForwardDiff.gradient(fobj, θ)
            for i in eachindex(G)
                G[i] = grad[i]
            end
        end
        function hessian_fobj(H, θ)
            hess = IDF.ForwardDiff.hessian(fobj, θ)
            for i in axes(H,1)
                for j in axes(H,2)
                    H[i,j] = hess[i,j]
                end
            end
        end

        res = IDF.Optim.optimize(fobj, grad_fobj, hessian_fobj, IDF.transformParams(initialmodel))
        @test result.θ̂ ≈ IDF.Optim.minimizer(res)
        @test result.I_Fisher ≈ IDF.ForwardDiff.hessian(fobj, result.θ̂)

    end

end