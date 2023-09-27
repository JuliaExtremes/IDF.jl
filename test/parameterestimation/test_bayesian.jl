@testset "bayesian.jl" begin

    n = 60
    d_ref = 360
    params = [3,2,0.1,0.7,4]
    μ, σ, ξ, α, δ = params
    data = IDF.DataFrame(Symbol("5 min") => rand(IDF.GeneralizedExtremeValue(μ * ( (5+δ) / (360+δ) ) ^ (-α), σ * ( (5+δ) / (360+δ) ) ^ (-α), ξ), n),
                    Symbol("1 h") => rand(IDF.GeneralizedExtremeValue(μ * ( (60+δ) / (360+δ) ) ^ (-α), σ * ( (60+δ) / (360+δ) ) ^ (-α), ξ), n),
                    Symbol("6 h") => rand(IDF.GeneralizedExtremeValue(μ * ( (360+δ) / (360+δ) ) ^ (-α), σ * ( (360+δ) / (360+δ) ) ^ (-α), ξ), n))
    dGEV_model = IDF.dGEVModel(d_ref,params)

    @testset "logPosterior()" begin

        @test IDF.logPosterior(dGEV_model, data) ≈ IDF.logpdf(dGEV_model, data)

        prior_distribs = Dict{String,IDF.Distributions.ContinuousUnivariateDistribution}()
        prior_distribs["μ_360"] = IDF.Distributions.Normal()

        @test IDF.logPosterior(dGEV_model, data, prior_distribs=prior_distribs) ≈ IDF.logpdf(dGEV_model, data) + IDF.Distributions.logpdf(Normal(), log(3))

        prior_distribs["δ"] = IDF.Distributions.Uniform()
        @test_throws AssertionError IDF.logPosterior(dGEV_model, data, prior_distribs=prior_distribs)

    end

    @testset "fitBayesian()" begin

        fitted_SS = IDF.fitBayesian(IDF.SimpleScalingModel, data, niter = 500, warmup=200)

        @test typeof(fitted_SS.abstract_model) == IDF.SimpleScalingModel
        @test length(IDF.getChainParam(fitted_SS, "σ_360")) == 500-200

        prior_distribs = Dict{String,IDF.Distributions.ContinuousUnivariateDistribution}()
        prior_distribs["ξ_360"] = IDF.Distributions.Normal(-10,1)

        fitted_SS_prior = IDF.fitBayesian(IDF.SimpleScalingModel, data, prior_distribs=prior_distribs, niter = 500, warmup=200)

        @test IDF.modelEstimation(fitted_SS_prior).params[3] <= IDF.modelEstimation(fitted_SS).params[3]

        
    end

end