@testset "fittedBayesian.jl" begin 

    d_ref = 360
    params = [3,2,0.1,0.7,3]
    model = IDF.dGEVModel(d_ref, params)
    θ̂ = IDF.transformParams(model)

    abstract_model = IDF.dGEVModel(d_ref, [1,1,0.0,0.5,1])

    n_chain = 100
    chain_array = Array{Float64, 3}(undef, n_chain, 5, 1)
    for i in 1:n_chain
        for j in 1:5
            chain_array[i,j,1] = rand(Normal(θ̂[j], 1/100))
        end
    end

    chain = IDF.MambaLite.Chains(chain_array)

    fitted_bayesian = IDF.FittedBayesian(abstract_model, chain)

    @testset "modelEstimation()" begin

        estimated_model = IDF.modelEstimation(fitted_bayesian)
        @test typeof(estimated_model) == IDF.dGEVModel
        @test estimated_model.params ≈ IDF.mean(model.params for model in IDF.setParams.(
                                                    Ref(fitted_bayesian.abstract_model),
                                                    [chain_array[i,:,1] for i in 1:n_chain]))
        @test estimated_model.d_ref == 360
        @test estimated_model.params_names == abstract_model.params_names

    end

    @testset "getChainParam()" begin

        @test_throws DomainError IDF.getChainParam(fitted_bayesian, "κ")

        chain_α = IDF.getChainParam(fitted_bayesian, "α")
        @test chain_α ≈ IDF.logistic_inverse.(chain_array[:,4,1])

    end

    @testset "getChainFunction()" begin

        g(θ) = θ[1] + IDF.logistic_inverse(θ[4])

        chain_g = IDF.getChainFunction(fitted_bayesian, g)
        @test chain_g ≈ g.([chain_array[i,:,1] for i in 1:n_chain])

    end

    @testset "returnLevelEstimation()" begin

        d = 30
        T = 20
        g_return_level(θ) = IDF.returnLevel(IDF.setParams(abstract_model, θ), d, T)
        
        return_level_estim = IDF.returnLevelEstimation(fitted_bayesian, d, T)
        @test return_level_estim ≈ IDF.Distributions.mean(g_return_level.([chain_array[i,:,1] for i in 1:n_chain]))

    end

    @testset "returnLevelCint()" begin

        d = 30
        T = 20
        g_return_level(θ) = IDF.returnLevel(IDF.setParams(abstract_model, θ), d, T)
        
        cint_estim = IDF.returnLevelCint(fitted_bayesian, d, T, p=0.8)

        chain_return_level = g_return_level.([chain_array[i,:,1] for i in 1:n_chain])
        inside_cint = (chain_return_level .>= cint_estim[1]) .&& (chain_return_level .< cint_estim[2])
        @test sum(inside_cint) ≈ 100*0.8

    end

end