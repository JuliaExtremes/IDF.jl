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
            chain_array[i,j,1] = rand(Normal(θ̂[j], sqrt(j)))
        end
    end

    chain = IDF.MambaLite.Chains(chain_array)

    fitted_bayesian = IDF.FittedBayesian(abstract_model, chain)

    @testset "modelEstimation()" begin

        estimated_model = IDF.modelEstimation(fitted_bayesian)
        @test typeof(estimated_model) == IDF.dGEVModel
        @test IDF.transformParams(estimated_model) ≈ [IDF.mean(chain_array[:,j,1]) for j in 1:5]
        @test estimated_model.d_ref == 360
        @test estimated_model.params_names == abstract_model.params_names

    end

    @testset "getChainParam()" begin
    end

    @testset "getChainFunction()" begin
    end

    @testset "returnLevelEstimation()" begin
    end

    @testset "returnLevelCint()" begin
    end

end