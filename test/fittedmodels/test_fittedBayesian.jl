@testset "fitteBayesian.jl" begin 

    D_values = [5,60,360]
    model = IDF.SimpleScalingModel(D_values)
    params = [3,2,0.1,0.7]
    θ̂ = IDF.transformParams(model, params)

    n_chain = 100
    chain_array = Array{Float64, 3}(undef, n_chain, 4, 1)
    for i in 1:n_chain
        for j in 1:4
            chain_array[i,j,1] = rand(Normal(θ̂[j], sqrt(j)))
        end
    end

    chain = IDF.MambaLite.Chains(chain_array)
    println(chain)
    @testset "getChainParam()" begin
    end

    @testset "getChainFunction()" begin
    end

    @testset "returnLevelEstimation()" begin
    end

    @testset "returnLevelCint()" begin
    end

end