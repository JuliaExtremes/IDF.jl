using IDF, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, Cairo, Fontconfig

df = CSV.read(joinpath("data","1108446.csv"), DataFrame)[:,2:end]
replace_negative_by_missing(x) = ismissing(x) ? x : ((x < 0) ? missing : x)
for name in names(df)
   df[:,Symbol(name)] = df[:,Symbol(name)] ./ (IDF.to_duration(name)/60.0)
    transform!(df, Symbol(name) => (x -> replace_negative_by_missing.(x)) => Symbol(name))
end

df

d_out=5

test_result = TestGEVGOF(SimpleScalingModel, df, d_out=d_out)
estim_model = test_result.estim_model

quantile(test_result.H0_distrib, 0.95)


# maintenant si je redéfinis le calcul de l'info de Fisher :

I_Fishers = Dict()

D_values_to_consider = IDF.to_duration.(names(df)[names(df) .!= IDF.to_french_name(d_out)])
for d in D_values_to_consider

    data_row = df[:,Symbol(IDF.to_french_name(d))]
    data_row = data_row[.!ismissing.(Vector(data_row))]

    vectorized_data_row = Vector{Float64}(data_row)
            
    function ll(θ)
        model = IDF.setParams(estim_model, θ)
        distrib_d = IDF.getDistribution(model, d) 
        return sum(IDF.Distributions.logpdf.(distrib_d, vectorized_data_row))
    end

    θ̂ = IDF.transformParams(estim_model)
    
    I_Fishers[d] = - IDF.ForwardDiff.hessian(ll, θ̂) / length(vectorized_data_row)

end

I_Fisher_tot = sum(mat for (key,mat) in I_Fishers)

new_H0_distrib = IDF.computeGEVGOFNullDistrib(estim_model, I_Fisher_tot, "cvm", k=100)
quantile(new_H0_distrib, 0.95)

# on est passés de 0.75 à 0.71. Or la statistique était de 0.72 :

test_result.statistic