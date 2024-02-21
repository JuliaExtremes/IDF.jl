using IDF, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)[:,2:end]
durations = to_duration.(names(df))
d_ref = 60
SimpleScalingModel(d_ref, [20, 5, .04, .76])

fitted_mle = fitMLE(dGEVModel, df, d_ref = 60)
modelEstimation(fitted_mle)
IDF.cint(fitted_mle, x -> exp(x[1]))
IDF.cint(fitted_mle, x -> exp(x[2]))
IDF.cint(fitted_mle, x -> IDF.logistic_inverse(x[3]) - 0.5)
IDF.cint(fitted_mle, x -> IDF.logistic_inverse(x[4]))
IDF.cint(fitted_mle, x -> exp(x[5])/60)

fitted_mle.I_Fisher

returnLevelCint(fitted_mle, 24*60, 1/0.05, p=  0.9)