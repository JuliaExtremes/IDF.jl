using IDF, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, Cairo, Fontconfig

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)[:,2:end]
durations = to_duration.(names(df))
d_ref = 60

ss_model = SimpleScalingModel(d_ref, [0, 1, 0, .8])

data = DataFrame(oneh = [2,3], twoh = [0,1])
rename!(data, :oneh => Symbol("1 h"), :twoh => Symbol("2 h"))
IDF.logpdf(ss_model,data)

fitted_mle = fitMLE(SimpleScalingModel, df, d_ref = 60)
modelEstimation(fitted_mle)
IDF.cint(fitted_mle, x -> exp(x[1]))
IDF.cint(fitted_mle, x -> exp(x[2]))
IDF.cint(fitted_mle, x -> IDF.logistic_inverse(x[3]) - 0.5)
IDF.cint(fitted_mle, x -> IDF.logistic_inverse(x[4]))
IDF.cint(fitted_mle, x -> exp(x[5])/60)

fitted_mle.I_Fisher

returnLevelCint(fitted_mle, 24*60, 1/0.05, p=  0.9)




SS_params = [2,0.3,0.1,0.7]
SS_model = SimpleScalingModel(24*60, SS_params)
drawIDFCurves(SS_model)

CS_params = [2,0.3,0.1,0.7, 0.7 + 0.7*2]
CS_model = CompositeScalingModel(24*60, CS_params)
set_default_plot_size(18cm, 12cm)
p = drawIDFCurves(CS_model)
draw(PDF("IDFcurve_composite_scaling_ratio_2.pdf", 18cm, 12cm), p)

