abstract type TestIDF end

include(joinpath("testingmodels", "testingsimplescaling.jl"))
include(joinpath("testingmodels", "testingdGEV.jl"))

function statistic(test_object::TestIDF)
    return test_object.statistic
end

function rejectH0(test_object::TestIDF, α::Real)
    @assert 0 < α < 1 "The level of significance α should be between 0 and 1, and its given value was " * string(α)
    return test_object.statistic > quantile(test_object.H0_distrib, 1-α)
end

function pvalue(test_object::TestIDF)
    return 1 - cdf(test_object.H0_distrib, test_object.statistic)
end