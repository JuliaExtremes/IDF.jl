function computeGEVGOFStatistic(model_type::Type{<:IDFModel}, data::DataFrame, d_out::Real;
                                information::String = "observed")
    """Computes the statistic associated to the test procedure that consists by :
        -   Fitting a set of model_type (simple scaling or dGEV) parameters to the data corresponding to every other duration
    and using that lat set of parameters to deduce GEV parameters for d_out
        -   Computing the Cramer von Mises criterion for that GEV distribution and the empirical distribution associated to the sample for d_out
    """

    # IDF model with every other duration
    one_out_data = data[:,names(data) .!= to_french_name(d_out)]
    fitted_IDF_one_out = fitMLE(model_type, one_out_data, d_ref = d_out, information = information)

    estim_distrib_d_out = getDistribution(modelEstimation(fitted_IDF_one_out), d_out)
    empirical_distrib_d_out = dropmissing( select( data, Symbol(IDF.to_french_name(d_out)) ) )[:,Symbol(IDF.to_french_name(d_out))]
    statistic = cvmcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))

    return statistic, modelEstimation(fitted_IDF_one_out), fitted_IDF_one_out.I_Fisher

end

struct TestGEVGOF <: TestIDF

    model_type::Type{<:IDFModel}
    data::DataFrame
    d_out::Real # duration that will be left out for testing
    statistic::Real
    estim_model::IDFModel
    I_Fisher::Matrix{Float64}
    H0_distrib::Union{ContinuousUnivariateDistribution, Nothing}

    function TestGEVGOF(model_type::Type{<:IDFModel}, data::DataFrame;
                            d_out::Union{Real, Nothing} = nothing,
                            information::String = "observed")

        if isnothing(d_out)
            D_values = to_duration.(names(data))
            d_out = minimum(D_values)
        end

        statistic, estim_model, I_Fisher = computeGEVGOFStatistic(model_type, data, d_out, information = information)

        return new(model_type, data, d_out, statistic, estim_model, I_Fisher,  nothing)
    end

end