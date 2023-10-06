function computeGEVGEVStatistic(model_type::Type{<:IDFModel}, data::DataFrame, d_out::Real)
    """Computes the statistic associated to the test procedure that consists by :
        -   Fitting a set of GEV parameters to the data corresponding to d_out
        -   Fitting a set of model_type (simple scaling or dGEV) parameters to the data corresponding to every other duration
    and using that lat set of parameters to deduce GEV parameters for d_out
        -   Computing the squared difference btw both estimations
    """

    # no scaling model with just d_out
    d_out_data = data[:,names(data) .== to_french_name(d_out)]
    fitted_d_out = fitMLE(NoScalingGEVModel, d_out_data)

    # IDF model with every other duration
    one_out_data = data[:,names(data) .!= to_french_name(d_out)]
    fitted_IDF_one_out = fitMLE(model_type, one_out_data, d_ref = d_out)

    try 
        # matrice de covariance totale
        Σ = I/fitted_d_out.I_Fisher + (I/fitted_IDF_one_out.I_Fisher)[1:3, 1:3]
        # statistique de test
        statistic = norm( sqrt(I/Σ) * (fitted_d_out.θ̂ .- fitted_IDF_one_out.θ̂[1:3]) )^2

        return statistic

    catch err

        println("The Fisher information matrix is singular :")
        println(err)
        println("Returning 0 as the statistic value")

        return 0

    end

end

struct TestGEVGEV <: TestIDF

    model_type::Type{<:IDFModel}
    data::DataFrame
    d_out::Real # duration that will be left out for testing
    statistic::Real
    H0_distrib::ContinuousUnivariateDistribution

    function TestGEVGEV(model_type::Type{<:IDFModel}, data::DataFrame;
                            d_out::Union{Real, Nothing} = nothing)

        if isnothing(d_out)
            D_values = to_duration.(names(data))
            d_out = minimum(D_values)
        end

        statistic = computeGEVGEVStatistic(model_type, data, d_out)

        return new(model_type, data, d_out, statistic, Chi(3))
    end

end