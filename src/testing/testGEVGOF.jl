function computeGEVGOFStatistic(model_type::Type{<:IDFModel}, data::DataFrame, d_out::Real;
                                criterion::String = "cvm")
    """Computes the statistic associated to the test procedure that consists by :
        -   Fitting a set of model_type (simple scaling or dGEV) parameters to the data corresponding to every other duration
    and using that lat set of parameters to deduce GEV parameters for d_out
        -   Computing the criterion (Cramer-Von Mises or modified Anderson-Darling) for that GEV distribution and the empirical distribution associated to the sample for d_out
    """

    # IDF model with every other duration
    one_out_data = data[:,names(data) .!= to_french_name(d_out)]
    fitted_IDF_one_out = fitMLE(model_type, one_out_data, d_ref = d_out)

    estim_distrib_d_out = getDistribution(modelEstimation(fitted_IDF_one_out), d_out)
    empirical_distrib_d_out = dropmissing( select( data, Symbol(IDF.to_french_name(d_out)) ) )[:,Symbol(IDF.to_french_name(d_out))]

    if criterion == "cvm"
        statistic = cvmcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))
    else 
        statistic = modifiedADcriterion(estim_distrib_d_out, Array(empirical_distrib_d_out))
    end

    return statistic, modelEstimation(fitted_IDF_one_out), fitted_IDF_one_out.I_Fisher

end


function get_g(model::IDFModel, d_out::Real)

    function g(u)

        θ̂ = IDF.transformParams(model)
        x = quantile(getDistribution(model, d_out), u)

        return gradF_dref(typeof(model), x, θ̂) 
    end

    return g

end


function computeGEVGOFNullDistrib(estim_model::IDFModel, I_Fisher::Matrix{Float64}, criterion::String; 
                                    k=100)
    """I_Fisher must the scaled information matrix, ie. divided by the size of the original dataset.
    """

    g = get_g(estim_model, estim_model.d_ref) # d_out is equal to the reference duration for estim_model

    cov_function_CVM(u,v) = minimum([u,v]) - u*v + Float64(g(u)'/I_Fisher*g(v))
    cov_function_AD(u,v) = sqrt(1/((1-u)*(1-v))) * ( minimum([u,v]) - u*v + Float64(g(u)'/I_Fisher*g(v)) )
    
    if criterion == "cvm"
        λs = approx_eigenvalues(cov_function_CVM, k)
    else 
        λs = approx_eigenvalues(cov_function_AD, k)
    end

    return ZolotarevDistrib(λs)

end


struct TestGEVGOF <: TestIDF

    model_type::Type{<:IDFModel}
    data::DataFrame
    d_out::Real # duration that will be left out for testing
    criterion::String
    statistic::Real
    estim_model::IDFModel
    I_Fisher::Matrix{Float64}
    H0_distrib::Union{ContinuousUnivariateDistribution, Nothing}

    function TestGEVGOF(model_type::Type{<:IDFModel}, data::DataFrame;
                            d_out::Union{Real, Nothing} = nothing,
                            criterion::String = "cvm",
                            k=100)

        if isnothing(d_out)
            D_values = to_duration.(names(data))
            d_out = minimum(D_values)
        end

        statistic, estim_model, I_Fisher = computeGEVGOFStatistic(model_type, data, d_out, criterion = criterion)
        H0_distrib = computeGEVGOFNullDistrib(estim_model, I_Fisher/size(data,1), criterion, k=k)

        return new(model_type, data, d_out, criterion, statistic, estim_model, I_Fisher,  H0_distrib)
    end

end