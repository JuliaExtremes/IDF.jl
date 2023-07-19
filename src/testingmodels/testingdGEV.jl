function computedGEVStatistic(D_values::Vector{<:Real}, data::DataFrame, d_out::Real)
    """Juste une petite différence : on utilise automatiquement D_out comme D_ref ! Ce qui permet de
    simplifier le calcul de la fonction qui transforme les paramètres. 
    """

    # no scaling model with just d_out
    D_out = [d_out]
    d_out_model = NoScalingGEVModel(D_out)
    fitted_d_out = fitMLE(d_out_model, data)

    # IDF model with every other duration
    D_values_one_out = D_values[D_values .!= d_out]
    IDF_model_one_out = dGEVModel(D_values_one_out, d_ref = d_out)
    fitted_IDF_one_out = fitMLE(IDF_model_one_out, data)

    # matrice de covariance totale
    Σ = I/fitted_d_out.I_Fisher + (I/fitted_IDF_one_out.I_Fisher)[1:3, 1:3]

    # statistique de test
    statistic = norm( sqrt(I/Σ) * (fitted_d_out.θ̂ .- fitted_IDF_one_out.θ̂[1:3]) )^2

    return statistic
end

struct TestdGEV <: TestIDF

    D_values::Vector{<:Real} # en minutes
    data::DataFrame
    d_out::Real # duration that will be left out for testing
    statistic::Real
    H0_distrib::ContinuousUnivariateDistribution

    function TestdGEV(D_values::Vector{<:Real}, data::DataFrame;
                            d_out::Union{Real, Nothing} = nothing)

        if isnothing(d_out)
            d_out = minimum(D_values)
        end

        statistic = computedGEVStatistic(D_values, data, d_out)

        return new(D_values, data, d_out, statistic, Chi(3))
    end

end
