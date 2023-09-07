struct FittedBayesian{T} <: Fitted{T}
    abstract_model::T
    sim::MambaLite.Chains
end

function modelEstimation(fitted_bayesian::FittedBayesian)
    """Returns a pointwise estimation of the model, based on pointwise estimations (by mean)
    of every parameter
    """

    chain = fitted_bayesian.sim
    x = chain.value[:,:,1]
    x = [x[i,:] for i in axes(x,1)]

    models_chain = setParams.(Ref(fitted_bayesian.abstract_model), x)

    estim_params = mean([model.params for model in models_chain])

    return setParams(fitted_bayesian.abstract_model, estim_params,
                        is_transformed=false)

end

function getChainParam(fitted_bayesian::FittedBayesian, name_param::String)
    """Returns the chain associated to the parameter of the model whose name is name_param
    Throws an error if there is no such name.
    """

    names = fitted_bayesian.abstract_model.params_names

    chain = fitted_bayesian.sim
    index_this_param = (names .== name_param)

    if sum(index_this_param)== 0
        throw(DomainError(name_param, "Not a valid parameter name. Valid parameter names are : \n "*join(names, ", ")))
    else 
        chain_transformed = vec(chain.value[:,:,1][:,index_this_param]) # this chain is for the transformed parameter
        fun_param(transformed_param) = getParams(typeof(fitted_bayesian.abstract_model), 
                                                    fill(transformed_param, 
                                                    length(names))
                                                )[index_this_param]
        chain_param = fun_param.(chain_transformed)
        return vec(reduce(hcat,chain_param))
    end

end

function getChainFunction(fitted_bayesian::FittedBayesian, g::Function)
    """Returns the chain associated to the values g(θ_i),
    with θ the vector of (transformed) parameters and i the index in the chain
    """

    chain = fitted_bayesian.sim
    x = chain.value[:,:,1]
    x = [x[i,:] for i in axes(x,1)]

    return g.(x)

end

function returnLevelEstimation(fitted_bayesian::FittedBayesian, d::Real, T::Real)
    """Renvoie une estimation ponctuelle (par la moyenne) du niveau de retour associé à une durée d'accumulation d et à un temps de retour T"""

    g_return_level(θ) = returnLevel(setParams(fitted_bayesian.abstract_model, θ), d, T)
    chain_return_level = getChainFunction(fitted_bayesian, g_return_level)

    return mean(chain_return_level)
end

function returnLevelCint(fitted_bayesian::FittedBayesian, d::Real, T::Real;
                            p::Real = 0.95)
    """Renvoie un intervalle de crédibilité au seuil 1-p 
    pour le niveau de retour associé à une durée d'accumulation d et à un temps de retour T.
    """

    g_return_level(θ) = returnLevel(setParams(fitted_bayesian.abstract_model, θ), d, T)
    chain_return_level = getChainFunction(fitted_bayesian, g_return_level)

    return MambaLite.hpd(chain_return_level, alpha=1-p)
end