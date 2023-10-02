function fitMLE(model_type::Type{<:IDFModel}, data::DataFrame;
                initialmodel::Union{IDFModel, Nothing} = nothing,
                print_evolution::Bool = false,
                d_ref::Union{Real, Nothing} = nothing)

    # Initial parameters
    if isnothing(initialmodel)
        initialmodel = initializeModel(model_type, data, d_ref = d_ref)
    end

    if print_evolution
        print("Initial parameter values : ")
        println(initialmodel.params)
    end

    @assert logpdf(initialmodel, data) > -Inf "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    # objective function and its derivatives
    fobj(θ) = - logpdf(setParams(initialmodel, θ), data)
    function grad_fobj(G, θ)
        grad = ForwardDiff.gradient(fobj, θ)
        for i in eachindex(G)
            G[i] = grad[i]
        end
    end
    function hessian_fobj(H, θ)
        hess = ForwardDiff.hessian(fobj, θ)
        for i in axes(H,1)
            for j in axes(H,2)
                H[i,j] = hess[i,j]
            end
        end
    end

    # optimization
    # first step : for some model must avoid 0.0 as an initial value for ξ because likelihood singularn at ξ=0
    if model_type == NoScalingGEVModel
        for i in eachindex(initialmodel.D_values)
            if initialmodel.params[3*i] == 0.0
                initialmodel.params[3*i] = 0.0001
            end
        end
    elseif model_type != NoScalingGumbelModel
        if initialmodel.params[3] == 0.0
            initialmodel.params[3] = 0.0001
        end
    end
    
    # optimization
    res = nothing
    try 
        res = Optim.optimize(fobj, grad_fobj, hessian_fobj, transformParams(initialmodel))
    catch e
        println("Gradient-descent algorithm could not converge - trying gradient-free optimization")
        res = Optim.optimize(fobj, transformParams(initialmodel))
    end


    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = transformParams(initialmodel)
    end

    if print_evolution
        print("Final parameter values : ")
        println(getParams(model_type, θ̂))
        print("Optimal likelihood : ")
        println(-fobj(θ̂))
    end

    #Fisher information 
    I_Fisher = ForwardDiff.hessian(fobj, θ̂)

    return FittedMLE(initialmodel, θ̂, I_Fisher)

end