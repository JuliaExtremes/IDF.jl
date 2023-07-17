function fitMLE(model::IDFModel, data::DataFrame;
                initialvalues::Union{Vector{<:Real}, Nothing} = nothing,
                print_evolution::Bool = false)

    # Initial parameters
    if isnothing(initialvalues)
        initialvalues = initializeParams(model, data)
    end

    if print_evolution
        print("Initial parameter values : ")
        println(getParams(model, initialvalues))
    end

    @assert logLikelihood(model, data, initialvalues) > -Inf "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    # objective function and its derivatives
    fobj(θ) = -logLikelihood(model, data, θ)
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
    if isa(model, NoScalingGumbelModel)
        res = Optim.optimize(fobj, grad_fobj, hessian_fobj, initialvalues)
    else 
        if initialvalues[3] == 0.0
            initialvalues[3] = 0.0001
        end
        res = Optim.optimize(fobj, grad_fobj, hessian_fobj, initialvalues)
    end

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    if print_evolution
        print("Final parameter values : ")
        println(getParams(model, θ̂))
        print("Optimal likelihood : ")
        println(-fobj(θ̂))
    end

    #Fisher information 
    I_Fisher = ForwardDiff.hessian(fobj, θ̂)

    return FittedMLE(model, θ̂, I_Fisher)

end