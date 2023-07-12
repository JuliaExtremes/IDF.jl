function fitMLE(model::IDFModel, data::DataFrame;
                    initialvalues::Union{Vector{<:Real}, Nothing} = nothing,
                    print_evolution::Bool = false)

    # Initial parameters
    if isnothing(initialvalues)
        initialvalues = initializeParams(model, data)
    end

    if print_evolution
        print("initial values (transformed) : ")
        println(getParams(model, initialvalues))
    end

    @assert logLikelihood(model, data, initialvalues) > -Inf "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    # objective function
    fobj(θ) = -logLikelihood(model, data, θ)

    # optimization
    res = Optim.optimize(fobj, initialvalues)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end

    if print_evolution
        print("Final values : ")
        println(getParams(model, θ̂))
        print("Optimal likelihood : ")
        println(-fobj(θ̂))
    end

    #Fisher information 
    I_Fisher = ForwardDiff.hessian(fobj, θ̂)

    return FittedMLE(model, θ̂, I_Fisher)

end