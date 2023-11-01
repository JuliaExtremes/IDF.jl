# basic util functions

function logistic(x::Real;
                    a::Real = 0, b::Real = 1)
    return log((x-a)/(b-x))
end

function logistic_inverse(y::Real;
                            a::Real = 0, b::Real = 1)
    return (a+b*exp(y))/(1+exp(y))
end

function to_french_name(d::Real)
    d = Int64(round(d))
    if (d<60)
        return string(d)*" min"
    else   
        if d%60 == 0
            return string(d÷60)*" h"
        else
            return string(d÷60)*" h" * " " * to_french_name(d%60)
        end
    end
end

function to_duration(d_string::String)
    last_char = d_string[length(d_string):length(d_string)]
    if cmp(last_char, "n") == 0 # the duration is given in minutes
        return Int64(parse(Float64, d_string[1:length(d_string)-4]))
    else   
        return Int64(parse(Float64, d_string[1:length(d_string)-2]) * 60)
    end
end

function get_durations_labels(D_values::Vector{<:Real})
    """Returns a vector of strings that wll be the labels given to the durations on the x-axis of the plotted IDF curve"""

    D_values = sort(D_values)
    D_values_lower_60 = D_values[D_values .< 60]
    D_values_bigger_60 = D_values[D_values .>= 60]

    labels = Vector{String}()
    if length(D_values_lower_60) >= 1
        push!(labels, to_french_name(D_values_lower_60[1]))
        if length(D_values_lower_60) >= 2
            for d in D_values_lower_60[2:end-1]
                push!(labels, string(Int64(round(d))))
            end
            push!(labels, to_french_name(D_values_lower_60[end]))
        end
    end
    if length(D_values_bigger_60) >= 1
        push!(labels, to_french_name(D_values_bigger_60[1]))
        if length(D_values_bigger_60) >= 2
            for d in D_values_bigger_60[2:end-1]
                push!(labels, string(Int64(round(Int64(round(d))/60))))
            end
            push!(labels, to_french_name(D_values_bigger_60[end]))
        end 
    end

    return labels
    
end

function cvmcriterion(pd::UnivariateDistribution, x::Vector{<:Real})
    """Returns the Crémer - Von Mises criterion associated to the distribution pd fitted to the vector of observations x"""

    x̃ = sort(x)
    n = length(x)

    T = 1/(12*n) + sum( ((2*i-1)/(2*n) - cdf(pd,x̃[i]) )^2 for i=1:n)

    ω² = T

    return ω²

end

function modifiedADcriterion(pd::UnivariateDistribution, x::Vector{<:Real})
    """Returns the Crémer - Von Mises criterion associated to the distribution pd fitted to the vector of observations x"""

    x̃ = sort(x)
    n = length(x)

    T = n/2 - 2*sum(cdf(pd,x̃[i]) for i in 1:n) - sum( ( 2-(2*i-1)/n ) * log( 1 - cdf(pd,x̃[i]) ) for i=1:n)

    return T

end

function approx_eigenvalues(ρ::Function, k::Int64)
    """ Approximates the k first eigenvalues of the kernel ρ by computing the associated symmetric matrix K
    """
    
    K = zeros(Float64,k,k)
    
    for i in 1:k
        for j in 1:k
            K[i,j] = (1/k) * ρ( (2*i-1)/(2*k) , (2*j-1)/(2*k) )
        end
    end

    return reverse(real.(eigvals(K))) # taking real parts in case approximations caused complex results
end

#####################
### "Zolotarev" distribution : distribution that approximates well the behaviour of the tail of the GEVGOF statstic, under H0
#####################

struct ZolotarevDistrib <: Distribution{Univariate,Continuous}

    λs::Vector{Float64}

end

minimum(zolo_distrib::ZolotarevDistrib) = 0.0
maximum(zolo_distrib::ZolotarevDistrib) = +Inf
insupport(zolo_distrib::ZolotarevDistrib, x::Real) = (x >= 0)

function cdf(zolo_distrib::ZolotarevDistrib, x::Real)
    """Valid for high quantiles only"""

    λs  = zolo_distrib.λs
    k = length(λs)

    λ₁ = λs[1]
    
    term1 = prod([ (1 - λs[i]/λ₁)^(-0.5) for i in 2:k])
    term2 = 1/gamma(0.5)
    term3 = ( x/(2*λ₁) )^(-0.5)
    term4 = exp( - (x/(2*λ₁)) )
        
    return maximum([1 - term1 * term2 * term3 * term4, 0 ])

end

function quantile(zolo_distrib::ZolotarevDistrib, p::Real)
    """Valid for high quantiles only"""

    to_be_zero(x) = cdf(zolo_distrib, x) - p

    sup_bound = 1
    while to_be_zero(sup_bound) <= 0
        sup_bound += 1
    end
    
    find_zero(to_be_zero, (0, sup_bound))

end