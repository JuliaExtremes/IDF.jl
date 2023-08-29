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
    if (d<60)
        return string(Int64(d))*" min"
    else   
        return string(Int64(d/60))*" h"
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

function cvmcriterion(pd::UnivariateDistribution, x::Vector{<:Real})
    """Returns the Crémer - Von Mises criterion associated to the distribution pd fitted to the vector of observations x"""

    x̃ = sort(x)
    n = length(x)

    T = 1/(12*n) + sum( ((2*i-1)/(2*n) - cdf(pd,x̃[i]) )^2 for i=1:n)

    ω² = T

    return ω²

end