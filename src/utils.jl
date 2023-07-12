# basic util functions

function logistic(x::Real;
                    a::Real = 0, b::Real = 1)
    return log((x-a)/(b-x))
end

function logistic_inverse(y::Real;
                            a::Real = 0, b::Real = 1)
    return (a+b*exp(y))/(1+exp(y))
end

function to_french_name(D::Real)
    if (D<60)
        return string(Int64(D))*" min"
    else   
        return string(Int64(D/60))*" h"
    end
end
