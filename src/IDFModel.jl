abstract type IDFModel <: ContinuousMultivariateDistribution end

include(joinpath("models", "noScalingGumbel.jl"))
include(joinpath("models", "noScaling.jl"))
include(joinpath("models", "simpleScaling.jl"))
include(joinpath("models", "dGEV.jl"))
include(joinpath("models", "compositeScaling.jl"))


function getDistribution(model::IDFModel, D_values::Vector{<:Real}) 
    """Returns the distribution of the maximal intensities for all durations in D_values, according to the specific model.
    For now independence btw durations is supposed true. Later we might add an argument 'copula'"""

    marginals = [getDistribution(model, d) for d in D_values]

    return Product(marginals)

end


function sample(model::IDFModel, D_values::Vector{<:Real}, n::Int64=1)
    """Returns a dataframe corresponding to a random sample of size n, of vectors of maximal intensities for all durations in D_values, 
    according to the specific model.
    """
    
    distrib = getDistribution(model, D_values)
    y = Distributions.rand(distrib, n)
    
    return DataFrame([to_french_name(D_values[i]) => y[i,:] for i in eachindex(D_values)])
    
end


function logpdf(model::IDFModel, data::DataFrame)
    """Returns the logpdf (or log-likelihood) of the model evaluated for the given dataframe"""

    value_logpdf = 0.0
    for data_row in eachrow(data)
        
        data_row = data_row[.!ismissing.(Vector(data_row))]
        vectorized_data_row = Vector{Float64}(data_row)

        if size(vectorized_data_row,1) >=1 # if at least one column is not missing for that year
            D_values = to_duration.(names(data_row))
            multiv_distrib = getDistribution(model, D_values) 
            
            value_logpdf += Distributions.logpdf(multiv_distrib, vectorized_data_row)
        end
        
    end
    
    return value_logpdf

end


function returnLevel(model::IDFModel, d::Real, T::Real)
    """Returns the return level associated to rain duration d and a return period T,
    according to the given model
    """

    @assert 1 < T "The return period must be bigger than 1 year."

    return quantile(getDistribution(model, d), 1 - 1/T)

end

function drawIDFCurves(model::Union{SimpleScalingModel, dGEVModel, CompositeScalingModel}; 
                        T_values::Vector{<:Real}=[2,5,10,25,50,100],
                        D_values::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12 ,24]*60,
                        location_name::String = "Imaginary location",
                        put_title::Bool = false,
                        to_be_returned::String = "plot",
                        noscaling_model_for_crosses::Union{NoScalingGumbelModel, NoScalingGEVModel, Nothing} = nothing,
                        y_ticks::Union{Vector{<:Real}, Nothing} = nothing)

    d_min = minimum(D_values)
    d_max = maximum(D_values)
    d_step = d_min/10

    T_values = sort(T_values)
    D_values = sort(D_values)

    data_return_levels = crossjoin(DataFrame(T = T_values),  DataFrame(d = d_min:d_step:d_max))
    transform!(data_return_levels, [:T, :d] => ((x,y) -> returnLevel.(Ref(model), y, x)) => :return_level)
    
    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        push!(layers, layer(data, x = :d, y = :return_level, color = :T, Geom.line()))
    end
    
    labels = get_durations_labels(D_values)
    f_label(x) = labels[D_values .≈ exp(x)][1]
    palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

    if !isnothing(noscaling_model_for_crosses)
        append!(layers,drawIDFCurves(noscaling_model_for_crosses, T_values = T_values, show_estimated_curves = false, to_be_returned = "layers"))  
    end

    if isnothing(y_ticks)
        y_ticks = range(log(minimum(data_return_levels[:,:return_level])), log(maximum(data_return_levels[:,:return_level])), 6)
    end

    if to_be_returned == "plot"

        if put_title
            p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
                    Scale.color_discrete_manual(palette...),
                    Guide.title("IDF curves at " * location_name),
                    Guide.xticks(ticks = log.(D_values)),
                    Guide.yticks(ticks = range(log(minimum(data_return_levels[:,:return_level])), log(maximum(data_return_levels[:,:return_level])), 6)),
                    Guide.xlabel("Rainfall duration"),
                    Guide.ylabel("Rainfall intensity (mm/h)"),
                    Guide.colorkey(title="Return period T (years)"),
                    Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                        key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
            )
            return p
        else
            p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
                    Scale.color_discrete_manual(palette...),
                    Guide.xticks(ticks = log.(D_values)),
                    Guide.yticks(ticks = y_ticks),
                    Guide.xlabel("Rainfall duration"),
                    Guide.ylabel("Rainfall intensity (mm/h)"),
                    Guide.colorkey(title="Return period T (years)"),
                    Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                        key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
            )
            return p
        end
    else
        return layers
    end

end


function drawIDFCurves(model::Union{NoScalingGumbelModel, NoScalingGEVModel}; 
                        T_values::Vector{<:Real}=[2,5,10,25,50,100],
                        location_name::String = "Imaginary location",
                        show_estimated_curves::Bool = true,
                        estim_method::String = "simple scaling",
                        put_title::Bool = false,
                        to_be_returned::String = "plot",
                        y_ticks::Union{Vector{<:Real}, Nothing} = nothing)

    D_values = model.D_values
    T_values = sort(T_values)

    data_return_levels = crossjoin(DataFrame(T = T_values),  DataFrame(d = D_values))
    transform!(data_return_levels, [:T, :d] => ((x,y) -> returnLevel.(Ref(model), y, x)) => :return_level)

    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        push!(layers, layer(data, x = :d, y = :return_level, color = :T, shape=[Shape.xcross], Geom.point()))
    end

    if estim_method == "simple scaling"
        scaling_model = estimSimpleScalingModel(model)
    else 
        scaling_model = estimdGEVModel(model)
    end

    if show_estimated_curves
        append!(layers,drawIDFCurves(scaling_model, T_values = T_values, D_values = D_values, location_name = location_name, to_be_returned = "layers"))
    end

    if to_be_returned == "plot"

        labels = get_durations_labels(D_values)
        f_label(x) = labels[D_values .≈ exp(x)][1]
        palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

        if isnothing(y_ticks)
            y_ticks = range(log(minimum(data_return_levels[:,:return_level])), log(maximum(data_return_levels[:,:return_level])), 6)
        end

        if put_title 
            p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
            Scale.color_discrete_manual(palette...),
            Guide.title("IDF curves at " * location_name),
            Guide.xticks(ticks = log.(D_values)),
            Guide.yticks(ticks = range(log(minimum(data_return_levels[:,:return_level])), log(maximum(data_return_levels[:,:return_level])), 6)),
            Guide.xlabel("Rainfall duration"),
            Guide.ylabel("Rainfall intensity (mm/h)"),
            Guide.colorkey(title="Return period T (years)"),
            Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
            )
        else
            p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
            Scale.color_discrete_manual(palette...),
            Guide.xticks(ticks = log.(D_values)),
            Guide.yticks(ticks = y_ticks),
            Guide.xlabel("Rainfall duration"),
            Guide.ylabel("Rainfall intensity (mm/h)"),
            Guide.colorkey(title="Return period T (years)"),
            Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
            )
        end

        return p

    else

        return layers
    
    end

end