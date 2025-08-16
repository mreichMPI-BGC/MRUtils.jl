### AlgebraOfGraphics Extensions

using GLMakie, Makie, AlgebraOfGraphics
using Polynomials
export polyfit, binned_agg, smooth_se, one2one, idline, binScatter

idline(;  kwargs...) = mapping([0], [1]) * visual(ABLines; linestyle = :dash, color = :black, linewidth = 1.5, label = "1:1", kwargs...)

binScatter(; kwargs...) = begin 
    analysis_keys = (:nbins, :agg, :error)
    # Programmatic keys for visuals
    scatter_keys = Makie.attribute_names(Scatter)
    errorbar_keys = Makie.attribute_names(Errorbars)

    # Split kwargs
    ana_kws   = NamedTuple(filter(kv -> kv[1] in analysis_keys, kwargs))
    sca_kws  = NamedTuple(filter(kv -> kv[1] in scatter_keys, kwargs))
    err_kws = NamedTuple(filter(kv -> kv[1] in errorbar_keys, kwargs))

    binned_agg(; ana_kws...) * (visual(Scatter; strokecolor=:black, strokewidth=2, sca_kws...) + visual(Errorbars; err_kws...))
end

Base.@kwdef struct PolyFitAnalysis
    degree::Int = 2
    npoints::Int = 200
    interval::Bool = true  # confidence band on/off
end

function (pfa::PolyFitAnalysis)(input::ProcessedLayer)
    output = map(input) do (x, y), group
        poly = Polynomials.fit(x, y, pfa.degree)
        x̂ = range(extrema(x)..., length=pfa.npoints)
        ŷ = poly.(x̂)

        if pfa.interval
            # Build polynomial basis matrices
            X = hcat([x .^ i for i in 0:pfa.degree]...)
            X̂ = hcat([x̂ .^ i for i in 0:pfa.degree]...)

            # Compute residuals and variance
            residuals = y .- poly.(x)
            dof = length(x) - (pfa.degree + 1)
            σ² = sum(residuals .^ 2) / dof

            # Standard error of predictions
            XT_X_inv = inv(X'X)
            se² = diag(X̂ * XT_X_inv * X̂') .* σ²
            stderr = sqrt.(se²)

            lower = ŷ .- stderr
            upper = ŷ .+ stderr

            return (x̂, ŷ), (; lower, upper)
        else
            return (x̂, ŷ), (;)
        end
    end

    default_plottype = isempty(output.named) ? Lines : LinesFill
    plottype = Makie.plottype(output.plottype, default_plottype)
    return ProcessedLayer(output; plottype)
end

polyfit(; kwargs...) = AlgebraOfGraphics.transformation(PolyFitAnalysis(; kwargs...))

Base.@kwdef struct SmoothSEAnalysis
    span::Real = 0.3
    npoints::Int = 100
    interval::Bool = true  # confidence band on/off
end

function (sma::SmoothSEAnalysis)(input::ProcessedLayer)
    output = map(input) do (x, y), _
        #result = loess_with_se(x, y; xgrid=range(extrema(x)..., length=sma.npoints), span=sma.span)
        result = loess_with_conf(x, y; xgrid=range(extrema(x)..., length=sma.npoints), span=sma.span)
        if sma.interval
            return (result.xgrid, result.yfit), (; lower=result.ymin, upper=result.ymax)
        else
            return (result.xgrid, result.yfit), (;)
        end
    end

    default_plottype = isempty(output.named) ? Lines : LinesFill
    plottype = Makie.plottype(output.plottype, default_plottype)
    return ProcessedLayer(output; plottype)
end

smooth_se(; kwargs...) = AlgebraOfGraphics.transformation(SmoothSEAnalysis(; kwargs...))


Base.@kwdef struct BinnedAnalysis
    nbins::Int = 10
    agg::Function = mean
    error::Function = x -> std(x) / sqrt(length(x))
end

function (bma::BinnedAnalysis)(input::ProcessedLayer)
    output = map(input) do (x, y), group
        # Compute bin edges
        edges = range(extrema(x)...; length = bma.nbins + 1)
        centers = [(edges[i] + edges[i+1]) / 2 for i in 1:bma.nbins]

        y_means = Float32[]
        y_errors = Float32[]

        for i in 1:bma.nbins
            xlo, xhi = edges[i], edges[i+1]
            idx = findall(x .>= xlo .&& x .< xhi)
            ybin = y[idx]

            if !isempty(ybin)
                push!(y_means, bma.agg(ybin))
                push!(y_errors, bma.error(ybin))
            else
                push!(y_means, NaN)
                push!(y_errors, NaN)
            end
        end

        lower = y_means .- y_errors
        upper = y_means .+ y_errors

        return (centers, y_means, y_errors), (; )
    end
    default_plottype = isempty(output.named) ? Errorbars : Errorbars
    plottype = Makie.plottype(output.plottype, default_plottype)
    return ProcessedLayer(output; plottype)
end

binned_agg(; kwargs...) = AlgebraOfGraphics.transformation(BinnedAnalysis(; kwargs...))

function demo_scatter()
   peng = AlgebraOfGraphics.penguins() |> DataFrame

   data(peng) * mapping(:bill_length_mm, :bill_depth_mm, group=:species, color=:species) *     
      (visual(Scatter) 
      + binScatter(nbins=8, agg=median, error=msmedse, strokewidth=2, whiskerwidth=10)
      # + MRUtils.polyfit(degree=5) 
      + smooth_se(span=0.5, npoints=100) 
       + smooth(span=0.5, degree=1) * visual(linestyle=:dash)
      ) |> draw


end