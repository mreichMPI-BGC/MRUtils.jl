### Utilities for Statistics
using LinearAlgebra, Statistics, StatsBase, DataFrames

export bootstrap, msmedse, loess_with_conf, loess_nd_with_se, extendrange


extendrange(xmin, xmax; frac=0.2) = [xmin, xmax] .+ [-1, 1] .* frac * (xmax - xmin)

"""
    msmedse(x::AbstractVector{<:Real})

Return the standard error of the median using the method by McKean and Schrader (1984). Adapted from RobustStatistics.jl Package.
"""
function msmedse(x::AbstractVector{<:Real})
    n = length(x)
    if n < 2
        error("Need at least 2 observations to compute standard error of the median")
    end

    y = sort(x)

    if length(unique(y)) < n
        @warn "Tied values detected. Estimate of standard error may be inaccurate"
    end

    q995 = quantile(Normal(), 0.995)
    av = round(Int, (n + 1)/2 - q995 * sqrt(n / 4))
    av = max(av, 1)
    top = n - av + 1

    return abs((y[top] - y[av]) / (2 * q995))
end


"""
    bootstrap(statfun, data...; nboot=1000, quantiles=(0.025, 0.975), rng=Random.GLOBAL_RNG)

Perform nonparametric bootstrap using the statistic function `statfun`, applied to one or more input vectors.

Returns a named tuple with:
- `estimate`: the original statistic value
- `bootvals`: all bootstrap replicates
- `se`: standard error of the statistic
- `ci`: confidence interval based on quantiles

You can also call `bootstrap(df, [:x, :y], (x, y) -> cor(x, y))` with a DataFrame interface.
"""
function bootstrap(statfun, data::AbstractVector...;
                   nboot::Int=1000,
                   quantiles::Tuple=(0.025, 0.975),
                   rng::AbstractRNG=Random.GLOBAL_RNG)

    if isempty(data)
        error("At least one input vector is required")
    end

    n = length(data[1])
    if any(length(d) != n for d in data)
        error("All input vectors must have the same length")
    end

    stat_orig = statfun(data...)
    bootvals = Vector{typeof(stat_orig)}(undef, nboot)

    for i in 1:nboot
        idx = rand(rng, 1:n, n)
        sampled = map(d -> d[idx], data)
        bootvals[i] = statfun(sampled...)
    end

    stats = _compute_stats(bootvals, quantiles)
    return (estimate = stat_orig, bootvals = bootvals, se = stats.se, ci = stats.ci)
end

"""
    bootstrap(df::DataFrame, cols::AbstractVector{Symbol}, statfun; kwargs...)

Call bootstrap using columns from a DataFrame by name.
"""
function bootstrap(statfun, df::DataFrame, cols::AbstractVector{Symbol}; kwargs...)
    data = Tuple(df[!, col] for col in cols)
    return bootstrap(statfun, data...; kwargs...)
end

# Internal: compute SE and CI from bootstrap replicates
function _compute_stats(vals, quantiles)
    if eltype(vals) <: Real
        se = std(vals)
        ci = quantile(vals, quantiles)
        return (se = se, ci = ci)

    elseif isa(vals[1], NamedTuple)
        names = keys(vals[1])
        se = NamedTuple{names}(map(n -> std(getfield.(vals, n)), names))
        ci = NamedTuple{names}(map(n -> quantile(getfield.(vals, n), quantiles), names))
        return (se = se, ci = ci)

    elseif isa(vals[1], Tuple)
        len = length(vals[1])
        se = ntuple(i -> std(getindex.(vals, i)), len)
        ci = ntuple(i -> quantile(getindex.(vals, i), quantiles), len)
        return (se = se, ci = ci)

    else
        error("Unsupported statistic return type: $(typeof(vals[1]))")
    end
end


"""
    loess_with_conf(x, y;
                             xgrid = automatic,
                             span = 0.3,
                             se_smooth_span = 0.3)

Fits a 1D LOESS smoother (locally linear) with local variance estimation
and smoothed standard errors for confidence bands.

Returns a NamedTuple with:
- `xgrid`     : prediction grid
- `yfit`     : LOESS fit
- `se`    : smoothed standard error
- `ymin`, `ymax`: 95% confidence band
"""
function loess_with_conf(x, y;
                                   xgrid = range(minimum(x), maximum(x), length=200),
                                   span = 0.3,
                                   se_smooth_span = 0.3)

    n = length(x)
    q = Int(clamp(round(span * n), 2, n))
    d = 1  # degree of local linear regression
    σ²_global = var(y)  # crude estimate of residual variance
    yfit = Float64[]
    se_raw = Float64[]

    for x0 in xgrid
        # Nearest neighbors
        dists = abs.(x .- x0)
        idx = sortperm(dists)[1:q]
        xq, yq = x[idx], y[idx]

        # Tricube weights
        max_dist = maximum(abs.(xq .- x0))
        w = (1 .- (abs.(xq .- x0) ./ max_dist).^3).^3
        W = Diagonal(w)

        # Local weighted regression
        X = hcat(ones(q), xq .- x0)
        β = (X' * W * X) \ (X' * W * yq)
        ŷ = β[1]

        # Local residuals and variance
        residuals = yq .- X * β
        dof = max(sum(w) - (d + 1), 1e-6)
        σ²_local = sum(w .* residuals.^2) / dof
        σ²_local = clamp(σ²_local, 1e-6, 2σ²_global)  # avoid numerical issues

        # Hat matrix diagonal
        H = X * inv(X' * W * X) * X' * W
        hii = H[1, 1]
        se = sqrt(σ²_local * hii)

        push!(yfit, ŷ)
        push!(se_raw, se)
    end

    # Smooth the SE estimates
    se_smooth = loess_no_se(xgrid, se_raw; span = se_smooth_span)

    ymin = yfit .- 1.96 .* se_smooth
    ymax = yfit .+ 1.96 .* se_smooth

    return (; xgrid, yfit, se = se_smooth, ymin, ymax)
end

"""
    loess_no_se(x, y; span)

Helper: performs LOESS smoothing without SE estimation.
Used to smooth the raw SE curve.
"""
function loess_no_se(x, y; span = 0.3)
    n = length(x)
    q = Int(clamp(round(span * n), 2, n))
    yfit = Float64[]

    for x0 in x
        dists = abs.(x .- x0)
        idx = sortperm(dists)[1:q]
        xq, yq = x[idx], y[idx]

        max_dist = maximum(abs.(xq .- x0))
        w = (1 .- (abs.(xq .- x0) ./ max_dist).^3).^3
        W = Diagonal(w)

        X = hcat(ones(q), xq .- x0)
        β = (X' * W * X) \ (X' * W * yq)
        ŷ = β[1]

        push!(yfit, ŷ)
    end

    return yfit
end



"""
    loess_nd_with_se(X, y, Xgrid; span=0.3)

Generalized LOESS for n-dimensional input. Returns predictions, standard errors, and 95% confidence intervals.

Arguments:
- X      : (n x d) matrix of input data
- y      : vector of target values (length n)
- Xgrid  : (m x d) matrix of prediction locations
- span   : fraction of nearest neighbors to use (default 0.3)

Returns:
- DataFrame with columns: :y, :se, :ymin, :ymax
"""
function loess_nd_with_se(X::Matrix{<:Real}, y::Vector{<:Real}, Xgrid::Matrix{<:Real}; span=0.3)
    n, d = size(X)
    m = size(Xgrid, 1)
    q = Int(clamp(round(span * n), d + 1, n))  # at least d+1 points
    σ² = var(y)

    yfit = Float64[]
    se = Float64[]

    for i in 1:m
        x0 = Xgrid[i, :]

        # Compute Euclidean distances and select q nearest neighbors
        dists = mapslices(x -> norm(x .- x0), X; dims=2)[:]
        idx = partialsortperm(dists, 1:q)

        Xq = X[idx, :]
        yq = y[idx]

        # Tricube weights
        max_dist = maximum(dists[idx])
        w = (1 .- (dists[idx] ./ max_dist).^3).^3
        W = Diagonal(w)

        # Centering for stability
        X_centered = Xq .- x0'
        X_design = hcat(ones(q), X_centered)

        # Local weighted linear regression
        β = (X_design' * W * X_design) \ (X_design' * W * yq)
        ŷ = β[1]  # prediction at x0

        # Hat matrix diag approx
        H = X_design * inv(X_design' * W * X_design) * X_design' * W
        hii = H[1, 1]
        se_ŷ = sqrt(σ² * hii)

        push!(yfit, ŷ)
        push!(se, se_ŷ)
    end

    ymin = yfit .- 1.96 .* se
    ymax = yfit .+ 1.96 .* se
    Xgrid = collect(Xgrid)  # ensure xgrid is a vector
    return (; Xgrid, yfit, se, ymin, ymax)

end
