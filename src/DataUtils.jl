### Utilities for DataWrangling and IO in Julia
using Random
using DataFrames
using StatsBase

export rowvec
export dfsample, agg_doy2month


"""
    rowvec(x::AbstractVector)

Creates a row vector from a 1D array-like input. E.g. for using with dims(2) in AlgebraOfGraphics
"""
rowvec(x::AbstractArray) = reshape(x, 1, :)




"""
    dfsample(df, args...; frac=nothing, kwargs...)

Sample rows from a DataFrame using `StatsBase.sample` under the hood.

- Pass `n` as a positional argument, e.g. `dfsample(df, 1000)`
- Or use `frac` keyword, e.g. `dfsample(df; frac=0.1)`
- Pass weights positionally before `n` (like in `StatsBase.sample`):
    `dfsample(df, Weights(w), 1000)` or `dfsample(df, Weights(w); frac=0.1)`
- You may also pass a plain vector `w::AbstractVector` directly:
    it will be converted to `Weights(w)` automatically.

All other keywords (`replace`, `ordered`, `rng`, …) are forwarded to `StatsBase.sample`.
"""
function dfsample(df::AbstractDataFrame, args...; frac::Union{Nothing,Real}=nothing, kwargs...)
    pop = 1:nrow(df)

    # Convert first arg to Weights if it's a plain vector, not already AbstractWeights
    newargs = args
    if !isempty(args) && args[1] isa AbstractVector && !(args[1] isa AbstractWeights)
        newargs = (Weights(args[1]), Base.tail(args)...)
    end

    if frac === nothing
        idx = sample(pop, newargs...; kwargs...)
    else
        n = round(Int, frac * length(pop))
        idx = sample(pop, newargs..., n; kwargs...)
    end

    return df[idx, :]
end


function agg_doy2month(df::DataFrame, doycol::Symbol; fun=mean, funcols=Number, groupvars=Symbol[] )
  cols2proc = names(df, funcols) 
  out =  @chain df begin
      transform!(doycol => ByRow(x->month(Date(2020, 1, 1) + Day(x - 1))) => :month)  
      #groupby([:X, :Y, :month])
      groupby([:month; groupvars])
      DataFrames.combine(_, cols2proc .=> fun .=> cols2proc)
     end
     out
end

