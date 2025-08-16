### Utilities for DataWrangling and IO in Julia
export rowvec

"""
    rowvec(x::AbstractVector)

Creates a row vector from a 1D array-like input. E.g. for using with dims(2) in AlgebraOfGraphics
"""
rowvec(x::AbstractArray) = reshape(x, 1, :)
