import NCDatasets
using Rasters, DataFrames, TidierData
using Dates

export drop_singletons, load_netcdf_dir

drop_singletons(A) = dropdims(A, dims = Tuple(d for (d, s) in enumerate(size(A)) if s==1))

"""
    load_netcdf_dict(dir::AbstractString, pattern::Regex) -> Dict{String,NamedTuple}

Scan `dir` for files whose *basename* matches `pattern`. For each matching NetCDF,
open it and return a Dict mapping the file's basename to a NamedTuple of:

- all variables as fields (Symbol(varname) => Array)
- `dims`  => NamedTuple of dimension lengths
- `attrs` => Dict of global attributes

Requires: NCDatasets.jl
"""
function load_netcdf_dir(dir::AbstractString, pattern::Regex)
    files = filter(f -> isfile(f) && occursin(pattern, basename(f)),
                   readdir(dir; join=true))
    out = Dict{String, Any}()

    for f in files
        try
          d = RasterStack(f)  
          out[basename(f)] = d
        finally
        end
    end

    return out
end



function myMSC_data(filter_regex=r"^(?!.*Basil).*msc.nc"; doy2month=false)
  d=load_netcdf_dict("D:/_FLUXCOM/0d50_monthly/", filter_regex)
  for (k,v) in d d[k] = v[:variable] end
  # or: foreach(d) do (fname, rast) d[fname] = rast[:variable] end
  good_keys = collect(keys(d)) #|> filter(k -> occursin(filter_regex, k))
  r=[d[k] for k in good_keys];
  rstack = RasterStack(r..., name=chop.(good_keys, tail=length(".msc.nc")))
  df = rstack |> DataFrame
  select!(df, names(df)[.![all(ismissing, df[!, c]) for c in names(df)]])
  df |> dropmissing! |> disallowmissing!
  if doy2month
   
  end
  return df
end