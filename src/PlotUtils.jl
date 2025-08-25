#### Utilities for Plotting (in Makie)
using GLMakie, Makie, MarchingCubes, AlgebraOfGraphics # MarchingCubes only for contourmesh!
using MultiKDE
using Distributions: pdf
using Base.Iterators: cycle, product

export wireframe4vec!, kde3d_est, test_gaussian_mix_3d_mass_plot, test_gaussian_3d_mass_plot, figax
export contourmesh!, plot3d!, swapdraw!

_asobs(x) = x isa Observable ? x : Observable(x)

function figax(; size=(1920, 1080), show_axis=true, axis3=true, kwargs...)
    fig = Figure(size=size)
    
    ax = axis3 ?  Axis3(fig[1, 1]; aspect=:equal, viewmode=:fit, kwargs...) :
                  LScene(fig[1, 1], scenekw=(show_axis=show_axis,))
    return fig, ax
end


```
swapdraw! for smoother animation if whole plot has to be redone, e.g.
  i2 = Observable(1)
fig, ax = figax(viewmode=:fitzoom);fig
limits!(ax, (-5,15), (-5, 15), (-5, 15))
#@lift @redraw ax begin
@lift swapdraw!(ax) do
  plot3d!(kdePerMonth[$i2] , levels=[20,50,80], from_grid=true, specular=5., obs_samplesize=0)   
end
for ii in 1:12; i2[]=ii; sleep(0.5); end
```

const _BUF = IdDict{Any, Vector{AbstractPlot}}()  # per-axis state

  function swapdraw!(f::Function, ax)
      # old buffer (may be empty)
      old = get!(_BUF, ax, AbstractPlot[])
      oldLim = ax.limits[]

      # draw new plots invisibly and collect them via diff
      before = Set(copy(ax.scene.plots))
      f()  # your plot3d! call(s)
      after = Set(copy(ax.scene.plots))
      new = collect(setdiff(after, before))
      limits!(ax, oldLim...)
      # hide new, show old for now
      for p in new; p.visible[] = false; end
      for p in old; p.visible[] = true;  end

      # flip on next frame and delete old
      @async begin
          yield()                          # let one frame render
          for p in old; p.visible[] = false; delete!(ax, p); end
          for p in new; p.visible[] = true;  end
      end

      _BUF[ax] = new
      nothing
  end

function contourmesh!(ax, vol::Observable, level::Observable; xrange=[], yrange=[], zrange=[], kwargs...)
    xr = _asobs(xrange); yr = _asobs(yrange); zr = _asobs(zrange)
    mesh_geo = @lift begin
      m = MC($vol; x=$xr |> collect, y=$yr |> collect, z=$zr |> collect)
      march(m, $level)
      MarchingCubes.makemesh(Makie.GeometryBasics, m)
    end
    return mesh!(ax, mesh_geo; kwargs...)
end

# plain arrays/levels just delegate by wrapping
function contourmesh!(ax, vol::AbstractArray, level::Number; kwargs...)
    contourmesh!(ax, _asobs(vol), _asobs(Float(level)); kwargs...)
end


function contourmesh!(ax, vol, levels::AbstractVector = quantile(vol, [0.1, 0.5, 0.9]); 
        alphas=[1, 0.5, 0.2], colors=[:red, :orange, :yellow], kwargs...)
        for (l, a, c) in zip(levels, cycle(alphas), cycle(colors))
                @info("Plotting level $l with alpha $a and color $c")
                contourmesh!(ax, vol, l; color=c, alpha=a, kwargs...)
        end
end

function demo_contourmesh()
  r = range(-1, 1, length=25)
  vol = [norm([1.5x,1.5y,z]) for x=r, y=r, z=r]
  vol = [vol vol vol]
  vol=[vol; vol]
  f = Figure(size=(1920, 1080))
  ax = LScene(f[1, 1])

  contourmesh!(ax, vol, [0.3, 0.5, 0.8]; xrange=[-1., 1], yrange=[-1, 1.], zrange=[-1., 1],
    colors=[:red, :orange, :yellow], alphas=[1, 0.5, 0.2],
    transparency=true, specular=0.8)
  return f
end

struct DensityVol
  obs_points :: AbstractVector{Point3f}
  pdf_vol :: AbstractArray
  xrange :: Tuple{Float64, Float64}
  yrange :: Tuple{Float64, Float64}
  zrange :: Tuple{Float64, Float64}
  levels_grid :: Dict
  levels_obs :: Dict
 end

 DensityVol(; pdf_vol, xrange, yrange, zrange, levels_grid, levels_obs) =
    DensityVol(pdf_vol, xrange, yrange, zrange, levels_grid, levels_obs)


function kde3d_est(x::Vector, y::Vector, z::Vector; bandwidth_rule::Symbol = :silverman, 
    bandwidth=nothing, bandwidth_scaler=1.0, grid_size::Int = 25)
    @assert length(x) == length(y) == length(z)

    # Stack input into 3xN matrix and convert to vector of vectors
    X = hcat(x, y, z)'
    observations = [X[:, i] for i in 1:size(X, 2)]
    # Compute rule-of-thumb bandwidth
    n = size(X, 2)
    d = 3
    stds = std(X; dims=2)[:]
    factor = bandwidth_rule == :silverman ? (4 / (d + 2))^(1 / (d + 4)) : 1.0
    bws = factor .* stds .* n^(-1 / (d + 4))
    @info "$bandwidth_rule bandwidths: $bws"
    bws = bws .* bandwidth_scaler
    @info "Scaled bandwidths: $bws"
    # If bandwidth is provided, use it instead
    if bandwidth !== nothing
        @info "Using provided bandwidth: $bandwidth"
        bws = fill(bandwidth, d)
    end
    # Create KDE
    dims_ = fill(ContinuousDim(), d)
    kde = KDEMulti(dims_, bws, observations)

    # Determine grid bounds from data extrema
    x_min, x_max = extendrange(extrema(x)...)
    y_min, y_max = extendrange(extrema(y)...)
    z_min, z_max = extendrange(extrema(z)...)
    x_range = range(x_min, x_max; length=grid_size)
    y_range = range(y_min, y_max; length=grid_size)
    z_range = range(z_min, z_max; length=grid_size)
    xgrid = Iterators.product(x_range, y_range, z_range) .|> collect

    # Evaluate KDE at sample points
    pdfatobs = [MultiKDE.pdf(kde, obs) for obs in observations]
    # Compute thresholds based on sample KDE values
    threshs=0:5:95
    q = quantile(pdfatobs, reverse(threshs) ./ 100)
    levels_obs = Dict(threshs[i] => q[i] for i in eachindex(threshs))


    # Evaluate KDE on grid
    pdf_vol = [MultiKDE.pdf(kde, pt) for pt in xgrid]

    # Compute thresholds based on gridded KDE values
    flatted = sort(vec(pdf_vol); rev=true)
    cumsum_probs = cumsum(flatted) ./ sum(flatted)
    #levels_grid = [cdf_thresh => findfirst(>=(cdf_thresh), flatted) for cdf_thresh in 0:0.05:1.0]
    levels_grid = Dict(t => flatted[searchsortedfirst(cumsum_probs, t/100)] for t in threshs)

    return DensityVol(Point3f.(x,y,z), pdf_vol, (x_min, x_max), (y_min, y_max), (z_min, z_max),
        levels_grid, levels_obs)
    
end

function plot3d!(dv::Observable{<:DensityVol}; ax=current_axis(), trans_fun::Observable{<:Function}=Observable(log10),
  levels::AbstractVector{<:Observable{<:Real}}=Observable.([10.,50.,90.]), 
  from_grid::Observable{Bool}=Observable(true), 
  obs_samplesize::Observable{Int}=Observable(500),  
  colors = [:red, :orange, :yellow],
  alphas = [0.5, 0.25, 0.125],
  kwargs...)

   
    
    vol_obs = @lift $trans_fun.($dv.pdf_vol)
    pdf_levels = @lift $from_grid ? $dv.levels_grid : $dv.levels_obs
    pdf_levels = [@lift $trans_fun($pdf_levels[$lev]) for lev in levels] 
    # Extract levels from dv
    #rom_grid ? dv.levels_grid : dv.levels_obs
    #pdf_levels = [pdf_levels[l] for l in levels]

    c_plot = contourmesh!(ax, vol_obs, pdf_levels; 
        xrange=@lift($dv.xrange), yrange=@lift($dv.yrange), zrange=@lift($dv.zrange),
        colors=colors, alphas=alphas, specular=0.8, transparency=true,
        kwargs...)


    # Observed points: keep one persistent scatter and update its positions
    pts_obs = Observable(Point3f[])
    if !isempty(dv[].obs_points)
        onany(dv, obs_samplesize) do d, n
            n = max(Int(n), 0)
            pts_obs[] = n == 0 ? Point3f[] :
                        sample(d.obs_points, min(n, length(d.obs_points)))
        end
    end
    sc_plot = scatter!(ax, pts_obs; color=:grey, markersize=5, alpha=0.5)
    
    # if obs_samplesize > 0
    #     @info "Sampling $obs_samplesize observation points for scatter plot"
    #     obs_sample = sample(dv.obs_points, min(obs_samplesize, length(dv.obs_points)))  
    #     scatter!(ax, obs_sample, color=:grey, markersize=5, alpha=0.5)

    # else
    #     @info "No observation points sampled for scatter plot"
    # end
    return (; c_plot, sc_plot)
end

function plot3d!(dv::DensityVol; ax=current_axis(), trans_fun=log10,
                 levels=[10,50,90], from_grid=true, obs_samplesize=0, kwargs...)
    # wrap plain args as Observables and delegate to the reactive core
    return plot3d!(_asobs(dv);
                   ax = ax,
                   trans_fun      = _asobs(trans_fun),
                   levels         = [_asobs(Float64(l)) for l in levels],
                   from_grid      = _asobs(from_grid),
                   obs_samplesize = _asobs(Int(obs_samplesize)),
                   kwargs...)
end

function demo_kde3d_est()
  peng = AlgebraOfGraphics.penguins() |> DataFrame
  fig, ax1 = figax(size=(1920, 1080), axis3=true)
  kde3d_est(peng.bill_length_mm, peng.bill_depth_mm, peng.flipper_length_mm) |> 
    x->plot3d!(x, levels=[10,50,90], from_grid=true)
  ax1.title = "KDE 3D Density Estimation (levels from grid)"
  ax2 = Axis3(fig[1, 2]; aspect=:equal, viewmode=:fit)
  current_axis!(ax2)
  kde3d_est(peng.bill_length_mm, peng.bill_depth_mm, peng.flipper_length_mm) |> 
    x->plot3d!(x, levels=[10,50,90], from_grid=false)
  ax2.title = "KDE 3D Density Estimation (levels from observations)"
  return fig
end


function test_gaussian_mix_3d_mass_plot(; threshold_mass=0.95, grid_size=50, N=1000)

    # Simulate 3D bimodal Gaussian mixture (equal weight)
    d1 = MvNormal([-2.0, 0.0, 0.0], I(3))
    d2 = MvNormal([2.0, 0.0, 0.0], I(3))
    data1 = rand(d1, N ÷ 2)
    data2 = rand(d2, N ÷ 2)
    data = hcat(data1, data2)
    points = [Point3f(data[1, i], data[2, i], data[3, i]) for i in 1:N]

    # Mixture PDF definition
    pdf_mixture(x) = 0.5 * pdf(d1, x) + 0.5 * pdf(d2, x)

    # Compute Euclidean distance from origin (Mahalanobis for centered case only)
    norms = [norm(data[:, i]) for i in 1:N]  # still useful for symmetric thresholds

    # Grid and grid PDF evaluation
    range_ = range(-6, 6; length=grid_size)
    grid = [(x, y, z) for x in range_, y in range_, z in range_]
    pdf_vals = [pdf_mixture([x, y, z]) for (x, y, z) in grid]

    # PDF at data points
    pdf_at_points = [pdf_mixture([data[1, i], data[2, i], data[3, i]]) for i in 1:N]

    # Theoretical threshold is not well-defined analytically for mixture → skip exact radius

    # Quantile-based threshold from points
    threshold_density_points = sort(vec(pdf_at_points), rev=true)[Int(round(threshold_mass * length(pdf_at_points)))]

    # Grid quantile threshold (not meaningful)
    threshold_density_grid = sort(vec(pdf_vals), rev=true)[Int(round(threshold_mass * length(pdf_vals)))]

    # Mass-based grid threshold
    sorted_grid = sort(vec(pdf_vals), rev=true)
    cumsum_probs = cumsum(sorted_grid) ./ sum(sorted_grid)
    threshold_density_mass = sorted_grid[findfirst(>=(threshold_mass), cumsum_probs)]

    # Classify outside regions by various thresholds
    outside_points = pdf_at_points .< threshold_density_points
    outside_mass   = pdf_at_points .< threshold_density_mass

    println("Theoretical density threshold: (N/A for mixture)")
    println("Density threshold based on observation quantile for ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density_points, digits=7))
    println("Density threshold based on grid quantile for ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density_grid, digits=7))
    println("Fraction outside based on Density threshold (obs quantile): ",
            round(100 * count(outside_points) / N, digits=2), "%")
    println("Fraction outside based on Density Mass threshold: ",
            round(100 * count(outside_mass) / N, digits=2), "%")

    # Plot
    fig = Figure(size=(1920, 1080))
    ax = LScene(fig[1, 1], scenekw=(show_axis=true,))
    scatter!(ax, points, color=ifelse.(outside_mass, :red, :blue), markersize=10)

    contour!(ax, -6..6, -6..6, -6..6, pdf_vals .|> log10;
        levels=[threshold_density_mass |> log10], alpha=0.1, colormap=cgrad([:blue, :blue]))

    return fig
end

function test_gaussian_3d_mass_plot(; threshold_mass=0.95, grid_size=50, N=1000)

    # Simulate 3D standard Gaussian points
    d = MvNormal(zeros(3), I(3))
    data = rand(d, N)
    points = [Point3f(data[1, i], data[2, i], data[3, i]) for i in 1:N]

    # Mahalanobis threshold (r such that ||x||^2 ≤ r^2 gives 95% mass)
    r2 = quantile(Chisq(3), threshold_mass)
    radius = sqrt(r2)

    # Compute Euclidean distance from origin
    norms = [norm(data[:, i]) for i in 1:N]
    outside = norms .> radius

    
    # Create 3D grid
    range_ = range(-4, 4; length=grid_size)
    grid = [(x, y, z) for x in range_, y in range_, z in range_]

    # Evaluate PDF on grid
    pdf_vals = [pdf(d, [x, y, z]) for (x, y, z) in grid]

    # PDF at data points
    pdf_at_points = [pdf(d, [data[1, i], data[2, i], data[3, i]]) for i in 1:N]

    # Density at Mahalanobis shell
    threshold_density = pdf(d, [radius, 0.0, 0.0])

    # Quantile-based threshold for density of grid
    threshold_density2 = sort(vec(pdf_vals), rev=true)[Int(round(threshold_mass * length(pdf_vals)))]

    # Quantile-based threshold for density of data points
    threshold_density2_points = sort(vec(pdf_at_points), rev=true)[Int(round(threshold_mass * length(pdf_at_points)))]

    # Mased-based threshold for density
    sorted = sort(vec(pdf_vals), rev=true)
    cumsum_probs = cumsum(sorted)
    cumsum_probs = cumsum_probs ./ sum(sorted)
    # Find the density value that corresponds to the cumulative mass threshold
    threshold_density_mass = sorted[findfirst(>=(threshold_mass), cumsum_probs)]

    # Mased-based threshold for density using data points
    sorted = sort(vec(pdf_at_points), rev=true)
    cumsum_probs = cumsum(sorted)
    cumsum_probs = cumsum_probs ./ sum(sorted)
    # Find the density value that corresponds to the cumulative mass threshold
    threshold_density_mass_from_obs = sorted[findfirst(>=(threshold_mass), cumsum_probs)]
 
    println("Theoretical density threshold for ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density, digits=7))
    println("Density threshold based on observation quantile for ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density2_points, digits=7))       
    println("Density threshold based on numerical Mass in Grid ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density_mass, digits=7))       
    println("Density threshold based on numerical Mass in Obs ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density_mass_from_obs, digits=7))       
    println("Density threshold based on grid quantile for ", round(threshold_mass * 100), "% mass: ",
            round(threshold_density2, digits=7))
    println("Fraction outside based on Density threshold: ",
            round(100 * count(pdf_at_points .< threshold_density) / N, digits=2), "%")
    println("Fraction outside based on Obs quantile threshold: ", 
            round(100 * count(pdf_at_points .< threshold_density2_points) / N, digits=2), "%")
    println("Fraction outside based on Density Mass threshold from Grid: ", 
            round(100 * count(pdf_at_points .< threshold_density_mass) / N, digits=2), "%")
    println("Fraction outside based on Density Mass threshold from Obs: ", 
            round(100 * count(pdf_at_points .< threshold_density_mass_from_obs) / N, digits=2), "%")

    println("Fraction outside based on Radius", round(threshold_mass * 100), "% mass: ",
            round(100 * count(outside) / N, digits=2), "%")

    # Plot
    fig = Figure(size=(1920, 1080))
    ax = LScene(fig[1, 1], scenekw=(show_axis=true,))
    scatter!(ax, points, color=ifelse.(outside, :red, :blue), markersize=10)

     contour!(ax, -4..4, -4..4, -4..4, pdf_vals .|> log10;
         levels=[threshold_density |> log10], alpha=0.1, colormap=cgrad([:blue, :blue]))

    #  contour!(ax, -4..4, -4..4, -4..4, pdf_vals;
    #      levels=[0.9859threshold_density], alpha=1.0, color=:yellow)

    return (;fig, pdf_at_points, pdf_vals, grid, threshold_density2_points, threshold_density_mass)
end



function wireframe4vec!(x::AbstractVector, y::AbstractVector, z::AbstractVector, ax=current_axis();
                                      add_surf=false, kwargs...)
    @assert length(x) == length(y) == length(z) "x, y, z must have the same length"

    # 1. Determine grid shape from unique x/y
    x_unique = sort(unique(x))
    y_unique = sort(unique(y))
    nx, ny = length(x_unique), length(y_unique)

    @assert nx * ny == length(x) "Input data doesn't match a full rectangular grid"

    # 2. Sort data row-wise: increasing y first, then x (same as meshgrid-style ordering)
    sorted_idx = sortperm(collect(zip(y, x)))
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    z_sorted = z[sorted_idx]

    # 3. Reshape into matrices (z[i,j] corresponds to x[i], y[j])
    xg = reshape(x_sorted, nx, ny)
    yg = reshape(y_sorted, nx, ny)
    zg = reshape(z_sorted, nx, ny)

    # 4. Call wireframe
    add_surf && surface!(ax, xg, yg, zg; kwargs...)
    wireframe!(ax, xg, yg, zg; color=:black, linewidth=0.5)

    return nothing
end
