using Pkg
pkg"activate ."

# include(file_path1)
include(joinpath(@__DIR__, "..", "ExperimentalSolvers.jl"))
include(joinpath(@__DIR__, "NeighborFinder", "src", "ObstructionGraphs.jl"))
using .ObstructionGraphs
using SparseArrays
using JLD2
using Random
using SLFA
using Optim
using Base.Threads
using CSV
using StatsBase
using Tables

## Run the neighbor finder on the Abalone data
const T = Float64
       function build_double_cone_graph_bruteforce(
       X::AbstractMatrix{T},
       alpha::T;
       sparse_output::Bool = true
       ) where {T<:Real}

       rule = ObstructionGraphs.ObstructionRule(R -> doubleCone(R, alpha))

       if sparse_output
               return ObstructionGraphs.build_sparseGraph_threaded(X, rule)
       else
               return ObstructionGraphs.build_denseGraph_threaded(X, rule)
       end
       end
## Generate the A,D matrices
   @load "obs_dewpoint_c_Xy.jld2"
    samples = randperm(size(y, 1))[1:10000]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_dewpoint_raw = X_flip[:,samples]
    y_dewpoint_raw = y[samples]
    y_dew_nans = findall(!isnan, y_dewpoint_raw)
    X_dewpoint = X_dewpoint_raw[:, y_dew_nans]
    y_dewpoint = y_dewpoint_raw[y_dew_nans]
    A_dewpoint, D_dewpoint = build_double_cone_graph_bruteforce(X_dewpoint, T.(tan.(deg2rad.(30))))
    @load "obs_precip_mm_Xy.jld2"
    samples = randperm(size(y, 1))[1:10000]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_precip_raw = X_flip[:,samples]
    y_precip_raw = y[samples]
    y_precip_nans = findall(!isnan, y_precip_raw)
    X_precip = X_precip_raw[:, y_precip_nans]
    y_precip = y_precip_raw[y_precip_nans]
    A_precip, D_precip = build_double_cone_graph_bruteforce(X_precip, T.(tan.(deg2rad.(30))))
    @load "obs_station_pressure_pa_Xy.jld2"
    samples = randperm(size(y, 1))[1:10000]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_pressure_raw = X_flip[:,samples]
    y_pressure_raw = y[samples]
    y_pressure_nans = findall(!isnan, y_pressure_raw)
    X_pressure = X_pressure_raw[:, y_pressure_nans]
    y_pressure = y_pressure_raw[y_pressure_nans]
    A_pressure, D_pressure = build_double_cone_graph_bruteforce(X_pressure, T.(tan.(deg2rad.(30))))
    @load "obs_temp_c_Xy.jld2"
    samples = randperm(size(y, 1))[1:10000]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_temp_raw = X_flip[:,samples]
    y_temp_raw = y[samples]
    y_temp_nans = findall(!isnan, y_temp_raw)
    X_temp = X_temp_raw[:, y_temp_nans]
    y_temp = y_temp_raw[y_temp_nans]
    A_temp, D_temp = build_double_cone_graph_bruteforce(X_temp, T.(tan.(deg2rad.(30))))
    @load "obs_rh_percent_Xy.jld2"
    samples = randperm(size(y, 1))[1:10000]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_rh_raw = X_flip[:,samples]
    y_rh_raw = y[samples]
    y_rh_nans = findall(!isnan, y_rh_raw)
    X_rh = X_rh_raw[:, y_rh_nans]
    y_rh = y_rh_raw[y_rh_nans]
    A_rh, D_rh = build_double_cone_graph_bruteforce(X_rh, T.(tan.(deg2rad.(30))))

    @save "obs_dewpoint_c_graphs.jld2" A_dewpoint D_dewpoint X_dewpoint y_dewpoint
    @save "obs_precip_mm_graphs.jld2" A_precip D_precip X_precip y_precip
    @save "obs_station_pressure_pa_graphs.jld2" A_pressure D_pressure X_pressure y_pressure
    @save "obs_temp_c_graphs.jld2" A_temp D_temp X_temp y_temp    
    @save "obs_rh_percent_graphs.jld2" A_rh D_rh X_rh y_rh

