using Pkg
pkg"activate ."

file_path1 = joinpath(@__DIR__, "..","..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
file_path2 = joinpath(@__DIR__, "..", "ExperimentalSolvers.jl")
include(file_path2)
using BenchmarkTools
using Plots
using SLFA
using Optim
using Measures
using JLD2
using PrettyTables
using GLMakie
using CSV
using DataFrames

## Generate omega vs L2 plot as well as w vs L-inf plot

@load "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_Benchmark2DSweepResults.jld2" RangeGrid Results_Store

# First we find the best L2 value
val, idx = findmin(Results_Store[1:(end-1),1:(end-1),end,1])
N0_best1 = RangeGrid[1][idx]
C_best1 = RangeGrid[2][idx]
Ns = collect(0:1:500)
omegas = C_best1.*exp.(-Ns./N0_best1)
bestResHistory1 = Results_Store[idx[1],idx[2],:,1]

val, idx = findmin(Results_Store[1:(end-1),1:(end-1),end,2])
N0_best2 = RangeGrid[1][idx]
C_best2 = RangeGrid[2][idx]
Ns = collect(0:1:500)
omegas = C_best2.*exp.(-Ns./N0_best2)
bestResHistory2 = Results_Store[idx[1],idx[2],:,2]

val, idx = findmin(Results_Store[1:(end-1),1:(end-1),end,3])
N0_best3 = RangeGrid[1][idx]
C_best3 = RangeGrid[2][idx]
Ns = collect(0:1:500)
omegas = C_best3.*exp.(-Ns./N0_best3)
bestResHistory3 = Results_Store[idx[1],idx[2],:,3]

f = GLMakie.Figure()
ax1 = Axis(f[1,1], xlabel = "L2 error", ylabel = "Omega", yscale = log10, xscale = log10, title = "Sine")
GLMakie.scatter!(ax1, bestResHistory1,omegas)
ax2 = Axis(f[1,2], xlabel = "L2 error", ylabel = "Omega", yscale = log10, xscale = log10, title="SineE")
GLMakie.scatter!(ax2, bestResHistory2,omegas)
ax3 = Axis(f[1,3], xlabel = "L2 error", ylabel = "Omega", yscale = log10, xscale = log10, title="Step")
GLMakie.scatter!(ax3, bestResHistory3,omegas)

# ax1 = Axis(f[1,1], xlabel = "L2 error", ylabel = "Omega", title = "Sine")
# GLMakie.scatter!(ax1, bestResHistory1,omegas)
# ax2 = Axis(f[1,2], xlabel = "L2 error", ylabel = "Omega", title="SineE")
# GLMakie.scatter!(ax2, bestResHistory2,omegas)
# ax3 = Axis(f[1,3], xlabel = "L2 error", ylabel = "Omega", title="Step")
# GLMakie.scatter!(ax3, bestResHistory3,omegas)

f

## Calculate L-inf norm for the RBFN as it builds
