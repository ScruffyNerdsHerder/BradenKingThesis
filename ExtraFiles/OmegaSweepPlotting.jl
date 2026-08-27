using Pkg
pkg"activate ."

file_path1 = joinpath(@__DIR__, "..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
include("ExperimentalSolvers.jl")
using BenchmarkTools
using Plots
using SLFA
using Optim
using Measures
using JLD2
using PrettyTables
using GLMakie
using Base.Threads

@load "C_OmegaSweep_1D.jld2" X_1D C_sweep resData_1 finalTheta_1 

    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "C",
    ylabel = "Final Residual Error",
    title = "Exp Omega Sweep for 1D Benchmarks",
    )
    GLMakie.scatter!(ax,C_sweep, resData_1[:,end])
    f

@load "ExtraFiles/BorahData/C_OmegaSweep_2D.jld2" resData_2_1 finalTheta_2_1 

    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "C",
    ylabel = "Final Residual Error",
    title = "Exp Omega Sweep for 2D Benchmarks",
    )
    f

## Plot and load the results of the dynamic omega ratio sweep for the 2D benchmarks
    @load "ExtraFiles/BorahData/DynamicOmegaRatios_2D.jld2" X ratios omega_TV resData_2_2_1 finalTheta_2_2_1 resData_2_2_2 finalTheta_2_2_2 resData_2_2_3 finalTheta_2_2_3 resData_2_2_4 finalTheta_2_2_4 resData_2_2_5 finalTheta_2_2_5 resData_2_2_6 finalTheta_2_2_6
    N_max = 500
    ## Plot the results of the ratio vs the residual error history
    f = GLMakie.Figure()
    ax = Axis(f[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D Sine Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f

    f2 = GLMakie.Figure()
    ax = Axis(f2[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D SineE Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_2[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_2[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_2[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_2[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_2[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f2[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f2

    f3 = GLMakie.Figure()
    ax = Axis(f3[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D FineSine Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_3[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_3[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_3[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_3[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_3[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f3[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f3

    f4 = GLMakie.Figure()
    ax = Axis(f4[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D Step Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_4[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_4[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_4[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_4[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_4[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f4[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f4

    f5 = GLMakie.Figure()
    ax = Axis(f5[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D Peaks Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_5[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_5[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_5[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_5[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_5[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f5[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f5

    f6 = GLMakie.Figure()
    ax = Axis(f6[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D Spiral Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_6[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_6[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_6[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_6[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_6[end,:], color = :purple, label = "Constant Omega = 2.5")
    Legend(f6[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega = 2.5"], title = "Legend")
    f6

## New thing