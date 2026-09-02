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

## Plot the Constant Omega 1D Sweep
    @load "ExtraFiles/BorahExperiments/OmegaSweepResults/1D_BenchmarkConstCoarseSweep.jld2" X_1D Omega_sweep resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5
    N_max = 250
    
    ## Plot the results of the ratio vs the residual error history
        Ns = collect(0:N_max)
        ers = resData_3
        scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
        ## Plot the residuals vs omega
        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = "Constant Omega Value",
        )
        GLMakie.heatmap!(ax,  Ns, Omega_sweep, ers', colormap = :jet1, 
        colorscale = scale)
        # colorrange = (1e-3, 1))
        f
    # Find the Omega that gives the best residual for each N
    best_omegas = zeros(5)
    best_omega_ind = argmin(resData_1[:,end], dims = 1)
    best_omega = Omega_sweep[best_omega_ind]
    best_omegas[1] = only(best_omega)
    best_omega_ind = argmin(resData_2[:,end], dims = 1)
    best_omega = Omega_sweep[best_omega_ind]
    best_omegas[2] = only(best_omega)
    best_omega_ind = argmin(resData_3[:,end], dims = 1)
    best_omega = Omega_sweep[best_omega_ind]
    best_omegas[3] = only(best_omega)
    best_omega_ind = argmin(resData_4[:,end], dims = 1)
    best_omega = Omega_sweep[best_omega_ind]
    best_omegas[4] = only(best_omega)
    best_omega_ind = argmin(resData_5[:,end], dims = 1)
    best_omega = Omega_sweep[best_omega_ind]
    best_omegas[5] = only(best_omega)
    best_omegas
## Plot the Constant Omega 2d Sweep

 @load "ExtraFiles/BorahData/OmegaSweepResults/2D_BenchmarkConstCoarseSweep.jld2" X_2D Omega_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_3 finalTheta_2_3 resData_2_4 finalTheta_2_4 resData_2_5 finalTheta_2_5 resData_2_6 finalTheta_2_6 
 N_max = 500
 
 ## Plot the results of the ratio vs the residual error history
    Ns = collect(0:N_max)
    ers = resData_2_1
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    ## Plot the residuals vs omega
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N's",
    xlabelsize = 20,
    ylabelsize = 20,
    ylabel = "Initial Value of Omega",
    )
    GLMakie.heatmap!(ax,  Ns, Omega_sweep, ers', colormap = :jet1, 
    colorscale = scale,
     colorrange = (1e-4, 1))
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

## Plot and load the results of the N0 ratio sweep for the 1D benchmarks
 @load "ExtraFiles/PreviousBorahRuns/N0_OmegaSweep_1D.jld2" X_1D N0_Sweep resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5
 N_max = 250
 
 ## Plot the results of the ratio vs the residual error history
    Ns = collect(0:N_max)
    ers = resData_5
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    ## Plot the residuals vs omega
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N's",
    xlabelsize = 20,
    ylabelsize = 20,
    ylabel = "N0 in Exp",
    )
    GLMakie.heatmap!(ax,  Ns, N0_Sweep, ers', colormap = :jet1, 
    colorscale = scale)
    #  colorrange = (1e-5, 1))
     f

## Plot the Initial Value For linear decreas omega Sweep slope - 1/500 cutoff Omega >.001

 @load "ExtraFiles/BorahData/OmegaInitialSweep_1D.jld2" X_1D Omega_sweep resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5
 N_max = 250
 
 ## Plot the results of the ratio vs the residual error history
    Ns = collect(0:N_max)
    ers = resData_4
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    ## Plot the residuals vs omega
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N's",
    xlabelsize = 20,
    ylabelsize = 20,
    ylabel = "Initial Value of Omega",
    )
    GLMakie.heatmap!(ax,  Ns, Omega_sweep, ers', colormap = :jet1, 
    colorscale = scale,
     colorrange = (1e-4, 1))
     f

