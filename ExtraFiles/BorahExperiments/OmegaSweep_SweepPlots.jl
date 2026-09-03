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
using CairoMakie
using Base.Threads

f, f1, f2, f3, f4 = PlotOmegaSweepResults("ExtraFiles/BorahExperiments/OmegaSweepResults/1D_BenchmarkLinearIncreaseSweep.jld2")
f2D, f12D, f22D, f32D, f42D, f52D = PlotOmegaSweepResults("ExtraFiles/BorahExperiments/OmegaSweepResults/2D_BenchmarkLinearDecreaseInitSweep.jld2")

function PlotOmegaSweepResults(Sweep::String; Dim=1, SinglePlot=false, SavePlots=true)
    if occursin("1D", Sweep)
        N_max = 250
        # Import the data from the 1D Omega sweep results
        sweepTargetName, sweepType = SweepNaming(Sweep)     
        @load Sweep X_1D resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5
        sweepData = load(Sweep)
        allkeys = collect(JLD2.keys(sweepData))
        sweep_key = findfirst(k -> occursin("_sweep", k), allkeys)
        sweepTarget = sweepData[allkeys[sweep_key]]
        
        # Plot the results of whatever Sweep vs the residual error history
        Ns = collect(0:N_max)
        scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
        f = CairoMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "1D Asymmetric RBF Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_1', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f[:, end+1], limits = (minimum(resData_1), maximum(resData_1)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f1 = CairoMakie.Figure()
        ax = Axis(f1[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "1D Sine Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f1[:, end+1], limits = (minimum(resData_2), maximum(resData_2)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f2 = CairoMakie.Figure()
        ax = Axis(f2[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "1D Step Function Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_3', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f2[:, end+1], limits = (minimum(resData_3), maximum(resData_3)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f3 = CairoMakie.Figure()
        ax = Axis(f3[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "1D SineE Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_4', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f3[:, end+1], limits = (minimum(resData_4), maximum(resData_4)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f4 = CairoMakie.Figure()
        ax = Axis(f4[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "1D Composite Sine Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_5', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f4[:, end+1], limits = (minimum(resData_5), maximum(resData_5)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)
        if SavePlots == true
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots1D/" * sweepType * "_" * sweepTargetName * "_1D_AsymmetricRBF_Benchmark.png", f)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots1D/" * sweepType * "_" * sweepTargetName * "_1D_Sine_Benchmark.png", f1)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots1D/" * sweepType * "_" * sweepTargetName * "_1D_Step_Benchmark.png", f2)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots1D/" * sweepType * "_" * sweepTargetName * "_1D_SineE_Benchmark.png", f3)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots1D/" * sweepType * "_" * sweepTargetName * "_1D_CompositeSine_Benchmark.png", f4)
        end
        return f, f1, f2, f3, f4
    elseif occursin("2D", Sweep)
        N_max = 500
        # Import the data from the 2D Omega sweep results
        sweepTargetName, sweepType = SweepNaming(Sweep)     
        @load Sweep X_2D resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_3 finalTheta_2_3 resData_2_4 finalTheta_2_4 resData_2_5 finalTheta_2_5 resData_2_6 finalTheta_2_6
        sweepData = load(Sweep)
        allkeys = collect(JLD2.keys(sweepData))
        sweep_key = findfirst(k -> occursin("_sweep", k), allkeys)
        sweepTarget = sweepData[allkeys[sweep_key]]
        
        # Plot the results of whatever Sweep vs the residual error history
        Ns = collect(0:N_max)
        scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
        f = CairoMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title =  "2D Sine Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_1', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f[:, end+1], limits = (minimum(resData_2_1), maximum(resData_2_1)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f1 = CairoMakie.Figure()
        ax = Axis(f1[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "2D SineE Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_2', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f1[:, end+1], limits = (minimum(resData_2_2), maximum(resData_2_2)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f2 = CairoMakie.Figure()
        ax = Axis(f2[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "2D Fine Sine Function Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_3', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f2[:, end+1], limits = (minimum(resData_2_3), maximum(resData_2_3)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f3 = CairoMakie.Figure()
        ax = Axis(f3[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "2D Step Function Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_4', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f3[:, end+1], limits = (minimum(resData_2_4), maximum(resData_2_4)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)

        f4 = CairoMakie.Figure()
        ax = Axis(f4[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "2D Peaks Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_5', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f4[:, end+1], limits = (minimum(resData_2_5), maximum(resData_2_5)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)
        
        f5 = CairoMakie.Figure()
        ax = Axis(f5[1,1],
        xlabel = "N",
        xlabelsize = 20,
        ylabelsize = 20,
        ylabel = sweepTargetName * " of Omega(N) Function",
        title = "2D Spiral Benchmark Sweep Behavior"
        )
        CairoMakie.heatmap!(ax,  Ns, sweepTarget, resData_2_6', colormap = :jet1, 
        colorscale = scale)
        CairoMakie.Colorbar(f5[:, end+1], limits = (minimum(resData_2_6), maximum(resData_2_6)), scale = scale, labelsize = 20, width = 30, colormap = :jet1)
        if SavePlots == true
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_Sine_Benchmark.png", f)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_SineE_Benchmark.png", f1)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_FineSine_Benchmark.png", f2)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_Step_Benchmark.png", f3)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_Peaks_Benchmark.png", f4)
            save("ExtraFiles/BorahExperiments/OmegaSweepPlots/" * sweepType * "_" * sweepTargetName * "_2D_Spiral_Benchmark.png", f5)
        end
        return f, f1, f2, f3, f4, f5
    else
        error("Dim not recognized")
    end
end

function SweepNaming(sweep::String)
    if occursin("Init", sweep)
        return "Initial Value Sweep", "Linear Decreasing"
    elseif occursin("Slope", sweep)
        return "Slope Sweep", "Linear Decreasing"
    elseif occursin("Ratio", sweep)
        return "Ratio Sweep", "Dynamic"
    elseif occursin("ConstCoarse", sweep)
        return "Coarse Omega Sweep", "Constant"
    elseif occursin("ConstFine", sweep)
        return "Fine Omega Sweep", "Constant"
    elseif occursin("Dynamic", sweep)
        return "Ratios", "Dynamic"
    elseif occursin("N0", sweep)
        return "N0 Sweep", "Exponential Decreasing"
    elseif occursin("C", sweep)
        return "C Sweep", "Exponential Decreasing"
    elseif occursin("LinearIncrease", sweep)
        return "Slope Sweep", "Linear Increasing"
    else
        error("Sweep type not recognized")
    end
end