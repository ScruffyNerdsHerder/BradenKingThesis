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
using CairoMakie
using Base.Threads
using Printf
using Statistics

BLAS.set_num_threads(2)
println("Running with $(Threads.nthreads()) Julia threads.")
println("Each thread uses $(BLAS.get_num_threads()) BLAS threads.")
## Function to do the single value plotting
function SweepPlot2D(Results_Store, N0_Range, C_Range, benchmark_name::String, N::Int)
    if benchmark_name == "SineE"
        plot_title = "2D Sweep of C and N0 for SineE Benchmark for N=$N"
        datanum = 2
    elseif benchmark_name == "Sine"
        plot_title = "2D Sweep of C and N0 for Sine Benchmark for N=$N"
        datanum = 1
    else
        plot_title = "2D Sweep of C and N0 for Step Benchmark for N=$N"
        datanum = 3
    end
    # Run for Benchmark
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    minval = minimum(Results_Store[1:(length(N0_Range)),1:(length(C_Range)),N+1,datanum])
    maxval = 1.00001*maximum(Results_Store[1:(length(N0_Range)),1:(length(C_Range)),N+1,datanum])
    tickVals = [minval, 0.1*(minval+maxval), 0.25*(minval+maxval), 0.5*(minval+maxval), maxval]
    tickLabels = [@sprintf("%.2e", x) for x in tickVals]
     
        f = CairoMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N0's",
    ylabel = "C's",
    title = plot_title,
    titlesize = 20,
    xlabelsize = 20,    
    ylabelsize = 20,
    )
    GLMakie.heatmap!(ax, N0_Range, C_Range, Results_Store[1:(length(N0_Range)),1:(length(C_Range)),N+1,datanum], colormap = :jet1, colorscale = scale)
    GLMakie.Colorbar(f[:, 2], limits = (minval, maxval), scale = scale, labelsize = 20, width = 30, colormap = :jet1, ticks = (tickVals, tickLabels))
    return f
end
## Function to do the gif plotting
function SweepPlot2DGIF(Results_Store, N0_Range, C_Range, benchmark_name::String, N_array)
    if benchmark_name == "SineE"
        plot_title = "2D Sweep of C and N0 for SineE Benchmark for N=0"
        datanum = 2
    elseif benchmark_name == "Sine"
        plot_title = "2D Sweep of C and N0 for Sine Benchmark for N=0"
        datanum = 1
    else
        plot_title = "2D Sweep of C and N0 for Step Benchmark for N=0"
        datanum = 3
    end
    # Run for Benchmark
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    f = CairoMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N0's",
    ylabel = "C's",
    title = plot_title,
    titlesize = 20,
    xlabelsize = 20,    
    ylabelsize = 20,
    )
    minval = minimum(Results_Store[1:(length(N0_Range)),1:(length(C_Range)),N_array[end]+1,datanum])
    maxval = maximum(Results_Store[1:(length(N0_Range)),1:(length(C_Range)),N_array[1]+1,datanum])
    colorrange = (minval, maxval)
    tickVals = [minval, 0.00005*(minval+maxval), 0.00025*(minval+maxval), 0.0025*(minval+maxval), 0.05*(minval+maxval), 0.25*(minval+maxval), maxval]
    tickLabels = [@sprintf("%.2e", x) for x in tickVals]
    
    hm = CairoMakie.heatmap!(ax, N0_Range, C_Range,
    Results_Store[1:(length(N0_Range)),1:(length(C_Range)), N_array[end]+1,datanum],
    colormap = :jet1, 
    colorscale = scale,
    colorrange = colorrange
    )
    CairoMakie.Colorbar(f[:, 2], limits = (minval, maxval), scale = scale, labelsize = 20, width = 30, colormap = :jet1, ticks = (tickVals, tickLabels))
    record(f, "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_Benchmark2DSweepResults_"*string(N_array[1]+1)*"_"*string(benchmark_name)*".gif", N_array; framerate = 5) do i
        ax.title = "2D Sweep of C and N0 for "*benchmark_name*" Benchmark for N=$(i)"
        CairoMakie.update!(hm, N0_Range, C_Range, Results_Store[1:(length(N0_Range)),1:(length(C_Range)), i+1, datanum],
        colormap = :jet1, 
        colorscale = scale,
        colorrange = colorrange
        )
    end
end
## Build a grid storing C and N0 values to plot
    N0_Range = collect(10:50:1010)
    C_Range = collect(0.0:0.1:15)

    C_Store = repeat(C_Range, 1, length(N0_Range))
    N0_Store = repeat(N0_Range, 1, length(C_Range))
    N0_Store_Flip = N0_Store'
    RangeGrid = (collect(N0_Store),collect(C_Store'))

    Results_Store = zeros(length(N0_Range),length(C_Range), 501, 3)


## Pull data from a previously run Borah experiment and store results into correct location in 2D Sweep matrix

    @load "ExtraFiles/BorahExperiments/OmegaSweepResults/1D_BenchmarkConstCoarseSweep.jld2" Grid_sweep resData_1 finalTheta_1 
# Evaluate the RMSE for each theta as the network builds, since resData is bounddiff
T_phi = Gaussian{Isotropic,Float64,2}
getError(X_Sine, y_Sine, finalTheta_2_1, errFunc, T_phi)


function getError(X,y,thetas,errFunc,T_phi)
    experiments = size(thetas,1)
    N_max = size(thetas,2)
    result = zeros(experiments,N_max+1)
    result[:,1] .= errFunc(y)
    for i = 1:3
        # iterate over N building
        res = copy(y)
        for N = 1:N_max
            res = res .- thetas[i,N,end-1] .* eval_phi(X,thetas[i,N,:],T_phi) .- thetas[i,N,end]
            result[i,N+1] = errFunc(res) 
        end
    end
    return result
end

function errFunc(y)
    return SLFA.RMSE(y)
end

    for i in eachindex(Grid_sweep)
        println("Storing results for index $i of $(length(Grid_sweep))")
        N0 = Grid_sweep[i][1]
        C = Grid_sweep[i][2]
        N0_index = findfirst(isequal(N0), N0_Range)
        C_index = findfirst(isequal(C), C_Range)
        Results_Store[N0_index, C_index, :, 1] = resData_2_1[i, :]
        Results_Store[N0_index, C_index, :, 2] = resData_2_2[i, :]
        Results_Store[N0_index, C_index, :, 3] = resData_2_4[i, :]
    end

    @save "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_Benchmark2DSweepResults.jld2" RangeGrid Results_Store

## Pull the data from the 2D sweep results and plot the residuals for pair of C and N0 values
@load "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_Benchmark2DSweepResults.jld2" Results_Store
    benchmark_name = "Step"
    N0_Range = collect(10:50:1010)
    C_Range = collect(0.0:0.1:15.0)
    N_array = collect(0:1:500)
    SweepPlot2DGIF(Results_Store, N0_Range, C_Range, benchmark_name, N_array)
    SweepPlot2D(Results_Store, N0_Range, C_Range, "Sine", 10)