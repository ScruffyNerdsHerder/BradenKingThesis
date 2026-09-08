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

## Build a grid storing C and N0 values to plot
N0_Range = collect(10:50:1010)
C_Range = collect(0.0:0.1:10.0)

C_Store = repeat(C_Range, 1, length(N0_Range))
N0_Store = repeat(N0_Range, 1, length(C_Range))
N0_Store_Flip = N0_Store'
RangeGrid = (C_Store, collect(N0_Store'))

Results_Store = zeros(length(N0_Range),length(C_Range), 501, 3)

## Pull data from a previously run Borah experiment and store results into correct location in 2D Sweep matrix
@load "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_BenchmarkExp2DSweep10_1.jld2" Grid_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_4 finalTheta_2_4

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

@load "ExtraFiles/BorahExperiments/OmegaSweepResults/2D_Benchmark2DSweepResults.jld2" RangeGrid Results_Store

    # Ns = collect(0:N_max)
    ers = resData_2_1
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    ## Plot the residuals vs omega
    f = CairoMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N0's",
    ylabel = "C's",
    xlabelsize = 20,
    ylabelsize = 20,
    )
    CairoMakie.heatmap!(ax, N0_Range[1:(end-1)], C_Range, Results_Store[1:(end-1),:,end,3], colormap = :jet1, colorscale = scale)
    f