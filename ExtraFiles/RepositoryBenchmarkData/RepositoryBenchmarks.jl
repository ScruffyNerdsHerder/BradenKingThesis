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

## Abalone Data importing
abalone_data = CSV.File("ExtraFiles/RepositoryBenchmarkData/abalone.data", header=false)
abalone_Data_Mat = Tables.matrix(abalone_data)
abalone_Age = abalone_Data_Mat[:,end] .+ 1.5
# convert the m/f/i into 0, 1, 0.5
abalone_Data_sex = [value == "M" ? -1 : value for value in abalone_Data_Mat[:,1]]
abalone_Data_sex = [value == "F" ? 1 : value for value in abalone_Data_sex]
abalone_Data_sex = [value == "I" ? 0 : value for value in abalone_Data_sex]

abalone_X_full = [Float64.(abalone_Data_sex) abalone_Data_Mat[:,2:(end-1)]]
abalone_X_raw = Float64.(permutedims(abalone_X_full,(2,1)))
abalone_Y_raw = Float64.(abalone_Age)

X_Abalone = 2 .* standardize(UnitRangeTransform,abalone_X_raw,dims=2).-1
y_Abalone = standardize(UnitRangeTransform,abalone_Y_raw)

A_Abalone, D_Abalone = build_double_cone_graph_bruteforce(X_Abalone, T.(tan.(deg2rad.(30))))

@save "ExtraFiles/RepositoryBenchmarkData/Abalone_Data.jdl2" X_Abalone y_Abalone A_Abalone D_Abalone

## AutoMPG Data Importing
mpg_data = CSV.File("ExtraFiles/RepositoryBenchmarkData/auto-mpg-csv.csv", header=false)
mpg_data_mat = Tables.matrix(mpg_data)
mpg_data_mpg = mpg_data[:,1]


## Run the neighbor finder on the Abalone data
const T = Float64
        function build_double_cone_graph_bruteforce(
        X::AbstractMatrix{T},
        alpha::T;
        sparse_output::Bool = true
        ) where {T<:Real}

        rule = ObstructionRule(R -> doubleCone(R, alpha))

        if sparse_output
                return build_sparseGraph_threaded(X, rule)
        else
                return build_denseGraph_threaded(X, rule)
        end
        end

## Review the Abalone Run
@load "ExtraFiles/RepositoryBenchmarkData/RepositoryBenchmarkExpTest.jld2" X_Abalone y_Abalone FinalTheta_Abalone resData_Abalone

f = GLMakie.Figure()
ax = Axis(f[1,1],
xlabel = "N",
ylabel = "L2 Error",
title = "Abalone Training Data",
titlesize = 20,
xlabelsize = 20,    
ylabelsize = 20,
)

Ns = collect(1:1:length(resData_Abalone))
GLMakie.scatter!(ax, Ns, resData_Abalone)
f