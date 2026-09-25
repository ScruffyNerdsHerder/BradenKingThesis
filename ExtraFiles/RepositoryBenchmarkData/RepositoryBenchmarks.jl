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

        # Normalize each column from -1 to 1
        X_Abalone_full = zero(abalone_X_raw)
        for i = 1:length(abalone_X_raw[:,1])
                X_Abalone_full[i,:]= 2 .* standardize(UnitRangeTransform,abalone_X_raw[i,:]).-1
        end         
        y_Abalone_full = standardize(UnitRangeTransform,abalone_Y_raw)

        # # Do random sampling of the data
        # samples = randperm(size(y_Abalone_full, 1))[1:2000]
        # X_Abalone = X_Abalone_full[:, samples]
        # Y_Abalone = y_Abalone_full[samples]
        
        # First 2000 samples
        X_Abalone = X_Abalone_full[:,1:2000]
        y_Abalone = y_Abalone_full[1:2000]
        # First 2000 samples
        X_Abalone = X_Abalone_full
        y_Abalone = y_Abalone_full
        
        A_Abalone, D_Abalone = build_double_cone_graph_bruteforce(X_Abalone, T.(tan.(deg2rad.(30))))
        @save "ExtraFiles/RepositoryBenchmarkData/Abalone_Data.jdl2" X_Abalone y_Abalone A_Abalone D_Abalone

## AutoMPG Data Importing
mpg_data = CSV.File("ExtraFiles/RepositoryBenchmarkData/auto-mpg-csv.csv", header=false)
mpg_data_mat = Tables.matrix(mpg_data)

y_mpg_data_raw = Float64.(copy(mpg_data_mat[:,1]))
X_mpg_data_raw = Float64.(copy(mpg_data_mat[:,2:end]))
y_mpg_data_full = standardize(UnitRangeTransform,y_mpg_data_raw)
X_mpg_data_full = 2 .* standardize(UnitRangeTransform,X_mpg_data_raw,dims=2).-1


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

## Review the Abalone Run
        @load "ExtraFiles/RepositoryBenchmarkData/RepositoryBenchmarkExpNC.jld2" X_Abalone y_Abalone FinalTheta_Abalone resData_Abalone
        @load "ExtraFiles/RepositoryBenchmarkData/Abalone_Const_Aniso_01_20.jld2" X_Abalone y_Abalone FinalTheta_Abalone resData_Abalone

        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        ylabel = "L2 Error",
        title = "Abalone Training Data",
        yscale = log10,
        yticks = LogTicks(-2:1),
        titlesize = 20,
        xlabelsize = 20,    
        ylabelsize = 20,
        )
        # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

        Ns = collect(0:1:(length(resData_Abalone)-1))
        GLMakie.scatter!(ax, Ns, resData_Abalone)
        f

        Base.print_matrix(IOContext(stdout, :limit => false), FinalTheta_Abalone)