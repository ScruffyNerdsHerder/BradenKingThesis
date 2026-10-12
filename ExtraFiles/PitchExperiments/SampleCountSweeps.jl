## Review the Abalone Run
        using Pkg
pkg"activate ."
file_path1 = joinpath(@__DIR__, "..", "..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
file_path2 = joinpath(@__DIR__, "..", "ExperimentalSolvers.jl")
include(file_path2)
using SLFA
using Optim
using JLD2
using Base.Threads
using GLMakie
using CairoMakie
using Base.Threads
using Printf
using Statistics        

function findTempRMSEsRound(DataFile::String)
        @load DataFile X_temp y_temp FinalTheta_temp resData_temp
        @load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        ylabel = "Training RMSE",
        title = "Pitch Temperature Training Data",
        yscale = log10,
        yticks = LogTicks(-2:1),
        titlesize = 20,
        xlabelsize = 20,    
        ylabelsize = 20,
        )
        # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

        Ns = collect(0:1:(length(resData_temp)-1))
        GLMakie.scatter!(ax, Ns, resData_temp)
        # f
        ## Evaluate the RBFN with the full dataset and the final theta
        T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
        tempRBFN = RBFN(FinalTheta_temp, T2)
        X_flip = Float64.(permutedims(X,(2,1)))
        y_temp_nans = findall(!isnan, y)
        X_temp_full = X_flip[:, y_temp_nans]
        y_temp_full = y[y_temp_nans]
        y_pred = tempRBFN(X_temp_full) 
        y_diff = y_pred - y_temp_full
        y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
        # println("Full Dataset RMSE: ", y_diff_norm)
        # println("Training Data RMSE: ", resData_temp[end])
        # println("Initial RMSE: ", resData_temp[1])
        return f, y_diff_norm, resData_temp[end], resData_temp[1]
end
       @load "ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.1000.jld2" X_temp y_temp FinalTheta_temp resData_temp
        @load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        ylabel = "Training RMSE",
        title = "Pitch Precipitation Training Data",
        yscale = log10,
        yticks = LogTicks(-2:1),
        titlesize = 20,
        xlabelsize = 20,    
        ylabelsize = 20,
        )
        # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

        Ns = collect(0:1:(length(resData_temp)-1))
        GLMakie.scatter!(ax, Ns, resData_temp)
        f
        println("Full Dataset RMSE: ", y_diff_norm)