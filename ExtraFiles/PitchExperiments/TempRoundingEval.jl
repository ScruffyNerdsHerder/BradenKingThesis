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
        @load "DataSource/obs_temp_c_Xy.jld2" X y
        ## Evaluate the RBFN with the full dataset and the final theta
        T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
        tempRBFN = RBFN(FinalTheta_temp, T2)
        X_flip = Float64.(permutedims(X,(2,1)))
        y_temp_nans = findall(!isnan, y)
        X_temp_full = X_flip[:, y_temp_nans]
        y_temp_full = y[y_temp_nans]
        y_diff_norm = zeros(length(resData_temp))
        y_diff_norm[1] = sqrt(norm(y_temp_full)^2/length(y_temp_full))
        @threads for i = 1:100
                RBF_now = RBFN(FinalTheta_temp[1:i,:], T2)
                y_pred_now = round.(Int, RBF_now(X_temp_full))
                y_diff_now = y_pred_now - y_temp_full
                y_diff_norm[i+1] = sqrt(norm(y_diff_now)^2/length(y_diff_now))
        end
        return y_diff_norm
end
y_diff_norm = findTempRMSEsRound("Results/PitchTestTemp0.1.jld2");
y_diff_norm2 = findTempRMSEsRound("Results/PitchTestTemp0.2.jld2");
y_diff_norm3 = findTempRMSEsRound("Results/PitchTestTemp0.3.jld2");
y_diff_norm4 = findTempRMSEsRound("Results/PitchTestTemp0.4.jld2");

@save "Results/PitchTestTempRoundedRMSEs.jld2" y_diff_norm y_diff_norm2 y_diff_norm3 y_diff_norm4