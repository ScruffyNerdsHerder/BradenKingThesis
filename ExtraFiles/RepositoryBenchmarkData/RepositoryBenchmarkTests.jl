## File to run a Coarse Omega Constant sweep on a series of benchmarks
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

BLAS.set_num_threads(4)
println("Running with $(Threads.nthreads()) Julia threads.")
println("Each thread uses $(BLAS.get_num_threads()) BLAS threads.")

# @time begin
function max_dist_test_Aniso(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}; tol=SLFA.MACHINE_EPS_FACTOR*eps(eltype(res))) where {T_x<:Real, dim}
    diff = [ getsample(X, i)-getsample(X,i_extrema) for i in I_terminal ]

    max_dist = abs.( only( maximum.( [ diff[:,i] for i in axes(diff, 2)] ) ) )
    min_dist = abs.( only( minimum.( [ diff[:,i] for i in axes(diff, 2)] ) ) )

    max_dist[max_dist .< 1e-14] .= 0.5*minimum(D[A])

    w0 = 2.5 ./ max.(max_dist, min_dist)
    c0 = getsample(X, i_extrema)
    b0 = zero(eltype(res))
    if abs(res[i_extrema]) < tol
        if length(support_set) != sum(support_set)
            b0 = sum(res[map(!,support_set)]) / (length(support_set) - sum(support_set))
        else
            if extremum_type isa Maximum
                b0 = minimum(res)
            else
                b0 = maximum(res)
            end
        end
    end
    a0 = res[i_extrema] - b0
    return [c0; w0; a0; b0]
end
# @load "ExtraFiles/RepositoryBenchmarkData/Abalone_Data.jdl2" X_Abalone y_Abalone A_Abalone D_Abalone
@load "ExtraFiles/RepositoryBenchmarkData/Abalone_Data.jdl2" X_Abalone y_Abalone A_Abalone D_Abalone
N0 = 5
C = 0
N_max = 5
monotonicity = Nonstrict()
rmse_error(res, res_validation, res_history, N) = RMSE(res);
start_gap = 0.00001*maximum(y_Abalone)
solver_Abalone(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_Omega2DSweepExpDecrease((N0,C), theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
error_threshold = [0.0, 0.0, 0.0]
print_iter=false
# Run the a single test on the Abalone benchmark
        FinalTheta_Abalone, resData_Abalone, _, _, _, _, _ = train_RBFN(
        X_Abalone, y_Abalone, A_Abalone, D_Abalone,
        N_max=N_max,
        solver=solver_Abalone,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 8},
        # start_gap = start_gap
        );

@save "RepositoryBenchmarkExpTestSG.jld2" X_Abalone y_Abalone FinalTheta_Abalone resData_Abalone

# end