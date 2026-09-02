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


@time begin
function max_dist_test(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Isotropic, T_x, dim}}; tol=SLFA.MACHINE_EPS_FACTOR*eps(eltype(res))) where {T_x<:Real, dim}
    diff = [ getsample(X, i)-getsample(X,i_extrema) for i in I_terminal ] 
    
    max_dist = maximum(norm.(diff))

    if max_dist < 1e-14
        max_dist = 0.5*minimum(D[A])
    end

    w0 = 2.5 ./ max_dist
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

## Generate benchmark data with no noise
dx = 0.01
X_1D = [0:dx:10...]
y_AsymRBF = Benchmark_1D(X_1D, "AsymmetricRBF", 0.0)
y_Sine = Benchmark_1D(X_1D, "Sine", 0.0)
y_Step = Benchmark_1D(X_1D, "Step", 0.0)
y_SineE = Benchmark_1D(X_1D, "SineE", 0.0)
y_CompositeSine = Benchmark_1D(X_1D, "CompositeSine", 0.0)

## Run the C omega sweep on each 1D benchmark and save the results
N0_sweep = [0:2.5:500...]
N_max = 250
monotonicity = Nonstrict()

resData_1 = zeros(length(N0_sweep),N_max+1)
finalTheta_1 = zeros(length(N0_sweep),N_max,4)
resData_2 = zeros(length(N0_sweep),N_max+1)
finalTheta_2 = zeros(length(N0_sweep),N_max,4)
resData_3 = zeros(length(N0_sweep),N_max+1)
finalTheta_3 = zeros(length(N0_sweep),N_max,4)
resData_4 = zeros(length(N0_sweep),N_max+1)
finalTheta_4 = zeros(length(N0_sweep),N_max,4)
resData_5 = zeros(length(N0_sweep),N_max+1)
finalTheta_5 = zeros(length(N0_sweep),N_max,4)

@threads for i in eachindex(N0_sweep)
    # Run the omega sweep on the AsymmetricRBF benchmark
        N0 = N0_sweep[i]
        solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaN0SweepExpDecrease(N0, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
        error_threshold = [0.0, 0.0, 0.0]
        print_iter=false
        Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
        X_1D, y_AsymRBF,
        N_max=N_max,
        solver=solver_OmegaSweep,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 1}
        );
        resData_1[i,:] = res_history_1
        finalTheta_1[i,:,:] = Theta_1
    # Run the omega sweep on the Sine benchmark
        Theta_2, res_history_2, _, _, _, _, _ = train_RBFN(
        X_1D, y_Sine,
        N_max=N_max,
        solver=solver_OmegaSweep,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 1}
        );
        resData_2[i,:] = res_history_2
        finalTheta_2[i,:,:] = Theta_2
    # Run the omega sweep on the Step benchmark
        Theta_3, res_history_3, _, _, _, _, _ = train_RBFN(
        X_1D, y_Step,
        N_max=N_max,
        solver=solver_OmegaSweep,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 1}
        );
        resData_3[i,:] = res_history_3
        finalTheta_3[i,:,:] = Theta_3
    # Run the omega sweep on the SineE benchmark
        Theta_4, res_history_4, _, _, _, _, _ = train_RBFN(
        X_1D, y_SineE,
        N_max=N_max,
        solver=solver_OmegaSweep,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 1}
        );
        resData_4[i,:] = res_history_4
        finalTheta_4[i,:,:] = Theta_4
    # Run the omega sweep on the CompositeSine benchmark
        Theta_5, res_history_5, _, _, _, _, _ = train_RBFN(
        X_1D, y_CompositeSine,
        N_max=N_max,
        solver=solver_OmegaSweep,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 1}
        );
        resData_5[i,:] = res_history_5
        finalTheta_5[i,:,:] = Theta_5
end

@save "1D_BenchmarkExpN0Sweep.jld2" X_1D N0_sweep resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5

end