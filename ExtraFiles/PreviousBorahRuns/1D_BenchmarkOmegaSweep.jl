## File to run a series of omega sweeps on a series of benchmarks
file_path1 = joinpath(@__DIR__, "..", "Benchmarks", "Benchmarks.jl")
# file_path1 = joinpath(@__DIR__, "Benchmarks.jl")
include(file_path1)
include("ExperimentalSolvers.jl")
using SLFA
using Optim
using JLD2
using Base.Threads

## Generate benchmark data with no noise
dx = 0.01
X_1D = [0:dx:10...]
y_AsymRBF = Benchmark_1D(X_1D, "AsymmetricRBF", 0.0)
y_Sine = Benchmark_1D(X_1D, "Sine", 0.0)
y_Step = Benchmark_1D(X_1D, "Step", 0.0)
y_SineE = Benchmark_1D(X_1D, "SineE", 0.0)
y_CompositeSine = Benchmark_1D(X_1D, "CompositeSine", 0.0)

## Run the C omega sweep on each 1D benchmark and save the results
C_sweep = [0:0.1:1...]
N_max = 2
monotonicity = Nonstrict()

# Run the omega sweep on the AsymmetricRBF benchmark
resData_1 = zeros(length(C_sweep),N_max+1)
finalTheta_1 = zeros(length(C_sweep),N_max,4)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
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
end
    
# Run the omega sweep on the Sine benchmark
resData_2 = zeros(length(C_sweep),N_max+1)
finalTheta_2 = zeros(length(C_sweep),N_max,4)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
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
end

# Run the omega sweep on the Step benchmark
resData_3 = zeros(length(C_sweep),N_max+1)
finalTheta_3 = zeros(length(C_sweep),N_max,4)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
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
end

# Run the omega sweep on the SineE benchmark
resData_4 = zeros(length(C_sweep),N_max+1)
finalTheta_4 = zeros(length(C_sweep),N_max,4)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
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
end

# Run the omega sweep on the CompositeSine benchmark
resData_5 = zeros(length(C_sweep),N_max+1)
finalTheta_5 = zeros(length(C_sweep),N_max,4)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
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

@save "C_OmegaSweep_1D.jld2" X_1D C_sweep resData_1 finalTheta_1 resData_2 finalTheta_2 resData_3 finalTheta_3 resData_4 finalTheta_4 resData_5 finalTheta_5
