## File to run a series of omega sweeps on a series of benchmarks
file_path1 = joinpath(@__DIR__, "..", "Benchmarks", "Benchmarks.jl")
# file_path1 = joinpath(@__DIR__, "Benchmarks.jl")
include(file_path1)
include("ExperimentalSolvers.jl")
using SLFA
using Optim
using JLD2
using Base.Threads

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
# X_sine, y_sine, A_sine, D_sine = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
# X_sineE, y_sineE, A_sineE, D_sineE = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
# X_FineSine, y_FineSine, A_FineSine, D_FineSine = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", "FineSine", 0.0)
# X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", "Step", 0.0)
# X_peaks, y_peaks, A_peaks, D_peaks = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", "Peaks", 0.0)
# X_spiral, y_spiral, A_spiral, D_spiral = benchmark_2D_data("yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)

X_sine, y_sine, A_sine, D_sine = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
X_sineE, y_sineE, A_sineE, D_sineE = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
X_FineSine, y_FineSine, A_FineSine, D_FineSine = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "FineSine", 0.0)
X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Step", 0.0)
X_peaks, y_peaks, A_peaks, D_peaks = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Peaks", 0.0)
X_spiral, y_spiral, A_spiral, D_spiral = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)
## Run the C omega sweep on each 2D benchmark and save the results
C_sweep = [0:0.1:1...]
N_max = 2
monotonicity = Nonstrict()

# Run the omega sweep on the Sine 2D benchmark
resData_2_1 = zeros(length(C_sweep),N_max+1)
finalTheta_2_1 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_sine, y_sine, A_sine, D_sine,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_1[i,:] = res_history_1
    finalTheta_2_1[i,:,:] = Theta_1
end

# Run the omega sweep on the SineE benchmark
resData_2_2 = zeros(length(C_sweep),N_max+1)
finalTheta_2_2 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_sineE, y_sineE, A_sineE, D_sineE,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_2[i,:] = res_history_1
    finalTheta_2_2[i,:,:] = Theta_1
end

# Run the omega sweep on the FineSine benchmark
resData_2_3 = zeros(length(C_sweep),N_max+1)
finalTheta_2_3 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_FineSine, y_FineSine, A_FineSine, D_FineSine,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_3[i,:] = res_history_1
    finalTheta_2_3[i,:,:] = Theta_1
end

# Run the omega sweep on the Step benchmark
resData_2_4 = zeros(length(C_sweep),N_max+1)
finalTheta_2_4 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_Step, y_Step, A_Step, D_Step,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_4[i,:] = res_history_1
    finalTheta_2_4[i,:,:] = Theta_1
end

# Run the omega sweep on the Peaks benchmark
resData_2_5 = zeros(length(C_sweep),N_max+1)
finalTheta_2_5 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_Peaks, y_Peaks, A_Peaks, D_Peaks,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_5[i,:] = res_history_1
    finalTheta_2_5[i,:,:] = Theta_1
end
# Run the omega sweep on the Spiral benchmark
resData_2_6 = zeros(length(C_sweep),N_max+1)
finalTheta_2_6 = zeros(length(C_sweep),N_max,5)
@threads for i in eachindex(C_sweep)
    C = C_sweep[i]
    solver_OmegaSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(C, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
    X_Spiral, y_Spiral, A_Spiral, D_Spiral,
    N_max=N_max,
    solver=solver_OmegaSweep,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData_2_6[i,:] = res_history_1
    finalTheta_2_6[i,:,:] = Theta_1
end

@save "C_OmegaSweep_2D.jld2" X_2D C_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_3 finalTheta_2_3 resData_2_4 finalTheta_2_4 resData_2_5 finalTheta_2_5 resData_2_6 finalTheta_2_6