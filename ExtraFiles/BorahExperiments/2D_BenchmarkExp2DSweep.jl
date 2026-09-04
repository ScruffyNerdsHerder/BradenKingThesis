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

BLAS.set_num_threads(2)
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

N0_Range = collect(parse(Int, ARGS[1]):50:parse(Int, ARGS[2]))
C_Range = collect(parse(Float64, ARGS[3]):0.1:parse(Float64, ARGS[4]))

# Tests for small grid
# N0_Range = collect(0:50:100)
# C_Range = collect(0:0.1:0.2)
# C_Range = collect(1)

Grid_sweep = [(N0, C) for N0 in N0_Range for C in C_Range]

# # Generate benchmark data with no noise (Borah version)
# X_Sine, y_Sine, A_Sine, D_Sine = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
# X_SineE, y_SineE, A_SineE, D_SineE = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
# X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Step", 0.0)

## Generate benchmark data with no noise (Local version)
X_Sine, y_Sine, A_Sine, D_Sine = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
X_SineE, y_SineE, A_SineE, D_SineE = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Step", 0.0)
## Run the Coarse Omega Constant sweep on each 2D benchmark and save the results
N_max = 5
monotonicity = Nonstrict()

resData_2_1 = zeros(length(N0_Range)*length(C_Range),N_max+1)
finalTheta_2_1 = zeros(length(N0_Range)*length(C_Range),N_max,5)
resData_2_2 = zeros(length(N0_Range)*length(C_Range),N_max+1)
finalTheta_2_2 = zeros(length(N0_Range)*length(C_Range),N_max,5)
resData_2_4 = zeros(length(N0_Range)*length(C_Range),N_max+1)
finalTheta_2_4 = zeros(length(N0_Range)*length(C_Range),N_max,5)
params_solver = [0.0, 0.0]
@threads for i in eachindex(N0_Range)
    N0 = N0_Range[i]
    # Loop over the C values for each N0 value
    for j in eachindex(C_Range)
        index = (i-1)*length(C_Range) + j
        C = C_Range[j]
        print("Running N0 = $N0, C = $(C) for index $index")
        # Define the solver function for the current N0 and C values
        params_solver[1] = N0
        params_solver[2] = C
        solver_N0Sweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_Omega2DSweepExpDecrease(params_solver, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
        error_threshold = [0.0, 0.0, 0.0]
        print_iter=false
        # Run the N0 sweep on the Sine 2D benchmark
            Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
            X_Sine, y_Sine, A_Sine, D_Sine,
            N_max=N_max,
            solver=solver_N0Sweep,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test,
            T_phi = Gaussian{Isotropic, Float64, 2}
            );
            resData_2_1[index,:] = res_history_1
            finalTheta_2_1[index,:,:] = Theta_1
        # Run the N0 sweep on the SineE benchmark
            Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
            X_SineE, y_SineE, A_SineE, D_SineE,
            N_max=N_max,
            solver=solver_N0Sweep,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test,
            T_phi = Gaussian{Isotropic, Float64, 2}
            );
            resData_2_2[index,:] = res_history_1
            finalTheta_2_2[index,:,:] = Theta_1
        # Run the N0 sweep on the Step benchmark
            Theta_1, res_history_1, _, _, _, _, _ = train_RBFN(
            X_Step, y_Step, A_Step, D_Step,
            N_max=N_max,
            solver=solver_N0Sweep,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test,
            T_phi = Gaussian{Isotropic, Float64, 2}
            );
            resData_2_4[index,:] = res_history_1
            finalTheta_2_4[index,:,:] = Theta_1
    end
end

X_2D = X_Sine
@save "2D_BenchmarkExp2DSweep"*string(ARGS[1])*"_"*string(ARGS[3])*".jld2" X_2D Grid_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_4 finalTheta_2_4
# @save "2D_BenchmarkExp2DSweep.jld2" X_2D Grid_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_4 finalTheta_2_4
end