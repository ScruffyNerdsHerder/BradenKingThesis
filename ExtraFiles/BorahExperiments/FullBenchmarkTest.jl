
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

N0_Range = collect(parse(Int, ARGS[1]):parse(Int, ARGS[2]):parse(Int, ARGS[3]))
C_Range = collect(parse(Float64, ARGS[4]):parse(Float64, ARGS[5]):parse(Float64, ARGS[6]))


# # Tests for small grid
# N0_Range = collect(10)
# C_Range = collect(15)

Grid_sweep = [(N0, C) for N0 in N0_Range for C in C_Range]

# # Generate benchmark data with no noise (Borah version)
X_Sine, y_Sine, A_Sine, D_Sine = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
X_SineE, y_SineE, A_SineE, D_SineE = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Step", 0.0)
X_FineSine, y_FineSine, A_FineSine, D_FineSine = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "FineSine", 0.0)
X_Peaks, y_Peaks, A_Peaks, D_Peaks = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Peaks", 0.0)
X_3DMexHat, y_3DMexHat, A_3DMexHat, D_3DMexHat = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "3DMexHat", 0.0)
X_Gabor, y_Gabor, A_Gabor, D_Gabor = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Gabor", 0.0)
X_SwingCos, y_SwingCos, A_SwingCos, D_SwingCos = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "SwingCos", 0.0)
X_Exponential, y_Exponential, A_Exponential, D_Exponential = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Exponential", 0.0)
X_Spiral, y_Spiral, A_Spiral, D_Spiral = benchmark_2D_data("../data/normal_1000_doublecone_30deg.jld2", "Spiral", 0.0)

# X_Sine, y_Sine, A_Sine, D_Sine = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Sine", 0.0)
# X_SineE, y_SineE, A_SineE, D_SineE = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "SineE", 0.0)
# X_Step, y_Step, A_Step, D_Step = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Step", 0.0)
# X_FineSine, y_FineSine, A_FineSine, D_FineSine = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "FineSine", 0.0)
# X_Peaks, y_Peaks, A_Peaks, D_Peaks = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Peaks", 0.0)
# X_3DMexHat, y_3DMexHat, A_3DMexHat, D_3DMexHat = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "3DMexHat", 0.0)
# X_Gabor, y_Gabor, A_Gabor, D_Gabor = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Gabor", 0.0)
# X_SwingCos, y_SwingCos, A_SwingCos, D_SwingCos = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "SwingCos", 0.0)
# X_Exponential, y_Exponential, A_Exponential, D_Exponential = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Exponential", 0.0)
# X_Spiral, y_Spiral, A_Spiral, D_Spiral = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_30deg.jld2", "Spiral", 0.0)

## Run the Coarse Omega Constant sweep on each 2D benchmark and save the results
N_max = 5
monotonicity = Nonstrict()
rmse_error(res, res_validation, res_history, N) = RMSE(res);

params_solver = [0.0, 0.0]
expParams = [0.0 10; 10 0; 25 1; 25 400]
@threads :static for i = 1:3
    C = expParams[i,1]
    N0 = expParams[i,2]
        # Define the solver function for the current N0 and C values
        solver_2DSweep(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_Omega2DSweepExpDecrease((N0,C), theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
        error_threshold = [0.0, 0.0, 0.0]
        print_iter=false
        # Run the N0 sweep on the Sine 2D benchmark
        finalTheta_2_1[i,:,:], resData_2_1[i,:] , _, _, _, _, _ = train_RBFN(
            X_Sine, y_Sine, A_Sine, D_Sine,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        # Run the N0 sweep on the SineE benchmark
        finalTheta_2_2[index,:,:], resData_2_2[index,:], _, _, _, _, _ = train_RBFN(
            X_SineE, y_SineE, A_SineE, D_SineE,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        # Run the N0 sweep on the FineSine benchmark
        finalTheta_2_3[index,:,:], resData_2_3[index,:], _, _, _, _, _ = train_RBFN(
            X_FineSine, y_FineSine, A_FineSine, D_FineSine,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        # Run the N0 sweep on the Step benchmark
        finalTheta_2_4[index,:,:], resData_2_4[index,:], _, _, _, _, _ = train_RBFN(
            X_Step, y_Step, A_Step, D_Step,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_5[index,:,:], resData_2_5[index,:], _, _, _, _, _ = train_RBFN(
            X_Peaks, y_Peaks, A_Peaks, D_Peaks,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_6[index,:,:], resData_2_6[index,:], _, _, _, _, _ = train_RBFN(
            X_3DMexHat, y_3DMexHat, A_3DMexHat, D_3DMexHat,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_7[index,:,:], resData_2_7[index,:], _, _, _, _, _ = train_RBFN(
            X_Gabor, y_Gabor, A_Gabor, D_Gabor,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_8[index,:,:], resData_2_8[index,:], _, _, _, _, _ = train_RBFN(
            X_SwingCos, y_SwingCos, A_SwingCos, D_SwingCos,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_9[index,:,:], resData_2_9[index,:], _, _, _, _, _ = train_RBFN(
            X_Exponential, y_Exponential, A_Exponential, D_Exponential,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
        finalTheta_2_10[index,:,:], resData_2_10[index,:], _, _, _, _, _ = train_RBFN(
            X_Spiral, y_Spiral, A_Spiral, D_Spiral,
            N_max=N_max,
            solver=solver_2DSweep,
            conv_conditions=rmse_error,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test_Aniso,
            redistribute_wts_final=false,   
            T_phi = Gaussian{Anisotropic{Aligned}, Float64, 2}
            );
    end
end

X_2D = X_Sine
@save "2D_BenchmarkExp2DAnisoSweep"*string(ARGS[1])*":"*string(ARGS[2])*":"*string(ARGS[3])*"_"*string(ARGS[4])*":"*string(ARGS[5])*":"*string(ARGS[6])*".jld2" X_2D Grid_sweep resData_2_1 finalTheta_2_1 resData_2_2 finalTheta_2_2 resData_2_3 finalTheta_2_3 resData_2_4 finalTheta_2_4
end