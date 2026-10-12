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

BLAS.set_num_threads(24)
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
# @load "ExtraFiles/RepositoryBenchmarkData/Pitch_Data.jdl2" X_Pitch y_Pitch A_Pitch D_Pitch

@load "obs_temp_c_graphs.jld2" X_temp y_temp A_temp D_temp
N0 = 5
C = 20
N_max = 150
monotonicity = Nonstrict()
rmse_error(res, res_validation, res_history, N) = RMSE(res);
start_gap = 0.001*maximum(y_temp)
print_iter=false
solver_dewpoint(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_Omega2DSweepExpDecrease((N0,C), theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
error_threshold = [0.0, 0.0, 0.0]

# Run the a single test on the Temperature benchmark
        FinalTheta_temp, resData_temp, _, _, _, _, _ = train_RBFN(
        X_temp, y_temp, A_temp, D_temp,
        N_max=N_max,
        solver=solver_dewpoint,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 17},
        # start_gap = start_gap
        );

@save "PitchTesttemp0.jld2" X_temp y_temp FinalTheta_temp resData_temp

# Run the a single test on the Pressure benchmark
@load "obs_station_pressure_pa_graphs.jld2" X_pressure y_pressure A_pressure D_pressure
        FinalTheta_pressure, resData_pressure, _, _, _, _, _ = train_RBFN(
        X_pressure, y_pressure, A_pressure, D_pressure,
        N_max=N_max,
        solver=solver_dewpoint,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 17},
        # start_gap = start_gap
        );

@save "PitchTestPressure0.jld2" X_pressure y_pressure FinalTheta_pressure resData_pressure

# Run the a single test on the Dewpoint benchmark
@load "obs_dewpoint_c_graphs.jld2" X_dewpoint y_dewpoint A_dewpoint D_dewpoint
        FinalTheta_dewpoint, resData_dewpoint, _, _, _, _, _ = train_RBFN(
        X_dewpoint, y_dewpoint, A_dewpoint, D_dewpoint,
        N_max=N_max,
        solver=solver_dewpoint,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 17},
        # start_gap = start_gap
        );

@save "PitchTestDewpoint0.jld2" X_dewpoint y_dewpoint FinalTheta_dewpoint resData_dewpoint

# Run the a single test on the Precip benchmark
@load "obs_precip_mm_graphs.jld2" X_precip y_precip A_precip D_precip
        FinalTheta_precip, resData_precip, _, _, _, _, _ = train_RBFN(
        X_precip, y_precip, A_precip, D_precip,
        N_max=N_max,
        solver=solver_dewpoint,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 17},
        # start_gap = start_gap
        );

@save "PitchTestPrecip0.jld2" X_precip y_precip FinalTheta_precip resData_precip

# Run the a single test on the RH benchmark
@load "obs_precip_mm_graphs.jld2" X_precip y_precip A_precip D_precip
        FinalTheta_precip, resData_precip, _, _, _, _, _ = train_RBFN(
        X_precip, y_precip, A_precip, D_precip,
        N_max=N_max,
        solver=solver_dewpoint,
            conv_conditions=rmse_error,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test_Aniso,
        redistribute_wts_final=false,   
        T_phi = Gaussian{Anisotropic{Aligned}, Float64, 17},
        # start_gap = start_gap
        );

@save "PitchTestPrecip0.jld2" X_precip y_precip FinalTheta_precip resData_precip
