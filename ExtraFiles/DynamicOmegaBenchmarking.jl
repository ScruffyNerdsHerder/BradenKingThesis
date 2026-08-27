using Pkg
pkg"activate ."
## File to run a series of omega sweeps on a series of benchmarks
# file_path1 = joinpath(@__DIR__, "Benchmarks.jl")
include("Benchmarks.jl")
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

## Test a series of ratios on the 2D Sine benchmark to see how the dynamic omega solver performs
function run_dynamic_omega_ratio_sweep(benchmark_name::String)
    omega_TV = 2.5

     if benchmark_name == "Spiral"
        X, y, A, D = benchmark_2D_data("yu_spiral_doublecone_30deg.jld2", benchmark_name, 0.0)
    else
        X, y, A, D = benchmark_2D_data("normal_1000_doublecone_30deg.jld2", benchmark_name, 0.0)
    end
    ratios = [2, 1, .25, .0625, 1/64]
    # as = collect(range(0.0, stop=0.999, length=50))
    N_max = 500
    monotonicity=Nonstrict()
    resData = zeros(length(ratios)+1,N_max+1)
    finalTheta = zeros(length(ratios)+1,N_max,5)

    @threads for i in eachindex(ratios)
        ratio = ratios[i]
        solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_DynamicOmega(ratio, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
        error_threshold = [0.0, 0.0, 0.0]
        print_iter=false
        Theta, res_history, _, _, _, _, _ = train_RBFN(
        X, y, A, D,
        N_max=N_max,
        solver=solver_LBFGS,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 2}
        );
        resData[i,:] = res_history
        finalTheta[i,:,:] = Theta
    end

    solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    Theta, res_history, _, _, _, _, _ = train_RBFN(
    X, y, A, D,
    N_max=N_max,
    solver=solver_LBFGS,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
    );
    resData[end,:] = res_history
    finalTheta[end,:,:] = Theta
    ## Save the results of the ratio sweep for later plotting
    return resData, finalTheta, ratios, omega_TV
end

resData_2_2_1, finalTheta_2_2_1, ratios, omega_TV = run_dynamic_omega_ratio_sweep("Sine")
resData_2_2_2, finalTheta_2_2_2, ratios, omega_TV = run_dynamic_omega_ratio_sweep("SineE")
resData_2_2_3, finalTheta_2_2_3, ~, ~ = run_dynamic_omega_ratio_sweep("FineSine")
resData_2_2_4, finalTheta_2_2_4, ~, ~ = run_dynamic_omega_ratio_sweep("Step")
resData_2_2_5, finalTheta_2_2_5, ~, ~ = run_dynamic_omega_ratio_sweep("Peaks")
resData_2_2_6, finalTheta_2_2_6, ~, ~ = run_dynamic_omega_ratio_sweep("Spiral")

@save "DynamicOmegaRatios_2D.jld2" X ratios omega_TV resData_2_2_1 finalTheta_2_2_1 resData_2_2_2 finalTheta_2_2_2 resData_2_2_3 finalTheta_2_2_3 resData_2_2_4 finalTheta_2_2_4 resData_2_2_5 finalTheta_2_2_5 resData_2_2_6 finalTheta_2_2_6

    ## Plot the results of the ratio vs the residual error history
    f = GLMakie.Figure()
    ax = Axis(f[1, 1], yscale = log10, xlabel = "N", ylabel = "Residual Error History", title = "Dynamic Omega Ratio Sweep for 2D Sine Benchmark")
    ratio1 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[1,:], color = :blue, label = "Ratio = 1")
    ratio2 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[2,:], color = :red, label = "Ratio = 0.25")
    ratio3 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[3,:], color = :green, label = "Ratio = 0.0625")
    ratio4 = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[4,:], color = :orange, label = "Ratio = 0.015625")
    constant_omega = GLMakie.scatter!(ax, [0:N_max...], resData_2_2_1[end,:], color = :purple, label = "Constant Omega")
    Legend(f[1,2], [ratio1, ratio2, ratio3, ratio4, constant_omega], ["Ratio = 1", "Ratio = 1/4", "Ratio = 1/16", "Ratio = 1/64", "Constant Omega"], title = "Legend")
    f
