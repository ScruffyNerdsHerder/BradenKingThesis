using Pkg
pkg"activate ."

file_path1 = joinpath(@__DIR__, "..", "1D-Benchmarks", "Benchmarks_1D.jl")
include(file_path1)
include("ExperimentalSolvers.jl")
using .Benchmarks
using .AltSolvers
using BenchmarkTools
using Plots
using SLFA
using Optim
using Measures  
using JLD2


## Run the RBFN training with different solvers on each benchmark and plot the convergence histories
# First run the LSQ solver
omega = 0.0
error_threshold = [0.0, 0.0, 0.0]
print_iter=false
dx = 0.01
X = [0:dx:10...]
noiseLevel = 0.05
y_clean = Benchmarks.Benchmark_1D(X,"CompositeSine",0.0)
y_noisy = Benchmarks.Benchmark_1D(X,"CompositeSine",noiseLevel)
start_gap = noiseLevel/2*maximum(y_clean) # start gap for the monotonicity constraint

solver_NM(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = SLFA.lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
solver_CG(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = AltSolvers.lsq_TV_solver_CG(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = AltSolvers.lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
solver_GD(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = AltSolvers.lsq_TV_solver_GradientDescent(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

monotonicity=Nonstrict()
Theta_IG, res_history_IG, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
Theta_LSQ, res_history_LSQ, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=SLFA.lsq_solver, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
Theta_NM, res_history_NM, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=solver_NM, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
Theta_CG, res_history_CG, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=solver_CG, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
Theta_LBFGS, res_history_LBFGS, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
Theta_GD, res_history_GD, _, _, _, _, _, _, _ = train_RBFN(X, y_clean, N_max=200, solver=solver_GD, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);

monotonicity=Nonstrict() # must be nonstrict for noisy data
Theta_IG_noisy, res_history_IG_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);
Theta_LSQ_noisy, res_history_LSQ_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=SLFA.lsq_solver, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);
Theta_NM_noisy, res_history_NM_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=solver_NM, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);
Theta_CG_noisy, res_history_CG_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=solver_CG, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);
Theta_LBFGS_noisy, res_history_LBFGS_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);
Theta_GD_noisy, res_history_GD_noisy, _, _, _, _, _, _, _ = train_RBFN(X, y_noisy, N_max=50, solver=solver_GD, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, start_gap=start_gap);

## Plot The convergence histories for the clean data
N_all = 0:200
lw = 6
cleanlim = (1e-3, 10.0)
noisylim = (1e-1, 10.0)
fig = plot(N_all, getindex.(res_history_IG, 1), 
    label="Initial Guess",
    title="RMSE Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    layout=(2,3), 
    subplot=1,
    ylim=cleanlim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ, 1), label="LSQ Solver", linewidth=lw, subplot=1)
plot!(fig, N_all, getindex.(res_history_NM, 1), label="Nelder-Mead", linewidth=lw, subplot=1)
plot!(fig, N_all, getindex.(res_history_LBFGS, 1), label="LBFGS", linewidth=lw, subplot=1)# Weirdly same as LSQ solver
plot!(fig, N_all, getindex.(res_history_CG, 1), label="Conjugate Gradient", linewidth=lw, subplot=1)
plot!(fig, N_all, getindex.(res_history_GD, 1), label="Gradient Descent", linewidth=lw, subplot=1)

plot!(fig, N_all, getindex.(res_history_IG, 2), 
    label="Initial Guess",
    title="Max Magnitude Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    subplot=2,
    ylim=cleanlim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ, 2), label="LSQ Solver", linewidth=lw, subplot=2)
plot!(fig, N_all, getindex.(res_history_NM, 2), label="Nelder-Mead", linewidth=lw, subplot=2)
plot!(fig, N_all, getindex.(res_history_LBFGS, 2), label="LBFGS", linewidth=lw, subplot=2) # Weirdly same as LSQ solver
plot!(fig, N_all, getindex.(res_history_CG, 2), label="Conjugate Gradient", linewidth=lw, subplot=2)
plot!(fig, N_all, getindex.(res_history_GD, 2), label="Gradient Descent", linewidth=lw, subplot=2)

plot!(fig, N_all, getindex.(res_history_IG, 3), 
    label="Initial Guess",
    title="Bound Difference Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    subplot=3,
    ylim=cleanlim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ, 3), label="LSQ Solver", linewidth=lw, subplot=3)
plot!(fig, N_all, getindex.(res_history_NM, 3), label="Nelder-Mead", linewidth=lw, subplot=3)
plot!(fig, N_all, getindex.(res_history_LBFGS, 3), label="LBFGS", linewidth=lw, subplot=3) # Weirdly same as LSQ solver
plot!(fig, N_all, getindex.(res_history_CG, 3), label="Conjugate Gradient", linewidth=lw, subplot=3)
plot!(fig, N_all, getindex.(res_history_GD, 3), label="Gradient Descent", linewidth=lw, subplot=3)

# Plot The convergence histories for the noisy data
N_all = 0:50
lw = 6
plot!(fig, N_all, getindex.(res_history_IG_noisy, 1), 
    label="Initial Guess",
    title="Noisy RMSE Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    subplot=4,
    ylim=noisylim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ_noisy, 1), label="LSQ Solver", linewidth=lw, subplot=4)
plot!(fig, N_all, getindex.(res_history_NM_noisy, 1), label="Nelder-Mead", linewidth=lw, subplot=4)
plot!(fig, N_all, getindex.(res_history_LBFGS_noisy, 1), label="LBFGS", linewidth=lw, subplot=4)
plot!(fig, N_all, getindex.(res_history_CG_noisy, 1), label="Conjugate Gradient", linewidth=lw, subplot=4)
plot!(fig, N_all, getindex.(res_history_GD_noisy, 1), label="Gradient Descent", linewidth=lw, subplot=4)

plot!(fig , N_all, getindex.(res_history_IG_noisy, 2), 
    label="Initial Guess",
    title="Noisy Max Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    subplot=5,
    ylim=noisylim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ_noisy, 2), label="LSQ Solver", linewidth=lw, subplot=5)
plot!(fig, N_all, getindex.(res_history_NM_noisy, 2), label="Nelder-Mead", linewidth=lw, subplot=5)
plot!(fig, N_all, getindex.(res_history_LBFGS_noisy, 2), label="LBFGS", linewidth=lw, subplot=5)
plot!(fig, N_all, getindex.(res_history_GD_noisy, 2), label="Gradient Descent", linewidth=lw, subplot=5)
plot!(fig, N_all, getindex.(res_history_CG_noisy, 2), label="Conjugate Gradient", linewidth=lw, subplot=5)

plot!(fig, N_all, getindex.(res_history_IG_noisy, 3), 
    label="Initial Guess",
    title="Noisy Bound Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24,
    subplot=6,
    ylim=noisylim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ_noisy, 3), label="LSQ Solver", linewidth=lw, subplot=6)
plot!(fig, N_all, getindex.(res_history_NM_noisy, 3), label="Nelder-Mead", linewidth=lw, subplot=6)
plot!(fig, N_all, getindex.(res_history_LBFGS_noisy, 3), label="LBFGS", linewidth=lw, subplot=6) # Weirdly same as LSQ solver
plot!(fig, N_all, getindex.(res_history_CG_noisy, 3), label="Conjugate Gradient", linewidth=lw, subplot=6)
plot!(fig, N_all, getindex.(res_history_GD_noisy, 3), label="Gradient Descent", linewidth=lw, subplot=6)

    
## Get Final errors for each solver
finalErrorOrder = ["Initial Guess", "LSQ Solver", "Nelder-Mead", "LBFGS", "Conjugate Gradient", "Gradient Descent"]
finalCleanErrors = [res_history_IG[end], res_history_LSQ[end], res_history_NM[end], res_history_LBFGS[end], res_history_CG[end], res_history_GD[end]]
finalNoisyErrors = [res_history_IG_noisy[end], res_history_LSQ_noisy[end], res_history_NM_noisy[end], res_history_LBFGS_noisy[end], res_history_CG_noisy[end], res_history_GD_noisy[end]]

@save "SolverComparison_CompositeSine_.jld2"  finalErrorOrder finalCleanErrors X y_clean y_noisy start_gap Theta_IG res_history_IG Theta_LSQ res_history_LSQ Theta_NM res_history_NM Theta_CG res_history_CG Theta_LBFGS res_history_LBFGS Theta_GD res_history_GD Theta_IG_noisy res_history_IG_noisy Theta_LSQ_noisy res_history_LSQ_noisy Theta_NM_noisy res_history_NM_noisy Theta_CG_noisy res_history_CG_noisy Theta_LBFGS_noisy res_history_LBFGS_noisy Theta_GD_noisy res_history_GD_noisy

## Plot The final errors for each solver for each benchmark




## Do the Time Benchmarking for the clean and noisy data
monotonicity=Strict() # must be for noisy data
@btime train_RBFN(X, y_clean, N_max=200, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_clean, N_max=200, solver=SLFA.lsq_solver, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_clean, N_max=200, solver=solver_NM, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_clean, N_max=200, solver=solver_CG, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_clean, N_max=200, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_clean, N_max=200, solver=solver_GD, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);

monotonicity=Nonstrict() # must be nonstrict for noisy data
@btime train_RBFN(X, y_noisy, N_max=50, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_noisy, N_max=50, solver=SLFA.lsq_solver, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_noisy, N_max=50, solver=solver_NM, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_noisy, N_max=50, solver=solver_CG, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_noisy, N_max=50, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
@btime train_RBFN(X, y_noisy, N_max=50, solver=solver_GD, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);