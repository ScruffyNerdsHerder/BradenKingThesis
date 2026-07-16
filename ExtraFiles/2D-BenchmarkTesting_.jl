using Pkg
pkg"activate ."

file_path1 = joinpath(@__DIR__, "..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
include("ExperimentalSolvers.jl")

using BenchmarkTools
using Plots
using SLFA
using Optim
using Measures  
using JLD2
using PrettyTables

function benchmark_2D_data(filename,benchmark,noiseLevel)
    X = load(filename,"X")
    A = load(filename,"A")
    D = load(filename,"D")
    if benchmark == "SinE"
        # Scale the Xes to 0:10 in each direction
        X = 2 .* X
        # Scale the D matrix (X^2+Y^2)
        D = 2*D
    end
    y = Benchmark_2D(X,benchmark,noiseLevel)
    return X, y, A, D
end

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



## Create 2D benchmarks
## Run the RBFN training with different solvers on each benchmark and plot the convergence histories
X, y, A, D = benchmark_2D_data("ExtraFiles/data/normal_1000_doublecone_15deg.jld2", "Sine", 0.0)
# First run the LSQ solver
omega = 0.1
error_threshold = [0.0, 0.0, 0.0]
print_iter = false
monotonicity = Nonstrict()

surface(X[1,:],X[2,:],y)

##

solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

Theta_IG, res_history_IG, _, _, _, _, _ = train_RBFN(
    X, y, A, D,
    N_max=500,
    solver=solver_LBFGS,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);



## Create RBFN and plot comparison to initial value
network = RBFN(Theta_IG, Gaussian{Isotropic, Float64, 2})
X_eval, ~, A, D = benchmark_2D_data("ExtraFiles/data/uniform_10000_doublecone_30deg.jld2", "Sine", 0.0)
y_val = network(X_eval)
N_all = 0:200
fig = surface(X_eval[1,:],X_eval[2,:],y_val,
    layout=(1,3), 
    subplot=1,
    legend=false,
    size=(1500, 1000),
    zlims=(-1, 1),
)
surface!(fig, X[1,:],X[2,:],y, subplot = 2, legend=false)
plot!(fig, getindex.(res_history_IG, 1), subplot=3, legend=false, label = false)

## Run the RBFN training with different solvers on each benchmark and plot the convergence histories
# First run the LSQ solver
omega = 0
error_threshold = [0.0, 0.0, 0.0]
print_iter=false
dx = 0.01
noiseLevel = 0.05
X, y_clean, A, D = benchmark_2D_data("ExtraFiles/data/normal_10000_doublecone_30deg.jld2", "Sine", 0.0)

start_gap = noiseLevel/2*maximum(y_clean) # start gap for the monotonicity constraint

solver_NM(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = SLFA.lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

monotonicity=Nonstrict()
Theta_LBFGS, res_history_LBFGS, _, _, _, _, _ = train_RBFN(
    X, y_clean, A, D,
    N_max=500,
    solver=solver_LBFGS,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);
Theta_LSQ, res_history_LSQ, _, _, _, _, _ = train_RBFN(
    X, y_clean, A, D,
    N_max=500,
    solver=lsq_solver,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);
Theta_NM, res_history_NM, _, _, _, _, _ = train_RBFN(
    X, y_clean, A, D,
    N_max=500,
    solver=solver_NM,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);
Theta_IG, res_history_IG, _, _, _, _, _ = train_RBFN(
    X, y_clean, A, D,
    N_max=500,
    solver=SLFA.initial_guess,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);
 # Plot The convergence histories for the data
N_all = 0:500
lw = 6
cleanlim = (.2e-2, 10.0)
fig = plot(N_all, res_history_IG, 
    label="Initial Guess",
    title="RMSE Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:topright,
    tickfont=18,
    titlefont=24,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=12,
    labelfontsize=24, 
    ylim=cleanlim,
    bottom_margin=10mm
)
plot!(fig, N_all, getindex.(res_history_LSQ, 1), label="LSQ Solver", linewidth=lw, subplot=1)
plot!(fig, N_all, getindex.(res_history_NM, 1), label="Nelder-Mead", linewidth=lw, subplot=1)
plot!(fig, N_all, getindex.(res_history_LBFGS, 1), label="LBFGS", linewidth=lw, subplot=1)# Weirdly same as LSQ solver

## Plot the resulting 3D geometry
network = RBFN(Theta_LBFGS, Gaussian{Isotropic, Float64, 2})
X_eval, ~, A, D = benchmark_2D_data("ExtraFiles/data/uniform_10000_doublecone_30deg.jld2", "Sine", 0.0)
 
y_val = network(X_eval)

fig = surface(X_eval[1,:],X_eval[2,:],y_val,
    layout=(1,3), 
    subplot=1,
    legend=false,
    size=(1500, 1000),
    zlims=(-1, 1),
)
surface!(fig, X[1,:],X[2,:],y_clean, subplot = 2, legend=false)
plot!(fig, getindex.(res_history_LBFGS, 1), subplot=3, legend=false, label = false)

## Do comparison of the 15deg-30deg-45deg versions
omega = 0.0
error_threshold = [0.0, 0.0, 0.0]
print_iter=false
dx = 0.01
noiseLevel = 0.05
start_gap = noiseLevel/2*maximum(y_clean) # start gap for the monotonicity constraint
solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

monotonicity=Nonstrict()
X, y_clean, A, D = benchmark_2D_data("ExtraFiles/data/normal_10000_doublecone_30deg.jld2", "Sine", 0.01)
Theta_LBFGS, res_history_LBFGS_20deg, _, _, _, _, _ = train_RBFN(
    X, y_clean, A, D,
    N_max=500,
    solver=solver_LBFGS,
    conv_thresholds=error_threshold,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test,
    T_phi = Gaussian{Isotropic, Float64, 2}
);

