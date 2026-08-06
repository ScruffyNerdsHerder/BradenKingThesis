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
using CairoMakie
using Base.Threads
CairoMakie.activate!(inline=false)

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

    if benchmark == "Spiral"
        is = 0:96
        Phis = 1/16*pi.*is
        rs = 6.5.*(104 .- is)/104
        x2 = -rs.*cos.(Phis)
        y =  append!(ones(size(x2)), -ones(size(x2)))
    else
        y = Benchmark_2D(X,benchmark,noiseLevel)
    end
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
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/normal_10000_doublecone_15deg.jld2", "Sine", 0.0)
    X, y, A, D = benchmark_2D_data("ExtraFiles/data/normal_10000_doublecone_15deg.jld2", "Sine", 0.0)
    # First run the LSQ solver
    omega = 0.0
    error_threshold = [0.0, 0.0, 0.0]
    print_iter = false
    monotonicity = Nonstrict()

    surface(X[1,:],X[2,:],y)


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

## Plot the result as input data, network and residual graph
    network = RBFN(Theta_LBFGS, Gaussian{Isotropic, Float64, 2})
    y_eval = network(X)
    N_all = 0:200
    f, ax, tr = CairoMakie.tricontourf(X[1,:],X[2,:],y_eval)
    resize!(f, 1300, 700)
    Label(f[1,1, Top()],"Initial Data")
    CairoMakie.tricontourf(f[1,2],X[1,:],X[2,:],y_eval)
    Label(f[1,2, Top()],"Final Network")
    CairoMakie.Colorbar(f[1,3],tr)
    ax1 = Axis(f[1,4], yscale = log10, title = "RMSE Residual Error History")
    CairoMakie.scatter!(ax1,N_all, getindex.(res_history_LBFGS_20deg, 1))
    display(f)

## Plot the resulting 3D geometry as surface
    network = RBFN(Theta_LBFGS, Gaussian{Isotropic, Float64, 2})
    
    y_val = network(X)

    fig = surface(X_eval[1,:],X_eval[2,:],y_val,
        layout=(1,3), 
        subplot=1,
        legend=false,
        size=(1500, 1000),
        zlims=(-1, 1),
    )
    surface!(fig, X[1,:],X[2,:],y_clean, subplot = 2, legend=false)
    plot!(fig, getindex.(res_history_LBFGS, 1), subplot=3, legend=false, label = false)


## Generate Topo plot using Makie
    N_all = 0:500
    f, ax, tr = CairoMakie.tricontourf(X[1,:],X[2,:],y_clean)
    resize!(f, 1300, 700)
    Label(f[1,1, Top()],"Final Network")
    CairoMakie.tricontourf(f[1,2],X[1,:],X[2,:],y_val)
    Label(f[1,2, Top()],"Initial Data")
    CairoMakie.Colorbar(f[1,3],tr)
    ax1 = Axis(f[1,4], yscale = log10, title = "Residual Error History")
    CairoMakie.scatter!(ax1,N_all, getindex.(res_history_LBFGS, 1),)
    f
    display(f)


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


## Run a benchmark
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/uniform_10000_doublecone_30deg.jld2", "Step", 0.0)
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/grid_100x100_doublecone_30deg.jld2", "FineSine", 0.0)
    X, y, A, D = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)
     omega = 0.0
    error_threshold = [0.0, 0.0, 0.0]
    print_iter=false
    noiseLevel = 0.00
    start_gap = 0 # start gap for the monotonicity constraint
    monotonicity=Nonstrict()

    solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

    Theta_LBFGS, res_history_LBFGS, _, _, _, _, _ = train_RBFN(
        X, y, A, D,
        N_max=1000,
        solver=solver_LBFGS,
        conv_thresholds=error_threshold,
        print_iter=print_iter,
        is_monotonic=monotonicity,
        get_initial_guess = max_dist_test,
        T_phi = Gaussian{Isotropic, Float64, 2}
    );

    network = RBFN(Theta_LBFGS, Gaussian{Isotropic, Float64, 2})
    y_eval = network(X)
    N_all = 0:1000
    f, ax, tr = CairoMakie.tricontourf(X[1,:],X[2,:],y_eval)
    resize!(f, 1300, 700)
    Label(f[1,1, Top()],"Initial Data")
    CairoMakie.tricontourf(f[1,2],X[1,:],X[2,:],y_eval)
    Label(f[1,2, Top()],"Final Network")
    CairoMakie.Colorbar(f[1,3],tr)
    ax1 = Axis(f[1,4], yscale = log10, title = "RMSE Residual Error History")
    CairoMakie.scatter!(ax1,N_all, getindex.(res_history_LBFGS, 1))
    display(f)

    # There's a massive high spot inside the data if you shift even slightly off the input data. (1.00000001 .*X)

## Run Omega Sweep on each 2D benchmark
        X, y, A, D = benchmark_2D_data("ExtraFiles/data/uniform_1000_doublecone_30deg.jld2", "SineE", 0.0)
        # X, y, A, D = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)

         monotonicity=Nonstrict()
        omegas = collect(range(0.0, stop=5, length=200))
        finalres = zeros(length(omegas))
        finaltheta = zeros(length(omegas),500,5)
        i=1
        for omega in omegas
            println(omega)
            solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
            error_threshold = [0.0, 0.0, 0.0]
            print_iter=false
            Theta, res_history, _, _, _, _, _ = train_RBFN(
            X, y, A, D,
            N_max=500,
            solver=solver_LBFGS,
            conv_thresholds=error_threshold,
            print_iter=print_iter,
            is_monotonic=monotonicity,
            get_initial_guess = max_dist_test,
            T_phi = Gaussian{Isotropic, Float64, 2}
            );
            finalres[i] = res_history[end]
            finaltheta[i,:,:] = Theta
            i=i+1
        end
        ## Plot the residuals vs omega
        @save "SineEOmegaSweep.jdl2" omegas finalres finaltheta
        
        @load "SineEOmegaSweep.jdl2" finalres
        f = CairoMakie.Figure()
        ax = Axis(f[1,1],
        yscale = log10,
        xlabel = "Omega",
        ylabel = "Final Residual Error",
        title = "SineE Fine Omega Sweep")
        CairoMakie.scatter!(ax,omegas, finalres)
        f
    ## 



## Generate and investigate support set on step function
dx = 0.2
X = [0:dx:6...]
y = Benchmark_1D(X,"Step",0.0)
fig = CairoMakie.scatter(X,y)

solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
get_nbr_matrix1D(X)
Theta, res_history, _, _, _, _, _, _, _ = train_RBFN(X, y, N_max=1, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);
