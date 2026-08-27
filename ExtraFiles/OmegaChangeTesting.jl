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
using GLMakie
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

function plot_results(X, y_train, y_eval, N, res_history_LBFGS)
    N_all = 0:500
    f, ax, tr = CairoMakie.tricontourf(X[1,:],X[2,:],y_eval)
    resize!(f, 1300, 700)
    Label(f[1,1, Top()],"Final Network")
    CairoMakie.tricontourf(f[1,2],X[1,:],X[2,:],y_train)
    Label(f[1,2, Top()],"Initial Data")
    CairoMakie.Colorbar(f[1,3],tr)
    ax1 = Axis(f[1,4], yscale = log10, title = "Residual Error History")
    CairoMakie.scatter!(ax1, N_all, getindex.(res_history_LBFGS, 1),)
    f
    display(f)
end

## Run the 2D benchmark with a linearly decaying omega (start at something and end at 0)
    # Linear decay
  
  X, y, A, D = benchmark_2D_data("ExtraFiles/data/uniform_1000_doublecone_30deg.jld2", "Sine", 0.0)
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)
    omegas_init = collect(range(0.0, stop=10, length=50))
    N_max = 500
    monotonicity=Nonstrict()
    finalres = zeros(length(omegas_init))
    finaltheta = zeros(length(omegas_init),N_max,5)

    @threads for i in eachindex(omegas_init)
        omega_init = omegas_init[i]
        omega_a = omegas_init[i]/N_max
        function omega_decrease_n(n) 
            return omega_init - (omega_a) * n
        end
        solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaSweep(omega_decrease_n, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
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
        finalres[i] = res_history[end]
        finaltheta[i,:,:] = Theta
    end
    ## Plot the residuals vs omega
    @save "LinearDecreasingOmegaSweep.jdl2" omegas_init finalres finaltheta
    @load "LinearDecreasingOmegaSweep.jdl2" omegas_init finalres
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    yscale = log10,
    xlabel = "Initial Omega",
    ylabel = "Final Residual Error",
    title = "Linearly Decreasing Omega Sweep (w_final = 0) for Sine Benchmark",
    )
    GLMakie.scatter!(ax,omegas_init, finalres)
    f
    ##
    
## Run the 2D benchmark with a linearly increasing omega (start at 0 and end at 2)
    # Linear decay
    X, y, A, D = benchmark_2D_data("ExtraFiles/data/uniform_1000_doublecone_30deg.jld2", "Sine", 0.0)
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)
    omegas_init = collect(range(0.0, stop=2.0, length=50))
    N_max = 250
    monotonicity=Nonstrict()
    finalres = zeros(length(omegas_init))
    finaltheta = zeros(length(omegas_init), N_max, 5)

    for i in eachindex(omegas_init)
        omega_final = omegas_init[i]
        omega_a = .001
        function omega_increase_n(n) 
            return omega_init + (omega_a) * n
        end
        solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaSweep(omega_decrease_n, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
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
        finalres[i] = res_history[end]
        finaltheta[i,:,:] = Theta
    end
    ## Plot the residuals vs omega
    @save "LinearIncreasingOmegaSweep.jdl2" omegas_init finalres finaltheta
    
    @load "LinearIncreasingOmegaSweep.jdl2" omegas_init finalres
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    yscale = log10,
    xlabel = "Initial Omega",
    ylabel = "Final Residual Error",
    title = "Linearly Increasing Omega Sweep (w=w_init+w_a*n) for Sine Benchmark",
    )
    GLMakie.scatter!(ax,omegas_init, finalres)
    f
    
## Run the 2D benchmark with a exponentially decreasing omega (start at something and decay exponentially)
    X, y, A, D = benchmark_2D_data("ExtraFiles/data/uniform_1000_doublecone_30deg.jld2", "Sine", 0.0)
    # X, y, A, D = benchmark_2D_data("ExtraFiles/data/yu_spiral_doublecone_30deg.jld2", "Spiral", 0.0)
    omegas_init = collect(range(0.0, stop=10, length=10))
    # as = collect(range(0.0, stop=0.999, length=50))
    N_max = 10
    monotonicity=Nonstrict()
    resData = zeros(length(omegas_init),N_max+1)
    finalTheta = zeros(length(omegas_init),N_max,5)

    @threads for i in eachindex(omegas_init)
        omega_init = omegas_init[i]
        # a = as[i]

        solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_OmegaCSweepExpDecrease(omega_init, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
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
    
    ## Create the errors vs Ns vs Omega matrices
    Ns = collect(0:N_max)
    ers = resData
    scale = ReversibleScale(x -> log2(x), x -> 2.0^x)
    ## Plot the residuals vs omega
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    xlabel = "N's",
    xlabelsize = 20,
    ylabelsize = 20,
    ylabel = "Initial Omega",
    )
    GLMakie.heatmap!(ax,  Ns, omegas_init, ers', colormap = :jet1, 
    colorscale = scale,
     colorrange = (1e-2, 1))
    # Make the color scale correctly
    # MAke the filename just 1, 2, etc

    f
    @save "LinearIncreasingOmegaSweep.jdl2" omegas_init finalres finaltheta
    
    @load "LinearIncreasingOmegaSweep.jdl2" omegas_init finalres
    f = GLMakie.Figure()
    ax = Axis(f[1,1],
    yscale = log10,
    xlabel = "A-Omega",
    ylabel = "Final Residual Error",
    title = "Linearly Increasing Omega Sweep (w=w_init+w_a*n) for SineE Benchmark",
    )
    GLMakie.scatter!(ax,omegas_init, resData[:,end])
    f
    ## 