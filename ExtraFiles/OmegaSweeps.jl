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

