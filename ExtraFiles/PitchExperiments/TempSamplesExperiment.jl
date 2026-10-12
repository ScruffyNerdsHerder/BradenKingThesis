using Pkg
pkg"activate ."

# include(file_path1)
include(joinpath(@__DIR__, "..", "ExperimentalSolvers.jl"))
include(joinpath(@__DIR__, "NeighborFinder", "src", "ObstructionGraphs.jl"))
using .ObstructionGraphs
using SparseArrays
using JLD2
using Random
using SLFA
using Optim
using Base.Threads
using StatsBase
using Tables

    BLAS.set_num_threads(24)
## Setup the overall functions for the RBFN training
    const T = Float64
       function build_double_cone_graph_bruteforce(
       X::AbstractMatrix{T},
       alpha::T;
       sparse_output::Bool = true
       ) where {T<:Real}

       rule = ObstructionGraphs.ObstructionRule(R -> doubleCone(R, alpha))

       if sparse_output
               return ObstructionGraphs.build_sparseGraph_threaded(X, rule)
       else
               return ObstructionGraphs.build_denseGraph_threaded(X, rule)
       end
       end

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

## Generate the A,D matrices
@load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
n_dims = size(X, 2)
N0 = 5
C = 0
N_max = 100
monotonicity = Nonstrict()
Rounded_error(res, res_validation, res_history, N) = [RMSE(res) ];
print_iter=false
solver_dewpoint(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_Omega2DSweepExpDecrease((N0,C), theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
error_threshold = [0.0, 0.0, 0.0]
sampleCount = 300

    # Remove the NaN values from the dataset and randomly sample 10,000 points
    samples = randperm(size(y, 1))[1:sampleCount]
    X_flip = Float64.(permutedims(X,(2,1)))
    X_temp_raw = X_flip[:,samples]
    y_temp_raw = y[samples]
    y_temp_nans = findall(!isnan, y_temp_raw)
    X_temp = X_temp_raw[:, y_temp_nans]
    y_temp = y_temp_raw[y_temp_nans]
    A_temp, D_temp = build_double_cone_graph_bruteforce(X_temp, T.(tan.(deg2rad.(30))))
    
    # Generate a single RBFN on the Temperature dataset and save the results to a file
    start_gap = 0.000*maximum(y_temp)
    FinalTheta_temp, resData_temp, _, _, _, _, _ = train_RBFN(
    X_temp, y_temp, A_temp, D_temp,
    N_max=N_max,
    solver=solver_dewpoint,
    conv_conditions=rmse_error,
    print_iter=print_iter,
    is_monotonic=monotonicity,
    get_initial_guess = max_dist_test_Aniso,
    redistribute_wts_final=false,   
    T_phi = Gaussian{Anisotropic{Aligned}, Float64, ndims},
    start_gap = start_gap
    );

    @save "PitchTesttemp." * string(sampleCount) * ".jld2" X_temp y_temp FinalTheta_temp resData_temp



    X_temp2 = X_temp[[1, 2, 12], :]