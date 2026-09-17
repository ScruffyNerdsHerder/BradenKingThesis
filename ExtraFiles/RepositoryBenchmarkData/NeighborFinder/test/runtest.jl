include(joinpath(@__DIR__, "..", "src", "ObstructionGraphs.jl"))
using .ObstructionGraphs
using Random
using JLD2
using SparseArrays

const OUTDIR = joinpath(@__DIR__, "..", "data")
const SEED = 1234
const T = Float64

const GRID_N = 100
const RANDOM_N_SMALL = 1_000
const RANDOM_N_LARGE = 10_000

const NORMAL_N_SMALL = 1_000
const NORMAL_N_LARGE = 10_000

const SAVE_SPARSE = true
const SPIRAL_X_FILE = "/bsuhome/emmanuelayanful/julia-projects/RBF_Ngbrs/data/SpiralData.jld2"

mkpath(OUTDIR)

@inline grid_index(ix::Int, iy::Int, grid_n::Int) = ix + (iy - 1) * grid_n

function equally_spaced_grid(::Type{T}, grid_n::Int) where {T<:Real}
    xs = collect(range(T(-1), T(1); length=grid_n))
    X = Matrix{T}(undef, 2, grid_n^2)

    for iy in 1:grid_n
        for ix in 1:grid_n
            idx = grid_index(ix, iy, grid_n)
            X[1, idx] = xs[ix]
            X[2, idx] = xs[iy]
        end
    end

    return X
end

function uniform_points(rng::AbstractRNG, ::Type{T}, N::Int) where {T<:Real}
    return T(2) .* rand(rng, T, 2, N) .- T(1)
end

function normal_points(
    rng::AbstractRNG,
    ::Type{T},
    N::Int;
    sgma::Real = 0.35
) where {T<:Real}

    X = Matrix{T}(undef, 2, N)
    sgmaT = T(sgma)

    i = 1
    while i <= N
        x = sgmaT * randn(rng, T)
        y = sgmaT * randn(rng, T)

        # rejection sampling keeps the domain exactly [-1, 1]^2
        if T(-1) <= x <= T(1) && T(-1) <= y <= T(1)
            X[1, i] = x
            X[2, i] = y
            i += 1
        end
    end

    return X
end

# function build_double_cone_graph_bruteforce(
#     X::AbstractMatrix{T},
#     alpha::T;
#     sparse_output::Bool = true
# ) where {T<:Real}

#     rule = ObstructionRule(R -> doubleCone(R, alpha))

#     A_dense, D_dense = build_denseGraph_threaded(X, rule)

#     if sparse_output
#         return dense_to_sparse_edge_graph(A_dense, D_dense)
#     else
#         return A_dense, D_dense
#     end
# end

function build_double_cone_graph_bruteforce(
    X::AbstractMatrix{T},
    alpha::T;
    sparse_output::Bool = true
) where {T<:Real}

    rule = ObstructionRule(R -> doubleCone(R, alpha))

    if sparse_output
        return build_sparseGraph_threaded(X, rule)
    else
        return build_denseGraph_threaded(X, rule)
    end
end

function save_graph_dataset(
    filename::AbstractString;
    X, A, D,
    sample_name::AbstractString,
    angle_deg::Int,
    alpha, sparse_output::Bool)

    path = joinpath(OUTDIR, filename)

    D_convention = sparse_output ?
        "Sparse edge-distance matrix: D[i,j] is stored only when A[i,j] is true. Nonedges are structural zeros. Use A to distinguish nonedges." :
        "Dense edge-distance matrix: D[i,j] is the edge distance when A[i,j] is true, Inf for nonedges, and 0 on the diagonal."

    jldsave(
        path;
        X = X,
        A = A,
        D = D,
        sample_name = sample_name,
        angle_deg = angle_deg,
        alpha = alpha,
        D_convention = D_convention,
        point_convention = "X is 2 x N; X[:, i] is point i."
    )

    println("Saved: ", path)
    println("  sample = ", sample_name)
    println("  angle  = ", angle_deg, " degrees")
    println("  alpha  = ", alpha)
    println("  size(X) = ", size(X))
    println("  nnz(A)  = ", sparse_output ? nnz(A) : count(A))
    println()
end

function main()
    println("Running with ", Threads.nthreads(), " Julia threads")
    println("Saving files to: ", OUTDIR)
    println()

    rng_uniform = MersenneTwister(SEED)
    rng_normal = MersenneTwister(SEED + 1)

    angles = [15, 30, 45, 46]
    alphas = T.(tan.(deg2rad.(angles)))

    println("Angles: ", angles)
    println("Alphas: ", alphas)
    println()

    X_grid = equally_spaced_grid(T, GRID_N)

    # Generate large sets once
    X_uniform_10000 = uniform_points(rng_uniform, T, RANDOM_N_LARGE)
    X_normal_10000 = normal_points(rng_normal, T, NORMAL_N_LARGE)

    # Generate small sets once
    X_uniform_1000 = uniform_points(rng_uniform, T, RANDOM_N_SMALL)
    X_normal_1000 = normal_points(rng_normal, T, NORMAL_N_SMALL)

    # println("Generated point sets once:")
    # println("  grid:    ", size(X_grid))
    # println("  uniform: ", size(X_uniform))
    # println("  normal:  ", size(X_normal))
    # println()

    datasets = [
        ("uniform_1000", X_uniform_1000),
        ("uniform_10000", X_uniform_10000),
        ("normal_1000", X_normal_1000),
        ("normal_10000", X_normal_10000),
    ]

    for (angle_deg, alpha) in zip(angles, alphas)
        println("Building GRID graph for angle ", angle_deg, " degrees")

        A, D = build_double_cone_graph_bruteforce(
            X_grid,
            alpha;
            sparse_output = SAVE_SPARSE
        )

        save_graph_dataset(
            "grid_$(GRID_N)x$(GRID_N)_doublecone_$(angle_deg)deg.jld2";
            X = X_grid,
            A = A,
            D = D,
            sample_name = "grid_$(GRID_N)x$(GRID_N)",
            angle_deg = angle_deg,
            alpha = alpha,
            sparse_output = SAVE_SPARSE
        )

        for (sample_name, X) in datasets
            println("Building ", sample_name, " graph for angle ", angle_deg, " degrees")
    
            A, D = build_double_cone_graph_bruteforce(
                X,
                alpha;
                sparse_output = SAVE_SPARSE
            )
    
            save_graph_dataset(
                "$(sample_name)_doublecone_$(angle_deg)deg.jld2";
                X = X,
                A = A,
                D = D,
                sample_name = sample_name,
                angle_deg = angle_deg,
                alpha = alpha,
                sparse_output = SAVE_SPARSE
            )
        end
    end

    if SPIRAL_X_FILE !== nothing
        println("Loading Yu spiral X from: ", SPIRAL_X_FILE)

        data = load(SPIRAL_X_FILE)
        X_spiral = data["X"]

        if size(X_spiral, 1) != 2 && size(X_spiral, 2) == 2
            X_spiral = Matrix(X_spiral')
        end

        @assert size(X_spiral, 1) == 2 "Expected spiral X to be 2 x N."

        for (angle_deg, alpha) in zip(angles, alphas)
            println("Building YU SPIRAL graph for angle ", angle_deg, " degrees using brute force")

            A, D = build_double_cone_graph_bruteforce(
                X_spiral,
                alpha;
                sparse_output = SAVE_SPARSE
            )

            save_graph_dataset(
                "yu_spiral_doublecone_$(angle_deg)deg.jld2";
                X = X_spiral,
                A = A,
                D = D,
                sample_name = "yu_spiral",
                angle_deg = angle_deg,
                alpha = alpha,
                sparse_output = SAVE_SPARSE
            )
        end
    else
        println("Skipping Yu spiral set because SPIRAL_X_FILE is nothing.")
    end

    println("All requested available datasets saved.")

end

if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
end