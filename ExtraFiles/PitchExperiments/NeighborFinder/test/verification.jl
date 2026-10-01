using JLD2
using SparseArrays
using LinearAlgebra
using Printf

const OUTDIR = joinpath(@__DIR__, "..", "data")

@inline grid_index(ix::Int, iy::Int, grid_n::Int) = ix + (iy - 1) * grid_n

function count_grid_diagonal_edges(A, grid_n::Int)
    count_diag = 0

    for iy in 1:grid_n
        for ix in 1:grid_n
            host = grid_index(ix, iy, grid_n)

            for (dx, dy) in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                jx = ix + dx
                jy = iy + dy

                if 1 <= jx <= grid_n && 1 <= jy <= grid_n
                    cand = grid_index(jx, jy, grid_n)

                    if A[cand, host]
                        count_diag += 1
                    end
                end
            end
        end
    end

    return count_diag
end

function infer_angle_from_filename(path::AbstractString)
    m = match(r"_(\d+)deg\.jld2$", basename(path))
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function infer_sample_type(path::AbstractString)
    fname = lowercase(basename(path))

    if occursin("grid", fname)
        return "grid"
    elseif occursin("uniform", fname)
        return "uniform"
    elseif occursin("normal", fname)
        return "normal"
    elseif occursin("spiral", fname)
        return "spiral"
    else
        return "unknown"
    end
end

function expected_neighbor_upper(sample_type::String, angle::Union{Int,Nothing})
    angle === nothing && return nothing

    if sample_type == "grid"
        if angle == 45 || angle == 46
            return 4
        elseif angle == 30
            return 12
        elseif angle == 15
            return 24
        end
    elseif sample_type == "uniform" || sample_type == "normal"
        if angle == 45 || angle == 46
            return 7
        elseif angle == 30
            return 11
        elseif angle == 15
            return 23
        end
    end

    return nothing
end

# function check_sparse_patterns(A, D)
#     if issparse(A) && issparse(D)
#         # For your convention, D should be stored wherever A is true.
#         # Since D stores positive distances, its nonzero pattern should match A.
#         A_pattern = sparse(rowvals(A), repeat(1:size(A, 2), diff(A.colptr)), trues(nnz(A)), size(A, 1), size(A, 2))
#         D_pattern = sparse(rowvals(D), repeat(1:size(D, 2), diff(D.colptr)), trues(nnz(D)), size(D, 1), size(D, 2))

#         return A_pattern == D_pattern
#     else
#         return nothing
#     end
# end

function check_sparse_patterns(A, D)
    if issparse(A) && issparse(D)
        A_clean = dropzeros(copy(A))
        D_clean = dropzeros(copy(D))

        IA, JA, _ = findnz(A_clean)
        ID, JD, _ = findnz(D_clean)

        return IA == ID && JA == JD
    else
        return nothing
    end
end

function verify_graph_file(path::AbstractString; grid_n::Int=100)
    data = load(path)

    required_keys = ["X", "A", "D"]

    if !all(k -> haskey(data, k), required_keys)
        println("============================================================")
        println("Skipping file: ", path)
        println("Reason: file does not contain X, A, and D.")
        println("Available keys: ", collect(keys(data)))
        println()
        return nothing
    end

    X = data["X"]
    A = data["A"]
    D = data["D"]

    sample_type = infer_sample_type(path)
    angle = infer_angle_from_filename(path)
    upper = expected_neighbor_upper(sample_type, angle)

    println("============================================================")
    println("File: ", path)
    println("sample type = ", sample_type)
    println("angle       = ", angle)
    println()

    println("Types:")
    println("  typeof(X) = ", typeof(X))
    println("  typeof(A) = ", typeof(A))
    println("  typeof(D) = ", typeof(D))
    println("  eltype(X) = ", eltype(X))
    println("  eltype(A) = ", eltype(A))
    println("  eltype(D) = ", eltype(D))
    println()

    println("Sizes:")
    println("  size(X) = ", size(X))
    println("  size(A) = ", size(A))
    println("  size(D) = ", size(D))
    println()

    println("Sparse checks:")
    println("  issparse(A) = ", issparse(A))
    println("  issparse(D) = ", issparse(D))

    if !issparse(A)
        println("  WARNING: A is dense. Your advisor expects sparse Bool.")
    end

    if !issparse(D)
        println("  WARNING: D is dense. Your advisor expects sparse Float64.")
    end

    if eltype(X) != Float64
        println("  WARNING: X is not Float64.")
    end

    if eltype(A) != Bool
        println("  WARNING: A is not Bool.")
    end

    if eltype(D) != Float64
        println("  WARNING: D is not Float64.")
    end

    println()

    degrees = vec(sum(A; dims=1))

    min_deg = minimum(degrees)
    max_deg = maximum(degrees)
    mean_deg = sum(degrees) / length(degrees)

    println("Neighbor counts using column sums:")
    println("  min neighbors  = ", min_deg)
    println("  max neighbors  = ", max_deg)
    @printf("  mean neighbors = %.4f\n", mean_deg)

    if upper !== nothing
        println("  advisor approximate upper = ", upper)

        if max_deg <= upper
            println("  neighbor-count check: OK")
        else
            println("  neighbor-count check: WARNING, max is larger than advisor expected")
        end
    end

    println()

    println("Nonzeros and file size:")
    println("  nnz(A) = ", issparse(A) ? nnz(A) : count(A))
    println("  nnz(D) = ", issparse(D) ? nnz(D) : count(!iszero, D))
    @printf("  file size MB = %.4f\n", filesize(path) / 1024^2)
    println()

    if issparse(A) && issparse(D)
        same_pattern = check_sparse_patterns(A, D)
        println("Sparse pattern check:")
        println("  nonzero pattern of A matches D? ", same_pattern)

        if same_pattern == false
            println("  WARNING: A and D do not store entries in the same locations.")
        end
        println()
    end

    println("Point/domain checks:")
    println("  X is 2 × N? ", size(X, 1) == 2)
    println("  all X in [-1,1]? ", all((-1 .<= X) .& (X .<= 1)))
    println()

    if sample_type == "grid"
        diag_edges = count_grid_diagonal_edges(A, grid_n)
        println("Grid diagonal-neighbor check:")
        println("  diagonal edges among immediate grid diagonals = ", diag_edges)

        if angle == 45 && diag_edges > 0
            println("  WARNING: 45 degrees produced diagonal grid edges. Advisor suggested using 46 degrees if this happens.")
        elseif angle == 46 && diag_edges == 0
            println("  diagonal check: OK")
        end

        println()
    end

    return nothing
end

function main()
    println("Verifying files in: ", OUTDIR)
    println()

    files = filter(
        f -> endswith(f, ".jld2") && occursin("doublecone", basename(f)),
        readdir(OUTDIR; join=true)
    )
    sort!(files)

    if isempty(files)
        println("No .jld2 files found in ", OUTDIR)
        return
    end

    for file in files
        verify_graph_file(file; grid_n=100)
    end

    println("Done verifying all files.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end