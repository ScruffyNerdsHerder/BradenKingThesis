using Base.Threads

@inline gabriel(R::T) where {T<:Real} = sqrt(max(zero(T), R * (one(T) - R)))

@inline ellipticGabriel(R::T, ratio::T=0.75) where {T<:Real} = ratio * gabriel(R)

@inline bow(R::T, alpha::T=1.0) where {T<:Real} = min(R * alpha, sqrt(max(zero(T), one(T) - R^2)))

@inline LoS(R::T) where {T<:Real} = zero(T)

@inline doubleCone(R::T, alpha_h::T=1.0, alpha_c::T=1.0) where {T<:Real} = min(R * alpha_h, (one(T) - R) * alpha_c)

@inline doubleCone(R::T, alpha::T=1.0) where {T<:Real} = min(R * alpha, (one(T) - R) * alpha)

struct ObstructionRule{F}
    g::F
end

struct EpsilonBall{T<:Real}
    eps2::T
end

EpsilonBall(epsilon::T) where {T<:Real} = EpsilonBall{T}(epsilon^2)

struct KNN
    k::Int
end

mutable struct NbrWorkspace{T<:Real}
    d2::Vector{T}
    perm::Vector{Int}
    nbr_ids::Vector{Int}
    dir::Vector{T}

    function NbrWorkspace(X::AbstractMatrix{T}) where {T<:Real}
        D, N = size(X)

        return new{T}(
            Vector{T}(undef, N),   # one distance per point
            collect(1:N),          # indices 1,2,...,N
            Vector{Int}(undef, N), # worst case: all points accepted
            Vector{T}(undef, D)    # direction vector in ambient dimension D
        )
    end
end

@inline function fill_sqdist!(d2::Vector{T}, X::AbstractMatrix{T}, host::Int) where {T<:Real}
    D, N = size(X)

    @inbounds for i in 1:N
        s = zero(T)

        @simd for k in 1:D
            diff = X[k, i] - X[k, host]
            s = muladd(diff, diff, s)  # s += diff^2, using fused multiply-add when available
        end

        d2[i] = s
    end

    return d2
end

@inline function fill_sqdist!(d2::Vector{T}, X::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    D, N = size(X)

    @inbounds for i in 1:N
        s = zero(T)

        @simd for k in 1:D
            diff = X[k, i] - x[k]
            s = muladd(diff, diff, s)
        end

        d2[i] = s
    end

    return d2
end

@inline function sort_indices!(perm::Vector{Int}, d2::Vector{T}) where {T<:Real}
    sortperm!(perm, d2)
    return perm
end

@inline function _accept_host!(
    rule::ObstructionRule{F},
    X::AbstractMatrix{T},
    host::Int,
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real,F}

    # Do not accept zero-distance points
    d2_cand > zero(T) || return false

    # Actual distance host -> candidate
    R0 = sqrt(d2_cand)
    invR0 = inv(R0)
    D = size(X, 1)

    # dir = unit vector from host to candidate
    @inbounds @simd for k in 1:D
        dir[k] = (X[k, cand] - X[k, host]) * invR0
    end

    # Check all previously accepted neighbors
    @inbounds for t in 1:nnbr
        nbr = nbr_ids[t]

        proj = zero(T)  # projection of nbr onto candidate direction
        v2 = zero(T)  # squared distance host -> nbr

        @simd for k in 1:D
            vk = X[k, nbr] - X[k, host]
            proj = muladd(dir[k], vk, proj)
            v2 = muladd(vk, vk, v2)
        end

        # proj in (0, R0) means neighbor lies somewhere "between"
        # the host and candidate along that direction
        if proj > zero(T) && proj < R0
            R = proj * invR0      # normalize to [0,1]
            gR = rule.g(R)        # obstruction width at normalized location R

            # Allowed squared orthogonal distance from centerline
            thresh2 = d2_cand * gR * gR

            # orth2 = squared perpendicular distance from nbr to the host->cand line
            orth2 = max(zero(T), v2 - proj * proj)

            # If nbr lies inside obstruction tube/cone/etc., reject candidate
            orth2 < thresh2 && return false
        end
    end

    return true
end

@inline function _accept_query!(
    rule::ObstructionRule{F},
    x_host::AbstractVector{T},
    X::AbstractMatrix{T},
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real,F}

    d2_cand > zero(T) || return false

    R0 = sqrt(d2_cand)
    invR0 = inv(R0)
    D = size(X, 1)

    # dir = unit vector from query point to candidate
    @inbounds @simd for k in 1:D
        dir[k] = (X[k, cand] - x_host[k]) * invR0
    end

    @inbounds for t in 1:nnbr
        nbr = nbr_ids[t]

        proj = zero(T)
        v2 = zero(T)

        @simd for k in 1:D
            vk = X[k, nbr] - x_host[k]
            proj = muladd(dir[k], vk, proj)
            v2 = muladd(vk, vk, v2)
        end

        if proj > zero(T) && proj < R0
            R = proj * invR0
            gR = rule.g(R)
            thresh2 = d2_cand * gR * gR
            orth2 = max(zero(T), v2 - proj * proj)

            orth2 < thresh2 && return false
        end
    end

    return true
end

# @inline function _accept_host_all_closer!(
#     rule::ObstructionRule{F},
#     X::AbstractMatrix{T},
#     host::Int,
#     cand::Int,
#     perm::Vector{Int},
#     p::Int,
#     d2_cand::T,
#     dir::Vector{T}
# ) where {T<:Real,F}

#     d2_cand > zero(T) || return false

#     R0 = sqrt(d2_cand)
#     invR0 = inv(R0)
#     D = size(X, 1)

#     # dir = unit vector from host to candidate
#     @inbounds @simd for k in 1:D
#         dir[k] = (X[k, cand] - X[k, host]) * invR0
#     end

#     # Check all closer points, not only accepted neighbors
#     @inbounds for q in 1:p-1
#         nbr = perm[q]
#         nbr == host && continue

#         proj = zero(T)
#         v2 = zero(T)

#         @simd for k in 1:D
#             vk = X[k, nbr] - X[k, host]
#             proj = muladd(dir[k], vk, proj)
#             v2 = muladd(vk, vk, v2)
#         end

#         if proj > zero(T) && proj < R0
#             R = proj * invR0
#             gR = rule.g(R)

#             thresh2 = d2_cand * gR * gR
#             orth2 = max(zero(T), v2 - proj * proj)

#             # Keep strict < if you want boundary points included.
#             # Use <= if boundary should obstruct.
#             orth2 < thresh2 && return false
#         end
#     end

#     return true
# end

function build_denseGraph_threaded(
    X::AbstractMatrix{T},
    rule::Union{ObstructionRule,EpsilonBall,KNN}
) where {T<:Real}

    N = size(X, 2)

    A = falses(N, N)

    D = Matrix{T}(undef, N, N)
    fill!(D, T(Inf))

    # One workspace per possible thread id
    workspaces = [NbrWorkspace(X) for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for host in 1:N
        ws = workspaces[Threads.threadid()]

        # Distances from host to all points
        fill_sqdist!(ws.d2, X, host)

        # Optional but useful: prevent host from appearing as nearest candidate
        ws.d2[host] = T(Inf)

        # Sort candidates by increasing distance
        sort_indices!(ws.perm, ws.d2)

        nnbr = 0

        # Diagonal distance is zero
        D[host, host] = zero(T)

        @inbounds for p in 1:N
            cand = ws.perm[p]
            cand == host && continue

            d2c = ws.d2[cand]

            if _accept_host!(rule, X, host, cand, ws.nbr_ids, nnbr, d2c, ws.dir)
                nnbr += 1
                ws.nbr_ids[nnbr] = cand

                A[cand, host] = true
                D[cand, host] = sqrt(d2c)
            end
        end
    end

    return A, D
end

using SparseArrays

function build_sparseGraph_threaded(
    X::AbstractMatrix{T},
    rule::Union{ObstructionRule,EpsilonBall,KNN},
) where {T<:Real}

    N = size(X, 2)

    workspaces = [NbrWorkspace(X) for _ in 1:Threads.maxthreadid()]
    edges = [Tuple{Int,Int,T}[] for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for host in 1:N
        tid = Threads.threadid()
        ws = workspaces[tid]
        local_edges = edges[tid]

        fill_sqdist!(ws.d2, X, host)
        ws.d2[host] = T(Inf)
        sort_indices!(ws.perm, ws.d2)

        nnbr = 0

        @inbounds for cand in ws.perm
            cand == host && continue

            d2c = ws.d2[cand]

            if _accept_host!(
                rule,
                X,
                host,
                cand,
                ws.nbr_ids,
                nnbr,
                d2c,
                ws.dir,
            )
                nnbr += 1
                ws.nbr_ids[nnbr] = cand

                push!(local_edges, (cand, host, sqrt(d2c)))
            end
        end
    end

    all_edges = reduce(vcat, edges; init=Tuple{Int,Int,T}[])

    rows = first.(all_edges)
    cols = getindex.(all_edges, 2)
    distances = last.(all_edges)

    A = sparse(rows, cols, trues(length(all_edges)), N, N)
    D = sparse(rows, cols, distances, N, N)

    return A, D
end

# function build_sparseGraph_threaded(
#     X::AbstractMatrix{T},
#     rule::Union{ObstructionRule,EpsilonBall,KNN}
# ) where {T<:Real}

#     N = size(X, 2)

#     nthreads = Threads.maxthreadid()

#     rows_local = [Int[] for _ in 1:nthreads]
#     cols_local = [Int[] for _ in 1:nthreads]
#     dist_local = [T[] for _ in 1:nthreads]

#     workspaces = [NbrWorkspace(X) for _ in 1:nthreads]

#     Threads.@threads :static for host in 1:N
#         tid = Threads.threadid()
#         ws = workspaces[tid]

#         rows = rows_local[tid]
#         cols = cols_local[tid]
#         dvals = dist_local[tid]

#         fill_sqdist!(ws.d2, X, host)
#         ws.d2[host] = T(Inf)

#         sort_indices!(ws.perm, ws.d2)

#         nnbr = 0

#         # @inbounds for p in 1:N
#         #     cand = ws.perm[p]
#         #     cand == host && continue

#         #     d2c = ws.d2[cand]

#         #     if _accept_host!(rule, X, host, cand, ws.nbr_ids, nnbr, d2c, ws.dir)
#         #         nnbr += 1
#         #         ws.nbr_ids[nnbr] = cand

#         #         push!(rows, cand)
#         #         push!(cols, host)
#         #         push!(dvals, sqrt(d2c))
#         #     end
#         # end

#         @inbounds for p in 1:N
#             cand = ws.perm[p]
#             cand == host && continue
        
#             d2c = ws.d2[cand]
        
#             if _accept_host_all_closer!(rule, X, host, cand, ws.perm, p, d2c, ws.dir)
#                 push!(rows, cand)
#                 push!(cols, host)
#                 push!(dvals, sqrt(d2c))
#             end
#         end
#     end

#     rows = reduce(vcat, rows_local)
#     cols = reduce(vcat, cols_local)
#     dvals = reduce(vcat, dist_local)

#     A = sparse(rows, cols, trues(length(rows)), N, N, |)
#     D = sparse(rows, cols, dvals, N, N, min)

#     return A, D
# end