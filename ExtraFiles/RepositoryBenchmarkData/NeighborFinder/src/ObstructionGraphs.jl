module ObstructionGraphs
using LinearAlgebra
using Random
using Base.Threads
using JLD2
using SparseArrays

include("core.jl")
export gabriel, ellipticGabriel, doubleCone, bow, LoS, ObstructionRule, NbrWorkspace, KNN, EpsilonBall, build_denseGraph_threaded, build_sparseGraph_threaded
end