using Pkg
Pkg.activate(".")
include("Benchmarks/Benchmarks.jl")
include("ExtraFiles/ExperimentalSolvers.jl")
using SLFA
using JLD2
X = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "X")
A = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "A")
D = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "D")
y = Benchmark_2D(X, "Step", 0.0)
println(typeof(X), " ", size(X))
println(methods(SLFA.train_RBFN))
