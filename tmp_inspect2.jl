using JLD2
using SLFA
X = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "X")
A = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "A")
D = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "D")
println(typeof(X))
println(typeof(A), " ", size(A))
println(typeof(D), " ", size(D))
println(eltype(A), " ", eltype(D))
println(methods(SLFA.train_RBFN))
