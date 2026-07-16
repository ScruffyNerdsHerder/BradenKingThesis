using Pkg
Pkg.activate(".")
using SLFA
using JLD2
X = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "X")
A = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "A")
D = load("ExtraFiles/data/grid_100x100_doublecone_15deg.jld2", "D")
y = [0.0 for _ in 1:size(X,2)]
try
    SLFA.train_RBFN(X, y, A, D, N_max=10, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity)
catch err
    showerror(stdout, err)
    println()
end
