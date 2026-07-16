using Pkg
Pkg.activate(".")
using SLFA
X = [1.0 2.0]
y = [0.0, 0.0]
A = Bool[1 0; 0 1]
D = [0.0 1.0; 1.0 0.0]
try
    SLFA.train_RBFN(X, y, A, D, N_max=1, solver=SLFA.initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity)
catch err
    showerror(stdout, err)
    println()
end
