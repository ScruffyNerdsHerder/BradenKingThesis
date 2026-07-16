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

X = sort(rand(3000)*10, rev=true)
X_validation = sort(rand(1500)*10, rev=true)

y = Benchmark_1D(X,"SinE",0.0)
y_validation_true = Benchmark_1D(X_validation,"SinE",0.0)

## Compare the effectiveness of our method to Yu's method on the 1D SinE problem
omega = 0.0
error_threshold = [0.0, 0.0, 0.0]
print_iter=false
noiseLevel = 0.05
monotonicity=Nonstrict()
y_clean = Benchmark_1D(X,"SinE",0.0)
y_noisy = Benchmark_1D(X,"SinE",noiseLevel)
start_gap = noiseLevel/2*maximum(abs.(y_clean)) # start gap for the monotonicity constraint
solver_LBFGS(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver_LBFGS(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
Theta_YuComp, res_history_YuComp, _, _, _, _, _, _, _ = train_RBFN(X, y, N_max=200, solver=solver_LBFGS, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);

y_validation_fit_200 = RBFN(Theta_YuComp, Gaussian{Isotropic, Float64, 1})
y_validation_fit_20 = RBFN(Theta_YuComp[1:20,:], Gaussian{Isotropic, Float64, 1})

errors_20 = validationErrors(y_validation_fit_20, X_validation, y_validation_true)
errors_200 = validationErrors(y_validation_fit_200, X_validation, y_validation_true)

function validationErrors(network::RBFN, X_validation, y_validation_true)
    y_validation_value = zeros(length(X_validation))
    for i in 1:length(X_validation)
        xval = X_validation[i]
        y_validation_value[i] = network(xval)
    end
    validation_error = y_validation_value .- y_validation_true
    validation_RMSE = sqrt(mean(validation_error.^2))
    validation_bounds = bound_diff(validation_error, y_validation_true, y_validation_value, 200)
    validation_max = maximum(validation_error)
    return validation_RMSE, validation_bounds, validation_max
end

# Error measures
bound_diff(res, res_validation, res_history, N) = maximum(res) - minimum(res);

plot(X_validation, y_validation_true, label="True Function", linewidth=3)
plot!(X_validation, y_validation_value, label="RBFN Fit", linewidth=3)


