# Set of 1D benchmarks that can be chosen from 
module Benchmarks

using LinearAlgebra
using Measures
using Random
using Optim
export Benchmark_1D

function Benchmark_1D(X,benchmark,noiseLevel=0.0)
    if benchmark == "AsymmetricRBF"
        # Make left (longer) RBF half
        w1 = 1/(0.1*(X[end]-X[1]))
        c1 = .75*(X[end]-X[1])
        a1 = 3
        b1 = 0
        # make right (smaller) RBF half
        w2 = 9*(X[end]-X[1])
        c2 = c1
        a2 = a1
        b2 = b1
        y = similar(X)

        # build new y
        y[X.<= c1] .= a1 .* exp.(-((X[X .<= c1] .- c1) .* w1).^2) .+ b1
        y[X .>  c1] .= a2 .* exp.(-((X[X .>  c1] .- c1) .* w2).^2) .+ b2
    elseif benchmark == "Sine"
        y = 2 .* sin.(2pi .* X) .+ 1
    elseif benchmark == "Step"
        periods = .25*(X[end]-X[1])
        y = zero(X)
        y[mod.(X,periods) .<= 1] .= 1
    elseif benchmark == "SinE"
        y = 0.8*exp.(-0.2*X) .* sin.(10*X)
    elseif benchmark == "CompositeSine"
        funcs = 30
        rand_Params = Random.randn(Xoshiro(4414),4,funcs);
        y = zero(X);
        for i = 1:funcs
            amplitude = rand_Params[1,i]
            phase = rand_Params[2,i]
            freq = rand_Params[3,i]
            bias = rand_Params[4,i]
            y .= y .+ amplitude .* sin.(1.5 .* pi .* freq .* X .+ phase) .+ bias;
        end
    else # Didnt put in a correct benchmark
        throw("Put in a correct benchmark please: AsymnmetricRBF, Sine, Step, SinE, or CompositeSine")
    end
    rng = Xoshiro(23)
    noise = 2 .* randn(rng,size(y)) .- 1 # from -1:1
    y_noisy = noiseLevel .* noise .* y .+ y
    return y_noisy
end

end # Module Benchmarks