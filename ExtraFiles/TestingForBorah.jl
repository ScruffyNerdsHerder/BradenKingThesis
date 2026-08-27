using SLFA
using Optim
using JLD2
using Base.Threads
print("Hello Cruel World")
i= zeros(Int, 10)
@threads for i in 1:10
    println("Thread $i is working")
    f[i] = i^2
end