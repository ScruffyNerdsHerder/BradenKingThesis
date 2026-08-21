using Pkg
pkg"activate ."

file_path1 = joinpath(@__DIR__, "..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
include("ExperimentalSolvers.jl")
using BenchmarkTools
# using Plots
using SLFA
using Optim
# using Measures
# using JLD2
# using PrettyTables
# using GLMakie
using Base.Threads
using CSV


## Abalone Data importing
abalone_data = CSV.input("ExtraFiles/RepositoryBenchmarkData/abalone.data", header=false, delim=' ', types=Dict(1=>Float64, 2=>Float64, 3=>Float64, 4=>Float64, 5=>Float64, 6=>Float64, 7=>Float64, 8=>Float64))