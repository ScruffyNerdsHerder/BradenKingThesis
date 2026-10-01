## Review the Abalone Run
        using Pkg
pkg"activate ."
file_path1 = joinpath(@__DIR__, "..", "..", "Benchmarks", "Benchmarks.jl")
include(file_path1)
file_path2 = joinpath(@__DIR__, "..", "ExperimentalSolvers.jl")
include(file_path2)
using SLFA
using Optim
using JLD2
using Base.Threads


@load "ExtraFiles/PitchExperiments/PitchTestDewpoint2.jld2" X_dewpoint y_dewpoint FinalTheta_dewpoint resData_dewpoint       
        @load "ExtraFiles/PitchExperiments/obs_dewpoint_c_Xy.jld2" X y

        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "N",
        ylabel = "L2 Error",
        title = "Abalone Training Data",
        yscale = log10,
        yticks = LogTicks(-2:1),
        titlesize = 20,
        xlabelsize = 20,    
        ylabelsize = 20,
        )
        # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

        Ns = collect(0:1:(length(resData_Abalone)-1))
        GLMakie.scatter!(ax, Ns, resData_Abalone)
        f

        Base.print_matrix(IOContext(stdout, :limit => false), FinalTheta_Abalone)



        @load "ExtraFiles/PitchExperiments/obs_temp_c_graphs.jld2" X_temp y_temp A_temp D_temp