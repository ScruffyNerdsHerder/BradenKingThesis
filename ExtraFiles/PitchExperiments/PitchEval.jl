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
using GLMakie
using CairoMakie
using Base.Threads 
using Printf
using Statistics

## ADD L-INF (global and training) TO ALL OF THIS
## RMSES for the Temperature

        function findTempRMSEs(DataFile::String)
                @load DataFile X_temp y_temp FinalTheta_temp resData_temp
                @load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Temperature Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_temp)-1))
                GLMakie.scatter!(ax, Ns, resData_temp)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                tempRBFN = RBFN(FinalTheta_temp[1:2,:], T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_temp_nans = findall(!isnan, y)
                X_temp_full = X_flip[:, y_temp_nans]
                y_temp_full = y[y_temp_nans]
                y_pred = tempRBFN(X_temp) 
                y_diff = y_pred - y_temp_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_temp[end])
                # println("Initial RMSE: ", resData_temp[1])
                return f, y_diff_norm, resData_temp[end], resData_temp[1]
        end

        function findTempRMSEsRound(DataFile::String)
                @load DataFile X_temp y_temp FinalTheta_temp resData_temp
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                @load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                X_flip = Float64.(permutedims(X,(2,1)))
                X_temp_full = X_flip[:, :]
                y_temp_full = y
                y_rnd_norm = zeros(length(resData_temp))
                y_norm = zeros(length(resData_temp))
                y_rnd_norm[1] = sqrt(norm(y_temp_full)^2/length(y_temp_full))
                y_norm[1] = sqrt(norm(y_temp_full)^2/length(y_temp_full))   
                res = y_temp_full
                res_evalphi = res
                l_inf = zeros(length(resData_temp))
                l_inf[1] = norm(res_evalphi, Inf)

                for i = 1:100
                        # rbf_now = Gaussian{Anisotropic{Aligned}, Float64, 17}(FinalTheta_temp[i,:])
                        # res_rbfmethod = res_rbfmethod - rbf_now(X_temp_full)
                        y_temp_pred = FinalTheta_temp[i,end-1] .* SLFA.eval_phi(X_temp_full, FinalTheta_temp[i,:],T2) .+ FinalTheta_temp[i,end]
                        res_evalphi = res_evalphi - y_temp_pred
                        y_temp_round = round.(res_evalphi)
                        y_temp_now = res_evalphi
                        # y_temp_round = (res_evalphi)
                        y_rnd_norm[i+1] = sqrt(norm(y_temp_round)^2/length(y_temp_full))
                        y_norm[i+1] = sqrt(norm(y_temp_now)^2/length(y_temp_full))
                        l_inf[i+1] = norm(res_evalphi, Inf)
                end

                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Temperature Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(y_rnd_norm)-1))
                GLMakie.scatter!(ax, Ns, y_rnd_norm)

                return f, y_rnd_norm, y_norm, resData_temp[end], resData_temp[1], l_inf, res_evalphi
        end
        f1_t, y_rnd_norm1_t, y_norm1_t, resData_temp_end1, resData_temp_1, l_inf1_t, res_evalphi1_t = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTestTemp0.1.jld2");
        f2_t, y_rnd_norm2_t, y_norm2_t, resData_temp_end2, resData_temp_2, l_inf2_t, res_evalphi2_t = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTestTemp0.2.jld2");
        f3_t, y_rnd_norm3_t, y_norm3_t, resData_temp_end3, resData_temp_3, l_inf3_t, res_evalphi3_t = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTestTemp0.3.jld2");
        f4_t, y_rnd_norm4_t, y_norm4_t, resData_temp_end4, resData_temp_4, l_inf4_t, res_evalphi4_t = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTestTemp0.4.jld2");
        println("Average Training Data RMSE: ", mean([resData_temp_end1, resData_temp_end2, resData_temp_end3, resData_temp_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_temp_end1, resData_temp_end2, resData_temp_end3, resData_temp_end4]))
        println("Min and Max RMSEs for the Non-Rounded Full Dataset: ", minimum([y_norm1_t[end], y_norm2_t[end], y_norm3_t[end], y_norm4_t[end]]), " and ", maximum([y_norm1_t[end], y_norm2_t[end], y_norm3_t[end], y_norm4_t[end]]))
        println("Average Full Non-Rounded Dataset RMSE: ", mean([y_norm1_t[end], y_norm2_t[end], y_norm3_t[end], y_norm4_t[end]]))
        println("Min and Max RMSEs for the Rounded Full Dataset: ", minimum([y_rnd_norm1_t[end], y_rnd_norm2_t[end], y_rnd_norm3_t[end], y_rnd_norm4_t[end]]), " and ", maximum([y_rnd_norm1_t[end], y_rnd_norm2_t[end], y_rnd_norm3_t[end], y_rnd_norm4_t[end]]))
        println("Average Rounded Full Dataset RMSE: ", mean([y_rnd_norm1_t[end], y_rnd_norm2_t[end], y_rnd_norm3_t[end], y_rnd_norm4_t[end]]))
        println("StD of Full Dataset RMSE: ", std([y_rnd_norm1_t[end], y_rnd_norm2_t[end], y_rnd_norm3_t[end], y_rnd_norm4_t[end]]))
        println("Initial RMSE: ", resData_temp_1)

        ## Run the temp evals for the smaller sample counts
        f5_t, y_rnd_norm5_t, y_norm5_t, resData_temp_end5, resData_temp_5 = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.500.jld2");
        f6_t, y_rnd_norm6_t, y_norm6_t, resData_temp_end6, resData_temp_6 = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.1000.jld2");
        f7_t, y_rnd_norm7_t, y_norm7_t, resData_temp_end7, resData_temp_7 = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.2000.jld2");
        f8_t, y_rnd_norm8_t, y_norm8_t, resData_temp_end8, resData_temp_8 = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.5000.jld2");


        println("Full Data Results/RMSEs for the Temperature sample counts", y_norm1_t[end], " ", y_norm2_t[end], " ", y_norm3_t[end], " ", y_norm4_t[end], " ", y_norm5_t[end], " ", y_norm6_t[end], " ", y_norm7_t[end], " ", y_norm8_t[end])
        println("Full Data Results/RMSEs for the Rounded Temperature sample counts", y_rnd_norm1_t[end], " ", y_rnd_norm2_t[end], " ", y_rnd_norm3_t[end], " ", y_rnd_norm4_t[end], " ", y_rnd_norm5_t[end], " ", y_rnd_norm6_t[end], " ", y_rnd_norm7_t[end], " ", y_rnd_norm8_t[end])

        println("Results/RMSEs for the Temperature sample counts")
        println("RMSE for 500 samples: ", y_rnd_norm1_t[end])
        println("RMSE for 1000 samples: ", y_rnd_norm2_t[end])
        println("RMSE for 2000 samples: ", y_rnd_norm3_t[end])
        println("RMSE for 5000 samples: ", y_rnd_norm4_t[end])
        println("RMSEs for 10000 samples: ", y_rnd_norm5_t[end])

                ft1_t, y_rnd_norm8_t, y_norm8_t, resData_temp_end8, resData_temp_8 = findTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTempSampleCount.5000.jld2");

                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training l-inf",
                title = "Pitch Temperature Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(l_inf1_t)-1))
                GLMakie.scatter!(ax, Ns, l_inf1_t)
## RMSES for the pressure
        function findPressureRMSEs(DataFile::String)
                @load DataFile X_temp y_pressure FinalTheta_pressure resData_pressure
                @load "ExtraFiles/PitchExperiments/obs_station_pressure_pa_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Pressure Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_pressure)-1))
                GLMakie.scatter!(ax, Ns, resData_pressure)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 8}
                pressureRBFN = RBFN(FinalTheta_pressure, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_pressure_nans = findall(!isnan, y)
                X_pressure_full = X_flip[:, y_pressure_nans]
                y_pressure_full = y[y_pressure_nans]
                y_pred = pressureRBFN(X_pressure_full) 
                y_diff = y_pred - y_pressure_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_pressure[end])
                # println("Initial RMSE: ", resData_pressure[1])
                return f, y_diff_norm, resData_pressure[end], resData_pressure[1]
        end
        f1_p, y_diff_norm1_p, resData_pressure_end1, resData_pressure_1 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPressure0.1.jld2");
        f2_p, y_diff_norm2_p, resData_pressure_end2, resData_pressure_2 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPressure0.2.jld2");
        f3_p, y_diff_norm3_p, resData_pressure_end3, resData_pressure_3 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPressure0.3.jld2");
        f4_p, y_diff_norm4_p, resData_pressure_end4, resData_pressure_4 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPressure0.4.jld2");
        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_p, y_diff_norm2_p, y_diff_norm3_p, y_diff_norm4_p]), " and ", maximum([y_diff_norm1_p, y_diff_norm2_p, y_diff_norm3_p, y_diff_norm4_p]))
        println("Average Training Data RMSE: ", mean([resData_pressure_end1, resData_pressure_end2, resData_pressure_end3, resData_pressure_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_pressure_end1, resData_pressure_end2, resData_pressure_end3, resData_pressure_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_p, y_diff_norm2_p, y_diff_norm3_p, y_diff_norm4_p]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_p, y_diff_norm2_p, y_diff_norm3_p, y_diff_norm4_p]))
        println("Initial RMSE: ", resData_pressure_1)

        ## Pressure RMSES for smaller sample counts    
        f5_p, y_diff_norm5_p, resData_pressure_end5, resData_pressure_5 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchPressureSampleCount.500.jld2");
        f6_p, y_diff_norm6_p, resData_pressure_end6, resData_pressure_6 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchPressureSampleCount.1000.jld2");
        f7_p, y_diff_norm7_p, resData_pressure_end7, resData_pressure_7 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchPressureSampleCount.2000.jld2");
        f8_p, y_diff_norm8_p, resData_pressure_end8, resData_pressure_8 = findPressureRMSEs("ExtraFiles/PitchExperiments/Results/PitchPressureSampleCount.5000.jld2");
        
        println("Full Data Results/RMSEs for the Pressure sample counts")
        println("Initial RMSE: ", resData_pressure_1)
        println("RMSE for 500 samples: ", y_diff_norm5_p)
        println("RMSE for 1000 samples: ", y_diff_norm6_p)
        println("RMSE for 2000 samples: ", y_diff_norm7_p)
        println("RMSE for 5000 samples: ", y_diff_norm8_p)
        println("RMSE for 10000 samples: ", y_diff_norm1_p)

        println("Full Data Results/RMSEs for the Pressure sample counts", y_diff_norm1_p, " ", y_diff_norm2_p, " ", y_diff_norm3_p, " ", y_diff_norm4_p, " ", y_diff_norm5_p, " ", y_diff_norm6_p, " ", y_diff_norm7_p, " ", y_diff_norm8_p)

## RMSES for the Dewpoints
        function findDewpointRMSEs(DataFile::String)
                @load DataFile X_dewpoint y_dewpoint FinalTheta_dewpoint resData_dewpoint
                @load "ExtraFiles/PitchExperiments/obs_dewpoint_c_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Dewpoint Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_dewpoint)-1))
                GLMakie.scatter!(ax, Ns, resData_dewpoint)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                dewpointRBFN = RBFN(FinalTheta_dewpoint, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_dewpoint_nans = findall(!isnan, y)
                X_dewpoint_full = X_flip[:, y_dewpoint_nans]
                y_dewpoint_full = y[y_dewpoint_nans]
                y_pred = dewpointRBFN(X_dewpoint_full) 
                y_diff = y_pred - y_dewpoint_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                res = y_dewpoint_full
                for i = 1:500
                        y_dewpoint_pred = FinalTheta_dewpoint[i,end-1] .* SLFA.eval_phi(X_dewpoint_full, FinalTheta_dewpoint[i,:],T2) .+ FinalTheta_dewpoint[i,end]
                end
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_dewpoint[end])
                # println("Initial RMSE: ", resData_dewpoint[1])
                return f, y_diff_norm, resData_dewpoint[end], resData_dewpoint[1]
        end
        f1_d, y_diff_norm1_d, resData_dewpoint_end1, resData_dewpoint_1 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestDewpoint0.1.jld2");
        f2_d, y_diff_norm2_d, resData_dewpoint_end2, resData_dewpoint_2 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestDewpoint0.2.jld2");
        f3_d, y_diff_norm3_d, resData_dewpoint_end3, resData_dewpoint_3 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestDewpoint0.3.jld2");
        f4_d, y_diff_norm4_d, resData_dewpoint_end4, resData_dewpoint_4 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestDewpoint0.4.jld2");
        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_d, y_diff_norm2_d, y_diff_norm3_d, y_diff_norm4_d]), " and ", maximum([y_diff_norm1_d, y_diff_norm2_d, y_diff_norm3_d, y_diff_norm4_d]))
        println("Average Training Data RMSE: ", mean([resData_dewpoint_end1, resData_dewpoint_end2, resData_dewpoint_end3, resData_dewpoint_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_dewpoint_end1, resData_dewpoint_end2, resData_dewpoint_end3, resData_dewpoint_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_d, y_diff_norm2_d, y_diff_norm3_d, y_diff_norm4_d]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_d, y_diff_norm2_d, y_diff_norm3_d, y_diff_norm4_d]))
        println("Initial RMSE: ", resData_dewpoint_1)

        f5_d, y_diff_norm5_d, resData_dewpoint_end5, resData_dewpoint_5 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchDewpointSampleCount.500.jld2");
        f6_d, y_diff_norm6_d, resData_dewpoint_end6, resData_dewpoint_6 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchDewpointSampleCount.1000.jld2");
        f7_d, y_diff_norm7_d, resData_dewpoint_end7, resData_dewpoint_7 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchDewpointSampleCount.2000.jld2");
        f8_d, y_diff_norm8_d, resData_dewpoint_end8, resData_dewpoint_8 = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchDewpointSampleCount.5000.jld2");

        f8_tv_d, y_diff_norm8_tv_d, resData_dewpoint_end8_tv_d, resData_dewpoint_8_tv_d = findDewpointRMSEs("ExtraFiles/PitchExperiments/Results/PitchdewpointTVOn5000.5000.jld2");

        println("Full Data Results/RMSEs for the Dewpoint sample counts", y_diff_norm1_d, " ", y_diff_norm2_d, " ", y_diff_norm3_d, " ", y_diff_norm4_d, " ", y_diff_norm5_d, " ", y_diff_norm6_d, " ", y_diff_norm7_d, " ", y_diff_norm8_d)

        println("Full Data Results/RMSEs for the Dewpoint sample counts")
        println("Initial RMSE: ", resData_dewpoint_1)
        println("RMSE for 500 samples: ", y_diff_norm5_d)
        println("RMSE for 1000 samples: ", y_diff_norm6_d)
        println("RMSE for 2000 samples: ", y_diff_norm7_d)
        println("RMSE for 5000 samples: ", y_diff_norm8_d)
        println("RMSE for 10000 samples: ", y_diff_norm1_d)
## Get final L-infs for Tv and no-tv 5000
        
        T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
        @load "ExtraFiles/PitchExperiments/obs_dewpoint_c_Xy.jld2" X y
        @load "ExtraFiles/PitchExperiments/Results/PitchdewpointTVOn5000.5000.jld2" X_dewpoint y_dewpoint FinalTheta_dewpoint resData_dewpoint
        dewpointRBFN = RBFN(FinalTheta_dewpoint, T2)
        X_flip = Float64.(permutedims(X,(2,1)))
        y_dewpoint_nans = findall(!isnan, y)
        X_dewpoint_full = X_flip[:, y_dewpoint_nans]
        y_dewpoint_full = y[y_dewpoint_nans]
        y_pred = dewpointRBFN(X_dewpoint_full)   
        y_diff = y_pred - y_dewpoint_full
        l_inf1_tv = norm(y_diff, Inf)
        println("Final L-inf for 5000 samples with TV: ", l_inf1_tv)

        @load "ExtraFiles/PitchExperiments/Results/PitchDewpointSampleCount.5000.jld2" X_dewpoint y_dewpoint FinalTheta_dewpoint resData_dewpoint
        dewpointRBFN = RBFN(FinalTheta_dewpoint, T2)
        y_pred = dewpointRBFN(X_dewpoint_full)   
        y_diff = y_pred - y_dewpoint_full
        l_inf = norm(y_diff, Inf)
        println("Final L-inf for 5000 samples without TV: ", l_inf)
        println("Initial l-inf for 5000 samples without TV: ", norm(y_dewpoint_full, Inf))
## RMSES for the precipitation stuff
        function findPrecipitationRMSEs(DataFile::String)
                @load DataFile X_precip y_precip FinalTheta_precip resData_precip
                @load "ExtraFiles/PitchExperiments/obs_precip_mm_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Precipitation Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_precip)-1))
                GLMakie.scatter!(ax, Ns, resData_precip)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 20}
                precipRBFN = RBFN(FinalTheta_precip, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_precip_nans = findall(!isnan, y)
                X_precip_full = X_flip[:, y_precip_nans]
                y_precip_full = y[y_precip_nans]
                y_pred = precipRBFN(X_precip_full) 
                y_diff = y_pred - y_precip_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_precip[end])
                # println("Initial RMSE: ", resData_precip[1])
                return f, y_diff_norm, resData_precip[end], resData_precip[1]
        end
        f1_precip, y_diff_norm1_precip, resData_precip_end1, resData_precip_1 = findPrecipitationRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestprecip0.jld2");
        f2_precip, y_diff_norm2_precip, resData_precip_end2, resData_precip_2 = findPrecipitationRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPrecip0.2.jld2");
        f3_precip, y_diff_norm3_precip, resData_precip_end3, resData_precip_3 = findPrecipitationRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPrecip0.3.jld2");
        f4_precip, y_diff_norm4_precip, resData_precip_end4, resData_precip_4 = findPrecipitationRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestPrecip0.4.jld2");
        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_precip, y_diff_norm2_precip, y_diff_norm3_precip, y_diff_norm4_precip]), " and ", maximum([y_diff_norm1_precip, y_diff_norm2_precip, y_diff_norm3_precip, y_diff_norm4_precip]))
        println("Average Training Data RMSE: ", mean([resData_precip_end1, resData_precip_end2, resData_precip_end3, resData_precip_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_precip_end1, resData_precip_end2, resData_precip_end3, resData_precip_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_precip, y_diff_norm2_precip, y_diff_norm3_precip, y_diff_norm4_precip]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_precip, y_diff_norm2_precip, y_diff_norm3_precip, y_diff_norm4_precip]))
        println("Initial RMSE: ", resData_precip_3) 

## Display RMSEs for wind direction stuff
        function findWindDirectionRMSEs(DataFile::String)
                @load DataFile X_WindDir y_WindDir FinalTheta_WindDir resData_WindDir
                @load "ExtraFiles/PitchExperiments/wind_direction_deg_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Wind Direction Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_WindDir)-1))
                GLMakie.scatter!(ax, Ns, resData_WindDir)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                dirRBFN = RBFN(FinalTheta_WindDir, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_dir_nans = findall(!isnan, y)
                X_dir_full = X_flip[:, y_dir_nans]
                y_dir_full = y[y_dir_nans]
                y_pred = dirRBFN(X_dir_full) 
                y_diff = y_pred - y_dir_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_WindDir[end])
                # println("Initial RMSE: ", resData_WindDir[1])
                return f, y_diff_norm, resData_WindDir[end], resData_WindDir[1]
        end

        f1_dir, y_diff_norm1_dir, resData_WindDir_end1, resData_WindDir_1 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindDir0.1.jld2");
        f2_dir, y_diff_norm2_dir, resData_WindDir_end2, resData_WindDir_2 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindDir0.2.jld2");
        f3_dir, y_diff_norm3_dir, resData_WindDir_end3, resData_WindDir_3 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindDir0.3.jld2");
        f4_dir, y_diff_norm4_dir, resData_WindDir_end4, resData_WindDir_4 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindDir0.4.jld2");
        f5_dir, y_diff_norm5_dir, resData_WindDir_end5, resData_WindDir_5 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchdirSampleCount.500.jld2");
        f6_dir, y_diff_norm6_dir, resData_WindDir_end6, resData_WindDir_6 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchdirSampleCount.1000.jld2");
        f7_dir, y_diff_norm7_dir, resData_WindDir_end7, resData_WindDir_7 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchdirSampleCount.2000.jld2");
        f8_dir, y_diff_norm8_dir, resData_WindDir_end8, resData_WindDir_8 = findWindDirectionRMSEs("ExtraFiles/PitchExperiments/Results/PitchdirSampleCount.5000.jld2");
        
        ## Display RMSEs for wind speed stuff
        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_dir, y_diff_norm2_dir, y_diff_norm3_dir, y_diff_norm4_dir]), " and ", maximum([y_diff_norm1_dir, y_diff_norm2_dir, y_diff_norm3_dir, y_diff_norm4_dir]))
        println("Average Training Data RMSE: ", mean([resData_WindDir_end1, resData_WindDir_end2, resData_WindDir_end3, resData_WindDir_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_WindDir_end1, resData_WindDir_end2, resData_WindDir_end3, resData_WindDir_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_dir, y_diff_norm2_dir, y_diff_norm3_dir, y_diff_norm4_dir]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_dir, y_diff_norm2_dir, y_diff_norm3_dir, y_diff_norm4_dir]))
        println("Initial RMSE: ", resData_WindDir_3)

        println("Full Data Results/RMSEs for the Temperature sample counts", resData_WindDir_1, " ", y_norm1_t[end], " ", y_norm2_t[end], " ", y_norm3_t[end], " ", y_norm4_t[end], " ", y_norm5_t[end], " ", y_norm6_t[end], " ", y_norm7_t[end], " ", y_norm8_t[end])

## RMSES for the wind speed
        function findWindSpeedRMSEs(DataFile::String)
                @load DataFile X_WindSpeed y_WindSpeed FinalTheta_WindSpeed resData_WindSpeed
                @load "ExtraFiles/PitchExperiments/wind_speed_ms_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Wind Speed Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_WindSpeed)-1))
                GLMakie.scatter!(ax, Ns, resData_WindSpeed)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                speedRBFN = RBFN(FinalTheta_WindSpeed, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_WindSpeed_nans = findall(!isnan, y)
                X_WindSpeed_full = X_flip[:, y_WindSpeed_nans]
                y_WindSpeed_full = y[y_WindSpeed_nans]
                y_pred = speedRBFN(X_WindSpeed_full) 
                y_diff = y_pred - y_WindSpeed_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_WindSpeed[end])
                # println("Initial RMSE: ", resData_WindSpeed[1])
                return f, y_diff_norm, resData_WindSpeed[end], resData_WindSpeed[1]
        end
        f1_WindSpeed, y_diff_norm1_WindSpeed, resData_WindSpeed_end1, resData_WindSpeed_1 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindSpeed0.1.jld2");
        f2_WindSpeed, y_diff_norm2_WindSpeed, resData_WindSpeed_end2, resData_WindSpeed_2 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestWindSpeed0.2.jld2");
        
        f5_WindSpeed, y_diff_norm5_WindSpeed, resData_WindSpeed_end5, resData_WindSpeed_5 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.500.jld2");
        f6_WindSpeed, y_diff_norm6_WindSpeed, resData_WindSpeed_end6, resData_WindSpeed_6 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.1000.jld2");
        f7_WindSpeed, y_diff_norm7_WindSpeed, resData_WindSpeed_end7, resData_WindSpeed_7 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.2000.jld2");
        f8_WindSpeed, y_diff_norm8_WindSpeed, resData_WindSpeed_end8, resData_WindSpeed_8 = findWindSpeedRMSEs("ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.5000.jld2");
        
        # @load "ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.5000.jld2" X_speed y_speed FinalTheta_speed resData_speed
        # X_WindSpeed = X_speed
        # y_WindSpeed = y_speed
        # FinalTheta_WindSpeed = FinalTheta_speed
        # resData_WindSpeed = resData_speed
        # @save "ExtraFiles/PitchExperiments/Results/PitchspeedSampleCount.5000.jld2" X_WindSpeed y_WindSpeed FinalTheta_WindSpeed resData_WindSpeed

        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_WindSpeed, y_diff_norm2_WindSpeed, y_diff_norm3_WindSpeed, y_diff_norm4_WindSpeed]), " and ", maximum([y_diff_norm1_WindSpeed, y_diff_norm2_WindSpeed, y_diff_norm3_WindSpeed, y_diff_norm4_WindSpeed]))
        println("Average Training Data RMSE: ", mean([resData_WindSpeed_end1, resData_WindSpeed_end2, resData_WindSpeed_end3, resData_WindSpeed_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_WindSpeed_end1, resData_WindSpeed_end2, resData_WindSpeed_end3, resData_WindSpeed_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_WindSpeed, y_diff_norm2_WindSpeed, y_diff_norm3_WindSpeed, y_diff_norm4_WindSpeed]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_WindSpeed, y_diff_norm2_WindSpeed, y_diff_norm3_WindSpeed, y_diff_norm4_WindSpeed]))
        println("Initial RMSE: ", resData_WindSpeed_5)
 println("Full Data Results/RMSEs for the Pressure sample counts", y_diff_norm1_WindSpeed, " ", y_diff_norm2_WindSpeed, " ", y_diff_norm3_WindSpeed, " ", y_diff_norm4_WindSpeed, " ", y_diff_norm5_WindSpeed)

## RMSES for the RH
        function findRHRMSEs(DataFile::String)
                @load DataFile X_RH y_RH FinalTheta_RH resData_RH
                @load "ExtraFiles/PitchExperiments/obs_rh_percent_Xy.jld2" X y
                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch RH Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(resData_RH)-1))
                GLMakie.scatter!(ax, Ns, resData_RH)
                # f
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 17}
                RHRBFN = RBFN(FinalTheta_RH, T2)
                X_flip = Float64.(permutedims(X,(2,1)))
                y_RH_nans = findall(!isnan, y)
                X_RH_full = X_flip[:, y_RH_nans]
                y_RH_full = y[y_RH_nans]
                y_pred = RHRBFN(X_RH_full) 
                y_diff = y_pred - y_RH_full
                y_diff_norm = sqrt(norm(y_diff)^2/length(y_diff))
                # println("Full Dataset RMSE: ", y_diff_norm)
                # println("Training Data RMSE: ", resData_RH[end])
                # println("Initial RMSE: ", resData_RH[1])
                return f, y_diff_norm, resData_RH[end], resData_RH[1]
        end
        f1_rh, y_diff_norm1_rh, resData_RH_end1, resData_RH_1 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestRH0.1.jld2");
        f2_rh, y_diff_norm2_rh, resData_RH_end2, resData_RH_2 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestRH0.2.jld2");
        f3_rh, y_diff_norm3_rh, resData_RH_end3, resData_RH_3 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestRH0.3.jld2");
        f4_rh, y_diff_norm4_rh, resData_RH_end4, resData_RH_4 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchTestRH0.4.jld2");
        f5_rh, y_diff_norm5_rh, resData_RH_end5, resData_RH_5 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchRHSampleCount.500.jld2");
        f6_rh, y_diff_norm6_rh, resData_RH_end6, resData_RH_6 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchRHSampleCount.1000.jld2");
        f7_rh, y_diff_norm7_rh, resData_RH_end7, resData_RH_7 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchRHSampleCount.2000.jld2");
        f8_rh, y_diff_norm8_rh, resData_RH_end8, resData_RH_8 = findRHRMSEs("ExtraFiles/PitchExperiments/Results/PitchRHSampleCount.5000.jld2");
        

        println("Min and Max RMSEs for the Full Dataset: ", minimum([y_diff_norm1_rh, y_diff_norm2_rh, y_diff_norm3_rh, y_diff_norm4_rh]), " and ", maximum([y_diff_norm1_rh, y_diff_norm2_rh, y_diff_norm3_rh, y_diff_norm4_rh]))
        println("Average Training Data RMSE: ", mean([resData_RH_end1, resData_RH_end2, resData_RH_end3, resData_RH_end4]))
        println("StD of Training Dataset RMSE: ", std([resData_RH_end1, resData_RH_end2, resData_RH_end3, resData_RH_end4]))
        println("Average Full Dataset RMSE: ", mean([y_diff_norm1_rh, y_diff_norm2_rh, y_diff_norm3_rh, y_diff_norm4_rh]))
        println("StD of Full Dataset RMSE: ", std([y_diff_norm1_rh, y_diff_norm2_rh, y_diff_norm3_rh, y_diff_norm4_rh]))
        println("Initial RMSE: ", resData_RH_1)

        ## Pressure RMSES for smaller sample counts    
        println("Full Data Results/RMSEs for the Pressure sample counts")
        println("Initial RMSE: ", resData_RH_1)
        println("RMSE for 500 samples: ", y_diff_norm5_rh)
        println("RMSE for 1000 samples: ", y_diff_norm6_rh)
        println("RMSE for 2000 samples: ", y_diff_norm7_rh)
        println("RMSE for 5000 samples: ", y_diff_norm8_rh)
        println("RMSE for 10000 samples: ", y_diff_norm1_rh)

        println("Full Data Results/RMSEs for the Pressure sample counts", y_diff_norm1_rh, " ", y_diff_norm2_rh, " ", y_diff_norm3_rh, " ", y_diff_norm4_rh, " ", y_diff_norm5_rh, " ", y_diff_norm6_rh, " ", y_diff_norm7_rh, " ", y_diff_norm8_rh)


## Evaluation of the Truncated Temperature Dataset

       function findTruncTempRMSEsRound(DataFile::String)
                # @load DataFile X_temp y_temp FinalTheta_temp resData_temp
                # f
                @load "ExtraFiles/PitchExperiments/Results/PitchTestTruncatedTV0.jld2" X_temp y_temp FinalTheta_temp resData_temp
                ## Evaluate the RBFN with the full dataset and the final theta
                @load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
                ## Evaluate the RBFN with the full dataset and the final theta
                T2 = Gaussian{Anisotropic{Aligned}, Float64, 3}
                X_flip = Float64.(permutedims(X,(2,1)))
                X_temp_full = X_flip[[1,2,15], :]
                y_temp_full = y
                y_rnd_norm = zeros(length(resData_temp))
                y_norm = zeros(length(resData_temp))
                y_rnd_norm[1] = sqrt(norm(y_temp_full)^2/length(y_temp_full))
                y_norm[1] = sqrt(norm(y_temp_full)^2/length(y_temp_full))   
                res = y_temp_full
                res_evalphi = res
                l_inf = zeros(length(resData_temp))
                l_inf[1] = norm(res_evalphi, Inf)

                for i = 1:500
                        y_temp_pred = FinalTheta_temp[i,end-1] .* SLFA.eval_phi(X_temp_full, FinalTheta_temp[i,:],T2) .+ FinalTheta_temp[i,end]
                        res_evalphi = res_evalphi - y_temp_pred
                        y_temp_round = round.(res_evalphi)
                        y_temp_now = res_evalphi
                        # y_temp_round = (res_evalphi)
                        y_rnd_norm[i+1] = sqrt(norm(y_temp_round)^2/length(y_temp_full))
                        y_norm[i+1] = sqrt(norm(y_temp_now)^2/length(y_temp_full))
                        l_inf[i+1] = norm(res_evalphi, Inf)
                        
                end

                f = GLMakie.Figure()
                ax = Axis(f[1,1],
                xlabel = "N",
                ylabel = "Training RMSE",
                title = "Pitch Temperature Training Data",
                yscale = log10,
                yticks = LogTicks(-2:1),
                titlesize = 20,
                xlabelsize = 20,    
                ylabelsize = 20,
                )
                # GLMakie.ylims!(ax, 10.0^-1, 10.0^0)

                Ns = collect(0:1:(length(y_rnd_norm)-1))
                GLMakie.scatter!(ax, Ns, y_rnd_norm)

                return f, y_rnd_norm, y_norm, resData_temp[end], resData_temp[1], l_inf, res_evalphi
        end 


        
                @load "ExtraFiles/PitchExperiments/Results/PitchTestTruncatedTV0.jld2" X_temp y_temp FinalTheta_temp resData_temp
        f1_tt, y_rnd_norm1_tt, y_norm1_tt, ~, ~, ~, resFinal1_tt = findTruncTempRMSEsRound("ExtraFiles/PitchExperiments/Results/PitchTestTruncatedTV02.jld2");
      

@load 
## Plot the Residual itself
@load "ExtraFiles/PitchExperiments/obs_temp_c_Xy.jld2" X y
        @load "ExtraFiles/PitchExperiments/Results/PitchTestTruncatedTV0.jld2" X_temp y_temp FinalTheta_temp resData_temp
        f = GLMakie.Figure()
        ax = Axis(f[1,1],
        xlabel = "TOD",
        ylabel = "Temperature Residual",
        title = "Index-plotted Residuals for the Complete Temperature Dataset",
        titlesize = 20,
        xlabelsize = 20,    
        ylabelsize = 20,
        )

        Ns = collect(0:1:(length(res_evalphi1_t)-1))
        GLMakie.scatter!(ax, X[:,1], res_evalphi1_t)
        GLMakie.scatter!(ax, X[:,1], resFinal1_tt, color = :red)