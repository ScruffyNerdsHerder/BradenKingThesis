
using SLFA
using Optim
using LsqFit
using LinearAlgebra
export lsq_TV_solver_LBFGS, lsq_TV_solver_CG, lsq_TV_solver_GradientDescent

function lsq_TV_solver_CG(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    res_new(theta) = res - theta[end-1] .* eval_phi(X, theta, T_phi) .- theta[end]

    if omega_TV > 0.0
        f_lsq_orig = norm(res_new(theta0))
        TV_orig = squaredTV(res_new(theta0), A, D)

        # Normalize/scale each term by their original value
        f_lsq(theta) = norm(res_new(theta)) / f_lsq_orig
        f_TV(theta) = squaredTV(res_new(theta), A, D) / TV_orig

        # Try constant omega
        omega = [1, omega_TV]
        f_lsq_TV(theta) = omega[1]*f_lsq(theta) + omega[2]*f_TV(theta) 

        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_lsq_TV, theta0, ConjugateGradient(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)

        # lsq_initial = f_lsq(theta0)
        # lsq_final = f_lsq(theta)
        # if lsq_final > lsq_initial
        #     theta_lsq = lsq_solver(theta0, X, res, A, D, N, T_phi)
        #     if squaredTV(res_new(theta_lsq), A, D) < TV_orig
        #         return theta_lsq
        #     else
        #         return theta0
        #     end
        # end

        return theta
    else
        f_obj(theta) = norm(res_new(theta))
        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_obj, theta0, ConjugateGradient(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)
        return theta
    end

end
function lsq_TV_solver_LBFGS(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    res_new(theta) = res - theta[end-1] .* eval_phi(X, theta, T_phi) .- theta[end]

    if omega_TV > 0.0
        f_lsq_orig = norm(res_new(theta0))
        TV_orig = squaredTV(res_new(theta0), A, D)

        # Normalize/scale each term by their original value
        f_lsq(theta) = norm(res_new(theta)) / f_lsq_orig
        f_TV(theta) = squaredTV(res_new(theta), A, D) / TV_orig

        # Try constant omega
        omega = [1, omega_TV]
        f_lsq_TV(theta) = omega[1]*f_lsq(theta) + omega[2]*f_TV(theta) 

        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_lsq_TV, theta0, LBFGS(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)

        # lsq_initial = f_lsq(theta0)
        # lsq_final = f_lsq(theta)
        # if lsq_final > lsq_initial
        #     theta_lsq = lsq_solver(theta0, X, res, A, D, N, T_phi)
        #     if squaredTV(res_new(theta_lsq), A, D) < TV_orig
        #         return theta_lsq
        #     else
        #         return theta0
        #     end
        # end

        return theta
    else
        f_obj(theta) = norm(res_new(theta))
        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_obj, theta0, LBFGS(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)
        return theta
    end

end
function lsq_TV_solver_GradientDescent(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    res_new(theta) = res - theta[end-1] .* eval_phi(X, theta, T_phi) .- theta[end]

    if omega_TV > 0.0
        f_lsq_orig = norm(res_new(theta0))
        TV_orig = squaredTV(res_new(theta0), A, D)

        # Normalize/scale each term by their original value
        f_lsq(theta) = norm(res_new(theta)) / f_lsq_orig
        f_TV(theta) = squaredTV(res_new(theta), A, D) / TV_orig

        # Try constant omega
        omega = [1, omega_TV]
        f_lsq_TV(theta) = omega[1]*f_lsq(theta) + omega[2]*f_TV(theta) 

        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_lsq_TV, theta0, GradientDescent(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)

        # lsq_initial = f_lsq(theta0)
        # lsq_final = f_lsq(theta)
        # if lsq_final > lsq_initial
        #     theta_lsq = lsq_solver(theta0, X, res, A, D, N, T_phi)
        #     if squaredTV(res_new(theta_lsq), A, D) < TV_orig
        #         return theta_lsq
        #     else
        #         return theta0
        #     end
        # end

        return theta
    else
        f_obj(theta) = norm(res_new(theta))
        Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
        result = optimize(f_obj, theta0, GradientDescent(); autodiff=AutoForwardDiff())
        
        theta = Optim.minimizer(result)
        return theta
    end

end