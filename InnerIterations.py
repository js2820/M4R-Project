from firedrake import *

import sys
sys.path.append("../home/bosh353/imp/M4R")

# import methods from siblings
from Inner import Inner_solve
from Outer import project_solve
from data_create import data_create

import numpy as np
import matplotlib.pyplot as plt


def vary_params(mesh, u, choice, vals):
    outer_list = np.zeros_like(vals)
    inner_list = np.zeros_like(vals)

    # Default parameters
    params = {
        'lambda': 0.5,
        'gamma': 0.1,
        'C': 0.1,
        'Sigma': 0.1,
        'tol' : 1e-6
    }
    
    for i, val in enumerate(vals):
        # Set the chosen parameter to the current value
        params[choice] = float(val)
        
        # Call Inner_solve with the current set of parameters
        outer_count, inner_count = Inner_solve(mesh, u, 
                                        Lambda = params['lambda'], 
                                        gamma_var = params['gamma'], 
                                        C_var = params['C'], 
                                        Sigma_var = params['Sigma'],
                                        tol = params['tol'])
        
        outer_list[i] = outer_count
        inner_list[i] = inner_count

    return outer_list, inner_list


if __name__ == "__main__":
    #Create mesh and measurement to look at varying parameters
    mesh, u = data_create()

    #A range of values to test for gamma, C and Sigma
    values = np.logspace(-1.5, 0, 50)

    #Get the number of iterations when we vary gamma, C and Sigma individual
    gamma_outer, gamma_inner = vary_params(mesh, u, 'gamma', values)
    C_outer, C_inner = vary_params(mesh, u, 'C', values)
    Sigma_outer, Sigma_inner = vary_params(mesh, u, 'Sigma', values)

    
    lambdas = np.linspace(0, 1, 51)[1:-1]
    lambdas_outer, lambdas_inner = vary_params(mesh, u, 'lambda', lambdas)
    

    #Plot them
    plt.plot(values, gamma_outer, "x", label = 'Gamma varied')
    plt.plot(values, C_outer, "+", label = 'C varied')
    plt.plot(values, Sigma_outer, ".", label = 'Sigma varied')
    plt.xscale('log')
    plt.title('Outer Iterations')
    plt.ylabel('Number of Outer Iterations')
    plt.xlabel('Weight Value')
    plt.legend()
    plt.show()

    #Plot them
    plt.plot(values, gamma_inner, "x", label = 'Gamma varied')
    plt.plot(values, C_inner, "+", label = 'C varied')
    plt.plot(values, Sigma_inner, ".", label = 'Sigma varied')
    plt.xscale('log')
    plt.title('Inner Iterations')
    plt.ylabel('Number of Inner Iterations')
    plt.xlabel('Weight Value')
    plt.legend()
    plt.show()

    #Plot them
    plt.plot(values, gamma_inner/gamma_outer, "x", label = 'Gamma varied')
    plt.plot(values, C_inner/C_outer, "+", label = 'C varied')
    plt.plot(values, Sigma_inner/Sigma_outer, ".", label = 'Sigma varied')
    plt.xscale('log')
    plt.title('Number of Inner Iterations per Outer Iteration vs Weight Changes')
    plt.xlabel('Weight Value')
    plt.ylabel('Inner Iterations per Outer Iteration')
    plt.legend()
    plt.show()

    
    plt.plot(lambdas, lambdas_outer, "x", label = '$\lambda$ outer')
    plt.plot(lambdas, lambdas_inner, "x", label = '$\lambda$ inner')
    plt.xlabel('$\lambda$ values')
    plt.ylabel('Number of Iterations')
    plt.title('Iterations vs $\lambda$')
    plt.yscale('log')
    plt.legend()
    plt.show()

    ave_errors = Inner_solve(mesh, u, show_errors = True)
    gamma_errors = Inner_solve(mesh, u, gamma_var = 0.001, show_errors = True)
    sigma_errors = Inner_solve(mesh, u, Sigma_var = 0.001, show_errors = True)
    C_errors = Inner_solve(mesh, u, C_var = 0.001, show_errors = True)
    
    plt.plot(sigma_errors, '*', label = 'Sigma small')
    plt.plot(C_errors, '+', label = 'C small')
    plt.plot(gamma_errors, '.', label = 'Gamma small')
    plt.plot(ave_errors, "x", label = 'Normal')
    
    plt.xlabel('Iteration number')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error vs Iteration Number')
    plt.legend()
    plt.show()    
