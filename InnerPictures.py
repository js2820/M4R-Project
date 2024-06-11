from firedrake import *

import sys
sys.path.append("../home/bosh353/imp/M4R")

# import methods from siblings
from Inner import Inner_solve
from data_create import data_create

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #Create mesh and measurement to look at varying parameters
    mesh, u = data_create()

    unormal, _ = Inner_solve(mesh, u, 
                                    gamma_var = 1., 
                                    C_var = 1., 
                                    Sigma_var = 1.,
                                    return_result = True)
    ugamma, _ = Inner_solve(mesh, u,
                                    gamma_var = 0.01, 
                                    C_var = 1., 
                                    Sigma_var = 1.,
                                    return_result = True)

    
    uC, _ = Inner_solve(mesh, u,
                                gamma_var = 1., 
                                C_var = 0.01, 
                                Sigma_var = 1.,
                                return_result = True)

    
    uSigma, _ = Inner_solve(mesh, u,
                                  gamma_var = 1., 
                                  C_var = 1., 
                                  Sigma_var = 0.01,
                                  return_result = True)
    


    # Plotting u with all equal
    contour_plot = tricontourf(unormal)
    plt.title("Normal")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.yticks([])
    plt.colorbar(contour_plot)

    # Plotting u with large gamma
    contour_plot = tricontourf(ugamma)
    plt.title("Gamma")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.yticks([])
    plt.colorbar(contour_plot)
    
    # Plotting u with large C
    contour_plot = tricontourf(uC)
    plt.title("C")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.yticks([])
    plt.colorbar(contour_plot)

    # Plotting u with large Sigma
    contour_plot = tricontourf(uSigma)
    plt.title("Sigma")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.yticks([])
    plt.colorbar(contour_plot)

    plt.show()




    
