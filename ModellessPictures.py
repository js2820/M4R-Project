from firedrake import *

import sys
sys.path.append("../home/bosh353/imp/M4R")

# import methods from siblings
from Modelless import Modelless_solve
from data_create import data_create

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #Create mesh and measurement to look at varying parameters
    mesh, u = data_create()

    unormal, _ = Modelless_solve(mesh, u, 
                                    gamma_var = 1., 
                                    C_var = 1., 
                                    Sigma_var = 1.,
                                    return_result = True)

    # Plotting u with all equal
    contour_plot = tricontourf(unormal)
    plt.title("Preconditioned Without Model Term")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.yticks([])
    plt.colorbar(contour_plot)

    plt.show()




    
