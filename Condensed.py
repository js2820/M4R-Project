from firedrake import *

def data_create(ncells = 20, nlayers = 10):

    ncells = ncells
    base_mesh = PeriodicUnitIntervalMesh(ncells)

    height = 1.
    nlayers = nlayers
    mesh = ExtrudedMesh(base_mesh, layers = nlayers, layer_height=height/nlayers, name = 'meshA')

    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)

    deltat = 0.1

    t = 0

    x, y = SpatialCoordinate(mesh)

    #Create some constants for the equations
    kappa = Constant(0.1)
    C = Constant(1)

    test = sin(pi*x)*cos(2*pi*x)
    f.interpolate(test)
    u0 = Function(V).assign(f)

    #Advection equation
    LHS = (inner(v, u) + kappa*deltat*inner(v.dx(0), u.dx(0)) + deltat*C*v*u.dx(0))*dx

    RHS = inner(v, f) * dx

    u = Function(V, name = 'u')

    while t < (1-deltat/2):
        solve(LHS == RHS, u, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
        f.assign(u)
        t+=deltat

    pcg = PCG64(seed=123456789)
    rg = RandomGenerator(pcg)

    f_beta = rg.normal(V, 0, 1)

    return mesh, u


#mesh, measurement = date_create()

def project_solve(mesh, measurement, gamma_var = 1, C_var = 0.1, Sigma_var = 0.1):

    # make some function spaces
    CG1_1D_elt = FiniteElement("CG", interval, 1)
    DG0_1D_elt = FiniteElement("DG", interval, 0)
    Uelt = TensorProductElement(CG1_1D_elt, CG1_1D_elt)
    VU = FunctionSpace(mesh, Uelt)
    Pelt = TensorProductElement(CG1_1D_elt, DG0_1D_elt)
    VP = FunctionSpace(mesh, Pelt)
    W = VU * VP # the mixed space for the coupled x-p system

    # solution!
    wk = Function(W)
    uk, pk = split(wk)

    wkp1 = Function(W)
    ukp1, pkp1 = split(wkp1)

    # model parameters
    # initial condition from before seeing the data
    x, t = SpatialCoordinate(mesh)
    u0 = sin(2*pi*x)*cos(pi*x)
    # gamma - variance in initial condition error
    gamma = Constant(gamma_var)
    # variance in observation error
    C = Constant(C_var)
    # fake some data
    u1 = 0.5*sin(2*pi*x)
    # variance for model error
    Sigma = Constant(Sigma_var)

    kappa = Constant(0.1)
    c = Constant(0.1)
    # the functional F
    print(gamma, C, Sigma)

    # penalty term for initial condition (f0)
    F = 0.5*(ukp1-u0)*(ukp1-u0)/gamma*ds_b # integral over t=0 "surface"

    # "data" (g0)
    F += 0.5*(ukp1-u1)*(ukp1-u1)/C*ds_t # integral over t=1 "surface"
    
    # "cost/control term"
    F += -0.5*pkp1*pkp1/Sigma*dx

    # "dynamical constraint"
    F += (pkp1*ukp1.dx(1) + kappa*pkp1.dx(0)*ukp1.dx(0) + c*pkp1*ukp1.dx(0))*dx

    # the equation to solve
    dF = derivative(F, wkp1)

    daparams_brute = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # make a problem to solve
    daparams_field = {
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type" : "additive",
        'fieldsplit_0_ksp_type' : 'preonly',
        'fieldsplit_0_pc_type' : 'lu',
        'fieldsplit_0_pc_factor_mat_solver_type' : 'mumps',
        'fieldsplit_1_ksp_type' : 'preonly',
        'fieldsplit_1_pc_type' : 'lu',
        'fieldsplit_1_pc_factor_mat_solver_type' : 'mumps',
        'snes_monitor' : None,
        'ksp_monitor' : None,
        'fieldsplit_0_ksp_monitor' : None,
        'fieldsplit_1_ksp_monitor' : None,
    } 


    boundary = DirichletBC(W.sub(0), u0 - gamma*pk, 'bottom')  #Is this the enforcement of the initial condition
    DAProblem = NonlinearVariationalProblem(dF, wkp1, bcs = boundary) 
    DASolver = NonlinearVariationalSolver(DAProblem,
                                          solver_parameters=daparams_brute)

    '''u, p = wkp1.subfunctions
    outfile = File("Para.pvd")
    outfile.write(u, p)'''

    error = 1e10
    tol = 1e-6
    count = 0
    alpha = 0.5

    #Loop until error is acceptable
    while error > tol:
        count += 1
        DASolver.solve()
        error = errornorm(wk, wkp1)
        
        wk.assign(alpha*wk + (1-alpha)*wkp1)
        '''outfile.write(u, p)'''

    #Write the solutions to another file

    return count, error

def wdidte(gamma = 0.1, C = 0.1, Sigma = 0.1, ncells = 20, nlayers = 10):
    mesh, u = data_create(ncells, nlayers)
    return project_solve(mesh, u, gamma, C, Sigma)

def vary_params(choice):
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .8, .9, 1]
    count_list = []
    params = [0.1, 0.1, 0.1]
    for val in vals:

        params[choice] = val
        print(params)
        
        count, error = wdidte(gamma = params[0], C = params[1], Sigma = params[2])

        count_list.append(count)

    import matplotlib.pyplot as plt

    plt.plot(vals, count_list)
    plt.show()

