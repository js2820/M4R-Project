from firedrake import *

with CheckpointFile("heat.h5", 'r') as afile:
    mesh = afile.load_mesh("meshA")
    measurement = afile.load_function(mesh, "u")

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
gamma = Constant(1)
# variance in observation error
C = Constant(0.1)
# fake some data
u1 = 0.5*sin(2*pi*x)
# variance for model error
Sigma = Constant(0.0001)

kappa = Constant(0.1)
c = Constant(1)
# the functional F

# penalty term for initial condition (f0)
F = 0.5*(ukp1-u0)*(ukp1-u0)/gamma*ds_b # integral over t=0 "surface"

# "data" (g0)
F += 0.5*(ukp1-u1)*(ukp1-u1)/C*ds_t # integral over t=1 "surface"

# "cost"
F += -0.5*pkp1*pkp1/Sigma*dx

# "dynamical constraint"
F += (pkp1*ukp1.dx(1) + kappa*pkp1.dx(0)*ukp1.dx(0) + c*pkp1*ukp1.dx(0))*dx

# the equation to solve
dF = derivative(F, wkp1)

# make a problem to solve
daparams = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
} 

boundary = DirichletBC(W.sub(0), u0 - gamma*pk, 'bottom')
DAProblem = NonlinearVariationalProblem(dF, wkp1, bcs = boundary)
DASolver = NonlinearVariationalSolver(DAProblem,
                                      solver_parameters=daparams)

u, p = wkp1.subfunctions
outfile = File("Para.pvd")
outfile.write(u, p)

for i in range(10):
    DASolver.solve()
    wk.assign(wkp1)
    outfile.write(u, p)



#Write the solutions to another file