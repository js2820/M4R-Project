from firedrake import *

ncells = 20
base_mesh = PeriodicUnitIntervalMesh(ncells)

height = 1.
nlayers = 10
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
u += f_beta

print('New')

with CheckpointFile("heat.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(u)

File("weak1creat.pvd").write(u, u0)
