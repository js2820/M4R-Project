from firedrake import *
mesh = UnitSquareMesh(10, 10, name = 'meshA')

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)

deltat = 0.1
time = [deltat*i for i in range(int(1/deltat) + 1)]

x, y = SpatialCoordinate(mesh)

f.interpolate(sin(2*pi*x)*cos(pi*x))

LHS = (inner(v, u) + deltat*inner(v.dx(0), u.dx(0)))*dx

RHS = inner(v, f) * dx

u = Function(V, name = 'u')

for t in time:
    solve(LHS == RHS, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})
    f = u
    RHS = inner(v, f) * dx

with CheckpointFile("heat.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(u)