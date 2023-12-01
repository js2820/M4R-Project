from firedrake import *

with CheckpointFile("heat.h5", 'r') as afile:
    mesh = afile.load_mesh("meshA")
    measurement = afile.load_function(mesh, "u")

print(measurement)

#final = measurement[0]