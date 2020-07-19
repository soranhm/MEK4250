import time
from dolfin import *
lu_time = []
cg_time = []
cgilu_time = []
cgamg_time = []
dofs = []
#for N in [32, 64, 128]:

parameters["krylov_solver"]["relative_tolerance"] = 1.0e-8
parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-8
#parameters["krylov_solver"]["monitor_convergence"] = True
#parameters["krylov_solver"]["report"] = True
#parameters["krylov_solver"]["maximum_iterations"] = 50000

#for N in [32, 64, 128, 256, 512, 1024]:
for N in [32, 64, 128, 256, 512]:


  mesh = UnitSquareMesh(N, N)
  print(" N ", N, " dofs ", mesh.num_vertices())

  dofs.append(mesh.num_vertices())
  V = FunctionSpace(mesh, "Lagrange", 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  f = Expression("sin(x[0]*12) - x[1]",degree =2)
  a = u*v*dx  + inner(grad(u), grad(v))*dx
  L = f*v*dx

  U = Function(V)

  A = assemble(a)
  b = assemble(L)

  t0 = time.time()
  solve(A, U.vector(), b, "lu")
  t1 = time.time()
  print("Time for lu ", t1-t0)
  lu_time.append(t1-t0)

  t0 = time.time()
  U.vector()[:] = 0
  solve(A, U.vector(), b, "cg")
  t1 = time.time()
  print("Time for cg ", t1-t0)
  cg_time.append(t1-t0)

  t0 = time.time()
  U.vector()[:] = 0
  solve(A, U.vector(), b, "cg", "ilu")
  t1 = time.time()
  print("Time for cg/ilu ", t1-t0)
  cgilu_time.append(t1-t0)

  t0 = time.time()
  U.vector()[:] = 0
  solve(A, U.vector(), b, "cg", "amg")
  t1 = time.time()
  print("Time for cg/amg ", t1-t0)
  cgamg_time.append(t1-t0)


import pylab

pylab.plot(dofs, lu_time)
pylab.plot(dofs, cg_time)
pylab.plot(dofs, cgilu_time)
pylab.plot(dofs, cgamg_time)
pylab.legend(["lu", "cg", "cg/ilu", "cg/amg"])
pylab.show()



pylab.loglog(dofs, lu_time)
pylab.loglog(dofs, cg_time)
pylab.loglog(dofs, cgilu_time)
pylab.loglog(dofs, cgamg_time)
pylab.legend(["lu", "cg", "cg/ilu", "cg/amg"])
pylab.show()
