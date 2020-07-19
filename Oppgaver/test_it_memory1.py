
from dolfin import *
lu_memory = []
amg_memory = []
ilu_memory = []
dofs = []
#for N in [32, 64, 128]:

for N in [32, 64, 128, 256, 512, 1024]:

  mesh = UnitSquareMesh(N, N)
  print(" N ", N, " dofs ", mesh.num_vertices() )

  dofs.append(mesh.num_vertices())
  V = FunctionSpace(mesh, "Lagrange", 1)
  u = TrialFunction(V)
  v = TestFunction(V)

  f = Expression("sin(x[0]*12) - x[1]")
  a = u*v*dx  + inner(grad(u), grad(v))*dx
  L = f*v*dx

  U = Function(V)

  A = assemble(a)
  b = assemble(L)

  solve(A, U.vector(), b, "lu")
  mem_lu = memory_usage(False)
  print(mem_lu)
  lu_memory.append(mem_lu[1])

  solve(A, U.vector(), b, "cg", "amg")
  mem_amg = memory_usage(False)
  amg_memory.append(mem_amg[1])

  solve(A, U.vector(), b, "cg", "ilu")
  mem_ilu = memory_usage(False)
  ilu_memory.append(mem_ilu[1])




import pylab

print(lu_memory)
print(amg_memory)
print(ilu_memory)

pylab.loglog(dofs, lu_memory)
pylab.loglog(dofs, amg_memory)
pylab.loglog(dofs, ilu_memory)
pylab.legend(["lu memory", "amg memory", "ilu memory"],  loc='upper left')
pylab.show()
