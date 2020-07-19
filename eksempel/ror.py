from fenics import *
import matplotlib.pyplot as plt
import numpy as np


def u_boundary(x):
    return x[0] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS


def p_boundary(x):
    return x[0] > 1.0 - DOLFIN_EPS


mesh = UnitSquareMesh(20, 20)
# Define function spaces
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#Q = FunctionSpace(mesh, "DG", 0)
TH = V * Q
W = FunctionSpace(mesh, TH)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)
f = Constant([0, 0])

u_analytical = Expression(("x[1]*(1-x[1])", "0.0"), degree=2)
p_analytical = Expression(("-2+2*x[0]"), degree=1)

bc_u = DirichletBC(W.sub(0), u_analytical, u_boundary)
bc = [bc_u]

a = inner(grad(u), grad(v)) * dx + div(u) * q * dx + div(v) * p * dx
print(f, v)
L = inner(f, v) * dx

UP = Function(W)
A, b = assemble_system(a, L, bc)

solve(A, UP.vector(), b, "lu")
U, P = UP.split()

#plot(U, title="Numerical velocity")
#plot(P, title="Numerical pressure")

#U_file = File("U.pvd")
# P_file = File("P.pvd") LAGER FOR STOR FIL

plt.figure(1)
plot(U, title="Numerical velocity")
# plt.savefig("Uo_fig.pdf")
plt.figure(2)
plot(P, title="Numerical pressure")
# plt.savefig("Po_fig.pdf")

V2 = VectorFunctionSpace(mesh, "Lagrange", 2)
Q2 = FunctionSpace(mesh, "Lagrange", 1)
U_analytical = project(u_analytical, V2)
P_analytical = project(p_analytical, Q2)

#Uo_file << U
#Po_file << P

test = inner(U - U_analytical, U - U_analytical) * dx
test = assemble(test)
test2 = inner(P - P_analytical, P - P_analytical) * dx
test2 = assemble(test2)

print(np.sqrt(test))
print(np.sqrt(test2))

plt.figure(3)
plot(U_analytical, title="Analytical velocity")
plt.savefig("plot1.pdf")
plt.figure(4)
plot(P_analytical, title="Analytical pressure")
plt.savefig("plot2.pdf")

plt.show()
