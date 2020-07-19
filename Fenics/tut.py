# Test Problem for the functin:
# eksakte: u(x,y) = 1 +x*x + 2*y*y
# Poissons equation gives Laplace(u) = f: f = -(2 + 4) = - 6

from fenics import *
import matplotlib.pyplot as plt # to show the plot

mesh = UnitSquareMesh(8, 8) # a mesh with 8*8 = 128 triangles of the unit square
V = FunctionSpace(mesh, 'Lagrange', 1) # finite element function space

u_e = Expression('1 + x[0] * x[0] + 2*x[1]*x[1]', degree = 2)

# setting the boundary
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_e, boundary)

# variation problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0) # right hand side f = -6

a = dot(grad(u),grad(v)) * dx
L = f*v * dx

u_h = Function(V)
solve(a == L, u_h, bc) # solving the variation problem

#using fenics to plot and matplotlib to show
plt.figure(1)
plot(u_h)

plt.figure(2)
plot(mesh)

plt.show()
