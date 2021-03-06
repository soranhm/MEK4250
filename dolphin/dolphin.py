from fenics import*
import matplotlib.pyplot as plt

# Load mesh and subdomains
mesh = Mesh("dolfin_fine.xml")
sub_domains = MeshFunction("size_t", mesh, "dolfin_fine_subdomains.xml")

#mesh = RectangleMesh(Point(0,0), Point(5, 1), 50, 10, "right/left")

#plot(mesh)
#plot(sub_domains)

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Collect boundary conditions
bcs = [bc0, bc1]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
print(f,v)

L = inner(inflow, v)*dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

'''
# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)
'''

# Split the mixed solution using a shallow copy
(u, p) = w.split()

print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

# # Split the mixed solution using a shallow copy
(u, p) = w.split()


u_file = File("u.pvd")
p_file = File("p.pvd")

# Plot solution
plot(u)
plot(p)

u_file << u
p_file << p
plt.show()
