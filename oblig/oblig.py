from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

# command line: python3 -m memory_profiler oblig.py
# need package: memory_profiler
# used to look at the memory_usage
#@profile
def oblig(m,k,l, plo = False, savef = False, error = False):
    """ Function to calculate u and v , and plot them """
    def u_boundary(x, on_boundary):
        return on_boundary #x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    '''
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1,DOLFIN_EPS)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0,DOLFIN_EPS)
    '''

    mesh = UnitSquareMesh(m,m)
    # Define function spaces
    V = VectorElement("Lagrange", mesh.ufl_cell(), k)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), l)
    #Q = FunctionSpace(mesh, "DG", 0)
    TH = V * Q
    W = FunctionSpace(mesh, TH)

    #by0 = Bottom()
    #by1 = Top()
    #boundary_markers = MeshFunction("size_t", mesh, 2-1)
    #by0.mark(boundary_markers, 4)
    #by1.mark(boundary_markers, 5)
    #ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Calculated f
    f = Expression(("pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])","pi*pi*cos(pi*x[0])"),pi = np.pi,degree =2)
    #f = Expression(("pi*pi*sin(pi*x[1])","pi*pi*cos(pi*x[0])+1"),pi = np.pi,degree =4)
    #f = Expression(("0","0"),pi = np.pi,degree =2)

    #given u and p
    u_analytical = Expression(("sin(pi*x[1])", "cos(pi*x[0])"),pi = np.pi,degree =k+1)
    #u_analytical = Expression(("0", "-0.5*(x[0]*(1-x[0]))"),pi = np.pi,degree =2)
    p_analytical = Expression(("sin(2*pi*x[0])"),pi = np.pi,degree =l+1)
    #p_analytical = Expression(("x[1]-0.5"),pi = np.pi,degree =4)

    bc_u = DirichletBC(W.sub(0), u_analytical, u_boundary)
    bc = [bc_u]

    #hx = Expression(("0","x[1]"),pi = np.pi, degree =2)
    #hy = Expression(("0","-x[1]"),pi = np.pi,degree =2)
    #hx = Expression(("-pi*cos(pi*x[1])","x[1]"),pi = np.pi, degree =4)
    #hy = Expression(("pi*cos(pi*x[1])","-x[1]"),pi = np.pi,degree =4)
    #du_dn = Expression((("0","np.pi*cos(np.pi*x[1])"),("-np.pi*sin(np.pi*x[0])")))
    #n = FacetNormal(mesh)
    #h = dot(du_dn,n) - dot(p,n)

    # A(u,p;v,q) := a(u,v) + b(v,p) + b(u,q) = L(v)
    a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
    L = inner(f, v)*dx #+ inner(hx,v)*ds(4) + inner(hy,v)*ds(5)
    #L = inner(f, v)*dx + h*v*ds
    UP = Function(W)
    A, b = assemble_system(a, L, bc)

    #print(UP.vector().size()) # prints array size
    solve(A, UP.vector(), b, "lu")
    U, P = UP.split()

    P_average = assemble(P*dx) # average pressure

    V2 = VectorFunctionSpace(mesh, "Lagrange", k+1)
    Q2 = FunctionSpace(mesh, "Lagrange", l+1)
    U_analytical = project(u_analytical, V2)
    P_analytical = project(p_analytical, Q2)

    if(plo):
        plt.subplot(2, 2, 1)
        plot(U, title="Numerical velocity")
        plt.subplot(2, 2, 2)
        plot(P, title="Numerical pressure")
        plt.subplot(2, 2, 3)
        plot(U_analytical, title="Analytical velocity")
        plt.subplot(2, 2, 4)
        plot(P_analytical + P_average, title="Analytical pressure")
        plt.suptitle("Numerical and Analytical plot with mesh: %d, k = %d, l = %d"%(m,k,l), color="red",y=1)
        plt.tight_layout()

    if(savef):
        Un_file = File("Un.pvd")
        Pn_file = File("Pn.pvd")
        Pa_file = File("Pa.pvd")
        Ua_file = File("Ua.pvd")
        Un_file << U
        Pn_file << P
        Pa_file << P_analytical
        Ua_file << U_analytical

    if (error):
        """ Calculating error estimate """
        #LHS
        u_uh = sqrt(assemble(inner(U_analytical - U,U_analytical - U)*dx))
        p_ph = sqrt(assemble(inner(P_analytical + P_average - P,P_analytical+ P_average - P)*dx))
        #RHS
        ''' Calculated Hilberts space by hand '''
        #Inserting Hilbert norms of u in 4 and 3
        u_0 = U_analytical
        u_1 = Expression(("pi*cos(pi*x[1])" ,"- pi*sin(pi*x[0])"),pi = np.pi,degree =k+1,domain=mesh)
        u_2 = Expression(("- pi_2*sin(pi*x[1])" ,"- pi_2*cos(pi*x[0])"),pi_2 = np.pi**2,pi = np.pi,degree =k+1,domain=mesh)
        u_3 = Expression(("- pi_3*cos(pi*x[1])" ," pi_3*sin(pi*x[0])"),pi_3 = np.pi**3,pi = np.pi,degree =k+1,domain=mesh)
        u_4 = Expression(("pi_4*sin(pi*x[1])" ," pi_4*cos(pi*x[0])"),pi_4 = np.pi**4,pi = np.pi,degree =k+1,domain=mesh)
        u_5 = Expression(("pi_5*cos(pi*x[1])" ,"- pi_5*sin(pi*x[0])"),pi_5 = np.pi**5,pi = np.pi,degree =k+1,domain=mesh)
        u_k5 = sqrt(assemble(inner(u_0,u_0)*dx + inner(u_1,u_1)*dx  +\
        inner(u_2,u_2)*dx + inner(u_3,u_3)*dx + inner(u_4,u_4)*dx + inner(u_5,u_5)*dx))
        u_k4 = sqrt(assemble(inner(u_0,u_0)*dx + inner(u_1,u_1)*dx  +\
        inner(u_2,u_2)*dx + inner(u_3,u_3)*dx + inner(u_4,u_4)*dx))
        #Inserting Hilbert norms of p in 4,3 and 2
        p_0 = P_analytical
        p_1 = Expression(("2*pi*cos(2*pi*x[0])"),pi = np.pi,degree =l+1,domain=mesh)
        p_2 = Expression(("- 4*pi_2*sin(2*pi*x[0])"),pi = np.pi,pi_2 = np.pi**2,degree =l+1,domain=mesh)
        p_3 = Expression(("- 8*pi_3*cos(2*pi*x[0])"),pi = np.pi,pi_3 = np.pi**3,degree =l+1,domain=mesh)
        p_4 = Expression(("16*pi_4*sin(2*pi*x[0])"),pi = np.pi,pi_4 = np.pi**4,degree =l+1,domain=mesh)
        p_l4 = sqrt(assemble(inner(p_0,p_0)*dx + inner(p_1,p_1)*dx +\
        inner(p_2,p_2)*dx + inner(p_3,p_3)*dx + inner(p_4,p_4)*dx))
        p_l3 = sqrt(assemble(inner(p_0,p_0)*dx + inner(p_1,p_1)*dx +\
        inner(p_2,p_2)*dx + inner(p_3,p_3)*dx))
        p_l2 = sqrt(assemble(inner(p_0,p_0)*dx + inner(p_1,p_1)*dx +\
        inner(p_2,p_2)*dx))
        h_k = (1./m)**k
        h_l = (1./m)**(l+1)
        if(k == 4 and l == 3):
            u_ = u_k5
            p_ = p_l4
            C = (u_uh + p_ph)/(h_k*u_k5*p_l4)
        elif (k == 4 and l == 2):
            u_ = u_k5
            p_ = p_l3
            C = (u_uh + p_ph)/(h_k*u_k5*p_l4) # Using previous C
        elif (k == 3 and l == 2):
            u_ = u_k4
            p_ = p_l3
            C = (u_uh + p_ph)/(h_k*u_k4*p_l3)
        elif (k == 3 and l == 1):
            u_ = u_k4
            p_ = p_l2
            C = (u_uh + p_ph)/(h_k*u_k4*p_l3) # Using previous C
        return h_k, h_l, u_uh, p_ph, u_, p_, C

def test(m,k,l,z):
    ''' This function runs for each mesh value '''
    for j in m:
        h_k, h_l, u_uh, p_ph, u_, p_, C = oblig(j,k,l, error = True)
        u_uh_p_ph = u_uh + p_ph
        err[z].append(u_uh_p_ph)
        hs[z].append(1./j)
        if(k == (l+1)):
            print("\033[1;36m|\x1b[0m     Mesh = %3d : %.2e + %.2e  =< C*%.2e*(%.2f + %.2f) --> C >= %.2e    \033[1;36m|\x1b[0m"%(j,u_uh,p_ph,h_k,u_,p_,C))
            s.append(C) # used to calculated the average C for next l
            err_R[z].append(C*h_k*(u_+p_))     # sum of right side
        else:
            if (k==4):
                r = 0
                for f in range(0,len(s)): # the first  C values, which is from P_4-P_3
                    r += s[f] # D for P_4 - P_2
                avg_C = (r/len(s))
                D = ((u_uh_p_ph - avg_C)*h_k*u_)/(h_l*p_) # Calculating D
                err_R[z].append(C*h_k*u_ + D*h_l*p_) # sum of right side
                print("\033[1;36m|\x1b[0m  Mesh = %3d : %.2e + %.2e  =< %.2e*%.2e*%.2f + D*%.2e*%.2f --> D >= %.2e   \033[1;36m|\x1b[0m"%(j,u_uh,p_ph,(r/len(s)),h_k,u_,h_l,p_,D))
            elif (k==3):
                r = 0
                for f in range(int(len(s)/2),len(s)): # rest of the C values,  which is from P_3-P_2
                    r += s[f] # D for P_3 - P_1
                avg_C = (r/(len(s)/2))
                D = ((u_uh_p_ph - avg_C)*h_k*u_)/(h_l*p_) # Calculating D
                err_R[z].append(C*h_k*u_ + D*h_l*p_) # sum of right side
                print("\033[1;36m|\x1b[0m  Mesh = %3d : %.2e + %.2e   =< %.2e*%.2e*%.2f + D*%.2e*%.2f --> D >= %.2e   \033[1;36m|\x1b[0m"%(j,u_uh,p_ph,(r/(len(s)/2)),h_k,u_,h_l,p_,D))
    return

def test2(m,k,l):
    ''' This function runs for each given k and l values '''
    z = 0
    for i, j in zip(k, l):
        if(i == (j+1)):
            print("\n\033[1;36m+------------------------------------------P_%1d - P_%1d------------------------------------------+\x1b[0m" %(i,j))
            print("\033[1;36m|\x1b[0m                  \033[1;32m|| u - u_h || + || p - p_h || =< C*h^%d*(||u||_%d + ||p||_%d)\x1b[0m                \033[1;36m|\x1b[0m"%(i,i+1,i))
            test(m,i,j,z)
            print("\033[1;36m+---------------------------------------------------------------------------------------------+\x1b[0m")
        else:
            print("\n\033[1;36m+-----------------------------------------------P_%1d - P_%1d------------------------------------------------+\x1b[0m" %(i,j))
            print("\033[1;36m|\x1b[0m                    \033[1;32m|| u - u_h || + || p - p_h || =< C*h^%d*||u||_%d + D*h^%d*||p||_%d\x1b[0m                      \033[1;36m|\x1b[0m"%(i,i+1,j+1,j+1))
            test(m,i,j,z)
            print("\033[1;36m+--------------------------------------------------------------------------------------------------------+\x1b[0m")
        z += 1
    return

m = [4, 8, 16, 32]
liste = ["error"]
# TO USE A GENERAL M AND LISTE VALUES REMOVE NEXT FOUR LINES (uses the lines above)
#m_in = input("\033[1;36mEnter a multiple MESH values, seperate with space(INTEGER):\x1b[0m")
#input_string = input("\033[1;36mChoose: plot, error, log_plot or/and savefigure:,separated by space(STRING):\x1b[0m")
#liste  = input_string.split()
#m = list(map(int, m_in.split()))

start = time.time() # Taking time for fun
k = [4,4,3,3]
l = [3,2,2,1]
#k =[4]
#l =[3]
if ("error" in liste):
    s = [] # used to save C values
    err = [[] for _ in range(len(k))] # 2D list to get the error values (LHS)
    err_R = [[] for _ in range(len(k))] # 2D list to get the error values (LHS)
    hs = [[] for _ in range(len(k))] # 2D list to get the error values (LHS)
    print("\033[1;36mPrinting out the error for all the given m,k and l values ...\x1b[0m")
    test2(m,k,l)
    print("\033[1;36mSaving the error to a file (run log_plot to run) ...\x1b[0m")
    with open("error.txt", 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in err)
    with open("hs.txt", 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in hs)

if ("log_plot" in liste):
    print("\033[1;36mPloting the error for all given m,k and l values ...\x1b[0m")
    try:
        ins = open( "error.txt", "r" )
    except FileNotFoundError:
        print("\033[1;31mYOU NEED TO RUN error FIRST\x1b[0m")
    try:
        ins2 = open( "hs.txt", "r" )
    except FileNotFoundError:
        print("\033[1;31mYOU NEED TO RUN error FIRST\x1b[0m")
    err = [[float(n) for n in line.split()] for line in ins]
    hs = [[float(n) for n in line.split()] for line in ins2]
    plt.plot(hs[0][:], err[0][:], 'k^:', label='k = %d, m = %d'%(k[0],l[0]))
    plt.plot(hs[1][:], err[1][:], 'b^:', label='k = %d, m = %d'%(k[1],l[1]))
    plt.plot(hs[2][:], err[2][:], 'r^:', label='k = %d, m = %d'%(k[2],l[2]))
    plt.plot(hs[3][:], err[3][:], 'g^:', label='k = %d, m = %d'%(k[3],l[3]))
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f"ERROR with MESH: {m}")
    plt.xlabel('h : [1/m]')
    plt.ylabel('Error : [|| u - u_h || + || p - p_h ||]')
    plt.savefig("log_plot.pdf")
    plt.show()


if ("plot" in liste):
    "using the bigges k and l"
    i = 1
    print("\033[1;36mPloting for each mesh with k = %d and l = %d ...\x1b[0m" %(k[0],l[0]))
    for j in m:
        plt.figure(i)
        oblig(j,k[0],l[0], plo = True)
        plt.savefig("figure_%d_.pdf"%(j))
        print("\033[1;36mSaving plot to pdf file: figure_%d_.pdf\x1b[0m"%(j))
        i += 1
    plt.show()

if ("savefigure" in liste):
    """Only savef of the biggest mesh"""
    print("\033[1;36mSaving PARAVIEW files for the biggest mesh: %d \nwith k = %d and l = %d ...\x1b[0m"%(m[-1],k[0],l[0]))
    oblig(m[-1],k[0],l[0], savef = True)

if ("error" in liste):
    print("\033[1;36m+----------------------------------------------------------------.----+\x1b[0m")
    for j in range(len(err)):
        print("\033[1;36m|----------------------------\x1b[0m \033[1;32mk = %d , l = %d \x1b[0m\033[1;36m--------------------------|\x1b[0m"%(k[j],l[j]))
        for i in range(len(err[j])-1):
            x = err[j][i]/err[j][i+1]
            x2 = err_R[j][i]/err_R[j][i+1]
            print("\033[1;36m|\x1b[0m m[{0:2d}]/m[{1:2d}] :  {2:.6f} <= {3:.6f} | 2^{4:.6f}  <=  2^{5:.6f}  \033[1;36m|\x1b[0m".format(m[i],m[i+1],x,x2,ln(x)/ln(2),ln(x2)/ln(2)))
    print("\033[1;36m+----------------------------------------------------------------.----+\x1b[0m")

end = time.time()
print("\033[1;31mTotal running time: %.2f Seconds\x1b[0m" %(end - start))
