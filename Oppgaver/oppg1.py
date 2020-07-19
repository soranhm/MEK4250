import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Tkinter import *

def all_func():
    opp = int(opp2.get())

    if opp == 1:
        x = sp.Symbol('x')
        print('\nOPPGAVE 1')
        print('\nCalculating |sin(kpix)| with different norms')
        print('k    |  L^2  |   L^7  |   H^1')
        print('------------------------------')
        for k in [1,10,100]:
            l2 = lambda x:sp.sin(k*sp.pi*x)**2
            l7 = lambda x:(abs(sp.sin(k*sp.pi*x))**7)
            H1 = lambda x:sp.sin(k*sp.pi*x)**2 + (k*sp.pi*sp.cos(k*sp.pi*x))**2
            i2,e2 = np.sqrt(quad(l2,0,1))
            i7,e7 = quad(l7,0,1,epsabs=10e-4,epsrel=10e-4)
            i7 = float(i7)
            i7 = pow(i7,1.0/7)
            iH1,eh1 = np.sqrt(quad(H1,0,1,epsabs=10e-4,epsrel=10e-4))
            print('%3d  | %4.3f | %4.3f  |%4.0d '%(k,i2,i7,iH1))
        return 'Oppgave 1'
    elif opp == 2:
        print('\nOPPGAVE 2')
        print('\nCalculating |u - f| with different norms')
        print('  n    |  L^2  |   L^7  |   L^inf |   H^1')
        print('------------------------------------------')
        for n in [10,100]:
            m = 500
            u,u1,x,du = func(n,m)
            norm2 = np.sum((u-f(x))**2)
            norm7 = np.sum((u-f(x))**7)
            normu = np.max(abs(u-f(x)))
            normH = np.sqrt(norm2 + np.sum((du-df(x,m))**2))
            norm2_ = np.sqrt(norm2)
            norm7_ = pow(norm7,1.0/7)
            print(' %5d | %4.3f |  %4.3f | %4.3f | %4.3f '%(n,norm2_,norm7_,normu,normH))
        return 'Oppgave 2'
    elif opp == 3:
        ('\nPloting all the cases')

        n = int(n2.get())
        m = int(m2.get())
        print 'x = ',m,'k = ',n
        x,u,u1,du = func(n,m)
        return plot2(x,u,u1,du)

def plot2(x,u,u1,du):
    return plt.plot(x,f(x),x,u,x,u1,x,u_x(x)),plt.xlabel('x'),plt.ylabel('u'),plt.title('Ploting all the cases'),plt.legend(['f(x)','u = f','-u_xx = f', 'u_e' ]),plt.show()

def f(x):
    if 1./3 <= x <=2./3:
        return 1
    else:
        return 0

def df(x,m):
    if x == 1./3:
        return m
    elif x == 2./3:
        return -m
    else:
        return 0

def u_x(x):
    if x < 1./3:
        return 1./6 * x
    elif 1./3 < x < 2./3:
        return 1./2 * x**2
    else: # x > 2./3
        return 1./6 * (1-x)

f = np.vectorize(f)
df = np.vectorize(df)
u_x = np.vectorize(u_x)

def func(n,m):
    x = np.linspace(0,1,m)
    u = np.zeros(m)
    u1 = np.zeros(m)
    du = np.zeros(m)
    for i in range(0,m):
        for j in range(1,n):
            c_j = -2./(np.pi*j)*(np.cos(j*np.pi*2./3)- np.cos(j*np.pi*1./3))
            c_j2 = -2./(np.pi*j*(j*np.pi)**2)*(np.cos(j*np.pi*2./3)- np.cos(j*np.pi*1./3))
            u[i] +=   c_j * np.sin(j*np.pi*x[i])
            u1[i] +=   c_j2 * np.sin(j*np.pi*x[i])
            du[i] += -2./(np.pi*j)*(np.cos(j*np.pi*2./3)- np.cos(j*np.pi*1./3)) * j * np.pi* np.cos(j*np.pi*x[i])
    return x,u,u1,du



def valg():
    opp = int(opp2.get())
    if opp == 1:
        return all_func()

    elif opp == 2:
        return all_func()

    if opp == 3:

        tekst1.pack()
        m2.pack()
        tekst2.pack()
        n2.pack()

        plot_ = Button(root, text='Plot',command=all_func)
        plot_.pack()


root = Tk()

root.title("Norm and Plot")
root.geometry("200x200")

tekst = Label(root, text='velg oppgave nr (1, 2 eller 3):')
tekst.pack()

opp2 = Entry(root, width=5)
opp2.pack()

tekst1 = Label(root, text='Velg X verider :')
m2 = Entry(root, width=5)

tekst2 = Label(root, text='Velg k verider :')
n2 = Entry(root, width=5)

run_b = Button(root, text=' Kjor',command=valg)
run_b.pack()


root.mainloop()
