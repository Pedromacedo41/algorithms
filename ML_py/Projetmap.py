import numpy as np
import matplotlib.pyplot as plt 
import scipy as sps
from math import *
import cmath as cm

def f(x): # sin x
    return cm.sin(x)

def f_prime(x):
    return cm.cos(x)

p = 1.5 # Point ou on va calculer l'approximation
N = -15 # log (valeur minimum du pas)

# Possibles valeurs de delta
d_vec = np.logspace(-1, N,num=-N)

def f_1(d):
    return (1/d)*(f(p+d)-f(p))

def f_2(d):
    return (1/(2*d))*(f(p+d)-f(p-d))
    
def f_3(d):
    return (1/d)*f(complex(p,d)).imag

vf_1= np.vectorize(f_1)
vf_2= np.vectorize(f_2)
vf_3= np.vectorize(f_3)
vf_prime= f_prime(p).real*np.ones(-N)

e1 = np.log10(abs(vf_1(d_vec).real- vf_prime)/vf_prime)
e2 = np.log10(abs(vf_2(d_vec).real- vf_prime)/vf_prime)
e3 = np.log10(abs(vf_3(d_vec).real- vf_prime)/vf_prime)
    
plt.ion()
fig = plt.figure(figsize=(8,5))
plt.plot(range(1,-N+1),e1,linewidth = 1.5, marker='s', color='r', label = "DF aval")
plt.plot(range(1,-N+1),e2,linewidth = 1.5, marker='o', color='b', label = "DF centrées")
plt.plot(range(1,-N+1),e3,linewidth = 1.5, marker='^', color='k', label = "DF complexes")
plt.legend(loc='lower right',fontsize = 'large')
plt.grid("b-", linewidth=0.5)
plt.ylim(-17,0)
plt.xlim(0,16)
plt.ylabel("log(erreur relative)", fontsize = 15)
plt.xlabel("$-log(\delta)$", fontsize = 15)
plt.autoscale(True,'both')
plt.show()
fig.savefig('Figure_1.png')


a= 0; b=0; L = 2*np.sinh(1/2)
N= 50;
alpha = 10 ** (-3)
delta = 10 ** (-10)
eps = 10 ** (-2)
err = 10 ** (-9)

v0 = np.zeros(N)
v0[0]=a; v0[N-1]=b; 
for i in range(1,N-1): 
    v0[i]=-0.1

h= 1/(N-1)

def J(v): 
    s=0; t=0
    for i in range(0,(N-1)):
        s+=((v[i]+v[i+1])/2)*cm.sqrt(1+((v[i+1]-v[i])/h)*((v[i+1]-v[i])/h))
        t+=cm.sqrt(1+((v[i+1]-v[i])/h)*((v[i+1]-v[i])/h))
    return s*h+ (1/eps)*(h*t-L)*(h*t-L)
    
def deriveJ(v,i):   # v vecteur, i composant 
    z = np.zeros(N, dtype= complex)
    for j in range(0,N):
        if(j==i): z[i] = complex(v[i],delta)
        else: z[j]= v[j]
    return (1/delta)*J(z).imag

def gradJ(v):    # v vecteur
    Vderive = np.zeros(N)
    for i in range (1,N-1):
        Vderive[i]= deriveJ(v,i)
    return Vderive

def J_1(v): 
    s=0
    for i in range(0,(N-1)):
        s+=((v[i]+v[i+1])/2)*cm.sqrt(1+((v[i+1]-v[i])/h)*((v[i+1]-v[i])/h))
    return s*h

def J_2(v): 
    t=0
    for i in range(0,(N-1)):
        t+=cm.sqrt(1+((v[i+1]-v[i])/h)*((v[i+1]-v[i])/h))
    return (1/eps)*(h*t-L)*(h*t-L)

def methodeGradient(v0, err):  # donné valeur initiale, err critère d'arrete 
    evol = [J(v0)]
    evol1 = [J_1(v0)]
    evol2 = [J_2(v0)]
    for i in range (1,12*N):
        f = gradJ(v0)
        v1 = v0 - alpha* f/np.linalg.norm(f);
        d= J(v1); 
        evol+=[d]; evol1+=[J_1(v1)];  evol2+=[J_2(v1)]
        v0 = v1
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(1,12*N+1),evol, color='r')
    plt.plot(range(1,12*N+1),evol1, color='b', label = "$J_1$")
    plt.plot(range(1,12*N+1),evol2, color='k', label = "$J_2$")
    plt.legend(loc='best')
    plt.grid("b-", linewidth=0.5)
    plt.ylabel("Valeur de $J(v^k)$", fontsize = 15)
    plt.xlabel("Iterations", fontsize = 15)
    plt.show()
    fig.savefig('Figure_2.png')
    return v0

v0 = methodeGradient(v0, err) 

a= np.linspace(0,1,N);
plt.plot(a,v0, marker='^' ,color='b')

plt.grid("b-", linewidth=0.5)
plt.show()

def deriveJ_dif_finie(v,i):   # v vecteur, i composant 
    z1 = np.zeros(N)
    z2 = np.zeros(N)
    for j in range(0,N):
        if(j==i): z1[j]= v[j]-delta;  z2[j]= v[j]+delta;
        else: z1[j]= v[j];  z2[j]= v[j];
    return (J(z1)- J(z2))/(2*delta)

def gradJ_dif_finie(v):    # v vecteur
    Vderive = np.zeros(N)
    for i in range (1,N-1):
        Vderive[i]= deriveJ_dif_finie(v,i)
    return Vderive


def methodeGradient_dif_finie(v0, err):  # donné valeur initiale, err critère d'arrete 
    evol = [J(v0)]
    evol1 = [J_1(v0)]
    evol2 = [J_2(v0)]
    for i in range (1,12*N):
        f = gradJ(v0)
        v1 = v0 - alpha* f/np.linalg.norm(f);
        d= J(v1); 
        evol+=[d]; evol1+=[J_1(v1)];  evol2+=[J_2(v1)]
        v0 = v1
    fig = plt.figure(figsize=(10,5))
    plt.plot(range(1,12*N+1),evol, color='r')
    plt.plot(range(1,12*N+1),evol1, color='b', label = "$J_1$")
    plt.plot(range(1,12*N+1),evol2, color='k', label = "$J_2$")
    plt.legend(loc='best')
    plt.grid("b-", linewidth=0.5)
    plt.ylabel("Valeur de $J(v^k)$", fontsize = 15)
    plt.xlabel("Iterations", fontsize = 15)
    plt.show()
    fig.savefig('Figure_2.png')
    return v0






