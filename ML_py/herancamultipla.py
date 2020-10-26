import numpy as np;
import matplotlib.pyplot as plt
import scipy.stats as sps;
from math import ceil

def liste_ecarte(n):
        cartes= list(range(n))
        supprimes = []
        while len(cartes)>1:
            supprimes+= [cartes[0]]
            cartes= cartes[2:]+ [cartes[1]];
        return supprimes,cartes[0]
        
        
def f(x):
     if x.any()==0: return 1
     else:  return np.sin(x)/x
     
def fibonacci(n, a=0,b=1):
    s_n= 2*np.random.binomial(1,.5,n)-1
    u=[a,b];
    for i in range(2,n):
        u+=[u[i-1]+s_n[i-2]*u[i-2]]
    return u
    
def TFC(n=1000,m=10000):
    x= 2*np.random.rand(m,n)-1
    sigma=3**(-.5)
    S=np.sum(x,axis=1)/(np.sqrt(n)*sigma)
    M= max(np.abs(S))
    plottt(sps.norm.pdf,a=M , labe= "N(0,1)", colo="r")
    plt.hist(S, bins = ceil(m**(1./3)*M*.5),density=1,label="Histogramme")
    plt.legend(loc='best')
    plt.title("TFC")
    plt.show()

def Loicomp(n=10000):
    x= 2*np.random.rand(n)-1
    y= 2*np.random.rand(n)-1
    S=x+y
    M= max(np.abs(S))
    plottt(f2,a=M , labe= "Theorique", colo="r")
    plt.hist(S, bins = ceil(n**(1./3)*M*.5),density=1,label="Histogramme")
    plt.legend(loc='best')
    plt.title("Comparison")
    plt.show()
    
def Loicomp2(n=10000):
    x= np.random.exponential(1,n)
    y= np.random.exponential(1,n)
    S=x+y
    M= max(np.abs(S))
    plottt2(f3,a=M, labe= "Theorique", colo="r")
    plt.hist(S, bins = ceil(n**(1./3)),density=1,label="Histogramme")
    plt.legend(loc='best')
    plt.title("Comparison")
    plt.show()

def f2(x):
    a= np.abs(x)
    for i in range(0,len(x)):
        a[i] = max(2-a[i],0)/4
    return a

def f3(x):
    
    if x.any()<0:  return 0
    else: return np.exp(-x)*x

    #plt.stem(np.arange(N+1), f, "r", label="loi theorique")
    #plt.stem(np.arange(N+1), s, "y", label="loi theorique")
    #plt.hist(x,bins=N+1, density=1, range=(-.5,N+.5), color="b", label="empirique")
    
def Question2(M= 100000, N=10,p=0.3):
    Prod= np.linspace(0,1,M+1)
    for n in range(0,M+1):
        x= np.random.binomial(N,p,n)
        prod=0
        s= np.linspace(0,1,N+1)
        for i in range (0,N+1):
            s[i]= np.sum(x == i,dtype=float)
            if s[i]!=0: prod+=(s[i]*np.log(s[i]/(n+1)))
            
            s[i]/=(n+1)
        Prod[n]=-prod/np.log(2)
    f=sps.binom.pmf(np.arange(N+1), N,p)
    E= Entropy(f)
    
    plt.plot((1,n),(0,0),"b--",label="y=0")
    P = (np.arange(M+1)*E)
    plt.plot(np.arange(M+1), 100*(Prod-P )/P, "r",label="Error de Pourcentage")
    plt.xlabel("n")
    plt.xscale('log')
    plt.ylabel("Error %")
    plt.title("Tirage Aléatoire")
    plt.legend(loc='best')
    plt.grid("b--", linewidth=0.5)
    plt.show()
    
def Question8(p,e,message=np.array(list([0,1,1,1,0,1,1,0,1,1,0,1]))):
    destin = []
    for i in range(0, len(message)):
        if message[i]==0:
            d= np.random.rand()
            if (d < (1-p-e)): destin+=[0]
            if ((1-p-e)<= d < (1-e)): destin+=['e']
            if (d >= (1-e)): destin+=[1]
        else:
            d= np.random.rand()
            if (d < (1-p-e)): destin+=[1]
            if ((1-p-e)<= d < (1-e)): destin+=['e']
            if (d >= (1-e)): destin+=[0]
    return destin

def CodageBinaire(n=4):
    mots= np.arange(1,np.power(2,n))
    encoda= np.array(np.ones((np.power(2,n),n)))
    for i in range(0, len(mots)):
        a= list(bin(mots[i]))
        a= a[2:]
        s= (n-len(a))
        t = list(np.zeros(s,int))
        encoda[i]=(t+a)
    return encoda

def Inversion(message,p=0.1,e=0,n=4,):   #inversion do canaux de communication
    a= e/(1-p)
    dec= []
    for i in range(0, len(message)):
        p=[]
        for j in range(0,n):
            k=message[i][j]
            if k=='e': 
                d= np.random.rand()
                if (d < 0.5): p+=[0]
                else: p+=[1]
            if k== 1 : 
                d= np.random.rand()
                if (d < a ): p+=[0]
                else: p+=[1]
            if k== 0 :
                d= np.random.rand()
                if (d < a) : p+=[1]
                else: p+=[0]
        dec+=[p]
    return dec

def Encodage(n=4, phrase = np.array(list([1,3,5,6,7,8,5,7,9,0,1,2,8,9,3,4]))):
    dic = CodageBinaire(n)
    codbin= np.array(np.ones((len(phrase), n)))
    for i in range(0, len(phrase)):
        codbin[i]= dic[phrase[i]-1]
    return codbin
        
def PassageCanaux(codbin,p=0.1,e=0):
    A= np.shape(codbin)
    m = []
    for i in range(0,A[0]):
        s = list(codbin[i])
        g=[]
        g=Question8(p,e,s)
        m+= [g]
    return m

    
def SimulationCanaux(M=100000,n=2,p=0.1,e=0):
    P= np.linspace(0,1,M)
    Pe= np.linspace(0,1,M)
    for i in range(0,M):
        k=np.power(2,n)
        #k = np.power(2,n-2)   # Arbitrairement, une phrase de taille 2^(n-1) 
        W = (np.power(2,n)*np.random.rand(k)) #Geration de la message uniformement dans Omega
        W = [ int(w) for w in W]
        X= Encodage(n,W)
        Y = PassageCanaux(X,p,e)
        x= Inversion(Y,p,e,n)
        
        v=list(X)
        v= [ list(f) for f in v]
        a= (x==v)  # verification si les messages sont égalles
        k= precision(x,v) 
        P[i]=k
        Pe[i]=1-a
    N= np.arange(1,M+1)
    Pe = np.cumsum(Pe)/N
    P = np.cumsum(P)/N
        
    plt.plot(N, P, "r",label="Pourcentange de message correct")
    plt.xlabel("M")
    plt.xscale('log')
    plt.ylabel(" %")
    plt.title("Erreur Moyenne de Codage")
    plt.legend(loc='best')
    plt.grid("b--", linewidth=0.5)
    plt.show()
    
    plt.plot(N, Pe, "r",label="Estimation Pe (décodage n'est pas parfait)")
    plt.xlabel("M")
    plt.xscale('log')
    plt.ylabel("Error %")
    plt.title("Estmation Pe")
    plt.legend(loc='best')
    plt.grid("b--", linewidth=0.5)
    plt.show()
    
   
    
def precision(x,X):
    l = len(X)
    s=0
    for i in range(0,l):
        if(X[i]==x[i]): s+=1
    return s/l

    
    
    
    
def Question3(n=1000):
    p=np.array([14./16,1./16/1./16])
    ''' Constrution da probabilité Conjunte: (de la definition)
    P(Y=0)= 14/16   , P(Y=1) = 1/8
    P({a,0})= 14/16 = P(X=a)
    P({b,0})= 0
    P({c,0})= 0
    P({a,1})= 0
    P({b,1})= 1/16
    P({c,1})= 1/16
    '''
    Pcon=np.array([[14./16,0,0],[0,1./16,1./16]])
    PY=np.array([14/16,1/8])
    print("H(X)=",Entropy(p))
    print("H(Y)=",Entropy(PY))
    print("H(X|Y)=",EntropyConditionele(Pcon,PY))
    print("H(X,Y)= ", EntropyConditionele(Pcon,np.ones(len(p))))

def Question3b(n=10000):
    Pcon=np.array([[14./16,0,0],[0,1./16,1./16]])
    PY=np.array([14/16,1/8])
    s=0
    a= np.linspace(0,1,n)
    for i in range(0,n):
        y= simulationY()
        h= variableConditionalle(y,PY,Pcon)
        s+= Entropy(h)
        a[i]=s/i
    plt.plot(range(1,n+1),a, 'r', label="Apromaximations Sucessives")
    C= EntropyConditionele(Pcon,PY)
    plt.plot((1,n),(C,C),"b--",label="H(X|Y)=0.125 Teorique")
    plt.xlabel("n")
    plt.legend(loc='best')
    plt.title("Convergence vers H(X|Y)")
    plt.grid("b--", linewidth=0.5)
    plt.show()
    

def Entropy(p):
    sum=0
    for i in range (0,len(p)):
        if p[i]==0: sum+=0
        else:   sum-=p[i]*np.log(p[i])
    return sum/np.log(2)

def EntropyConditionele(pcon,p):
    A=np.shape(pcon)
    sum=0
    for i in range(0, A[1]):
        for j in range(0,A[0]):
            if (p[j]==0 or pcon[j][i]==0): sum+=0
            else: sum-= pcon[j][i]*np.log(pcon[j][i]/p[j])
    return sum/np.log(2)       
        
def simulationY():
    x=0
    a= np.random.rand()
    if(0 <= a < 14/16): x=0
    if(14/16<= a < 15/16): x=1
    if(15/16<=a): x=2
    if(x==0): return 0
    else: return 1
    
def variableConditionalle(y1,py,pconj):
    A=np.shape(pconj)
    h= np.linspace(0,1,A[1])
    for i in range(0,A[1]):
        s=0
        if y1==1: s=1
        else: s=0
        h[i]= pconj[s][i]/py[s]
    return h
   
def LoiGrandCauchy(n,m):
    for i in range(n):
        x= 2*np.random.rand(m)-1
        x=np.tan(np.pi*x/2)
        S=np.cumsum(x)/np.arange(1,m+1)
        plt.plot(range(1,m+1), S, 'r', label="S_n")
        plt.plot((1,m),(0,0),"b--",label="Esperance")
        plt.xlabel("n")
        plt.legend(loc='best')
        plt.title("LGN")
        plt.show()
        
def Monte_Carlo(n=10000):
    x= np.random.normal(0,1,n);
    y= (np.sin(x)>1/2)
    return np.mean(y)

def Monte_Carlo2(m=1000,n=20):
    lim = 5
    p=0.2
    x= np.random.rand(m,n)
    S= np.cumsum(x, axis=1)
    y= np.sum(S<lim,axis=0,dtype=float)/m
    a= str(min(np.argwhere(y<=p)[0]))
    print(a)
    return a

def comp(n=12,p=0.5):
    a= sps.binom.pmf(np.arange(n+1),n,p)
    s= sps.poisson.pmf(np.arange(n+1),n*p)
    plt.stem(np.arange(n+1),a,"r", label="Binomial" )
    plt.stem(np.arange(n+1),s,"b--", label="Poisson" )
    plt.legend(loc='best')
    plt.show()
    
    
def plottt(func, a=10, labe="label", colo="r"):
    x=np.linspace(-a,a,1000)
    a= func(x)
    plt.plot(x,a, colo,label=labe)
    plt.grid("b--", linewidth=0.5)
    plt.legend(loc='best')
    
def plottt2(func, a=10, labe="label", colo="r"):
    x=np.linspace(0,a,1000)
    a= func(x)
    plt.plot(x,a, colo,label=labe)
    plt.grid("b--", linewidth=0.5)
    plt.legend(loc='best')
    

    