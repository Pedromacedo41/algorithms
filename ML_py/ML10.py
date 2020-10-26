

def countPalindromes(s):
    n= len(s) 
    T=[]  
    for i in range(0,n):
        T.append([-1]*n)
    print(T)
    
    def subs(i,j):
        if(T[i][j]==-1):
            d= 0
            if(s[j]==s[i]):
                if(i>j): d=0
                else:
                    if(i==j): d=1
                    else: d= 3+ 2* subs(i+1,j-1)
            else:  
                if(i>j): d=0
                else:
                    if(i==j): d=1
                    else: d= 1+subs(i,j-1)
            T[i][j]=d
            return d
        else: return T[i][j]
    
    quant= (subs(0,n-1))
    print(T)
    return (quant)

if __name__ == '__main__':  
    print(countPalindromes("geogbg"))
