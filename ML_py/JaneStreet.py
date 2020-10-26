import numpy as np
from tabulate import tabulate
import math

n = 28
stats = {}
stats[1] =1
stats[3] =2
stats[5] =3

exclusion_set = set([1,3,5])

def print_function(current_board):
    print(tabulate(current_board))


def next_alike(column, n_line):

    if(column>(n_line-3)):
        return -1
    else:
        if(column==0):
            if(2*(n_line-2) < n):
                return 2*(n_line-2)
            else:
                return -1
        
        if(column==1):
            if((2+2*(n_line-4)) < n):
                return (2+2*(n_line-4))
            else: return(-1)

        if(column!=0 and column!=1):
            offset= (n_line-column-3)
            nu = 2 + 2*offset
            if(nu < n):
                return nu
            else: return -1


def newline(current_board):
    n_line = current_board.shape[0]
    line = np.zeros(n).astype(int)
    #line[1] = current_board[n_line-1, n_line]
    line[0] = current_board[n_line-1, n_line-2]

    for i in range(0, n_line-1):
        res= next_alike(i, n_line)
        if(res!=-1):
            line[res] = current_board[n_line-1, i]


        if(i%4==1):
            base = math.floor(i/4)
            print(base)
            off1 = 2*base
            off2= n_line- base*4-3
            print(n_line)
            print(i)
            print(current_board[n_line-1, i+off1+off2])
            print(n_line-1, i+off1+off2)
            print("kk")
            line[i] = current_board[n_line-1, i+off1+off2]


    available_numbers = sorted(set(range(1, 2*n)) - exclusion_set - set(list(line)))
    
    it=0
    for i in range(0, n):
        if(line[i]==0):
            line[i] = available_numbers[it]
            it+=1    


    exclusion_set.add(line[n_line])

    stats[line[n_line]]= n_line+1

    return line[np.newaxis,:]


if __name__ == "__main__":

    a = np.array(range(1,n+1))
    b = np.array(range(2,n+2))
    c = np.array(range(3,n+3))
    c[0]=2
    a = a[np.newaxis,:]
    b = b[np.newaxis,:]
    c = c[np.newaxis,:]

    current_board = np.concatenate((a,b,c), 0)
    for i in range(1,10):
        current_board = np.concatenate((current_board,newline(current_board)), 0)


    print_function(current_board)
    li = sorted(stats.keys())
    for a in li:
        print(str(a)+ ": "+ str(stats[a]))
    
    
