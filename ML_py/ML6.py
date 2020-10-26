from collections import deque
from collections import namedtuple
import heapq
from math import * 
import numpy as np

#SinglyLinkedListNode= namedtuple("List", "data,value")


class SinglyLinkedListNode:
    def __init__(self,data, value):
        self.data = data
        self.value= value

class segmentTree:
    def __init__(self,arraybase, rang= ""):
        if(rang==""): self.range = range(0, len(arraybase))
        else: self.range= rang
        self.arraybase= arraybase
        print(len(arraybase))
        if(len(arraybase)==1 or len(arraybase)==0):
            self.right= None
            self.left= None
        else:
            try:
                split_index= floor(self.range[int(len(self.range)/2)])
                self.left= segmentTree(arraybase[0:split_index], rang=range(0, split_index))
                self.right= segmentTree(arraybase[(split_index):-1], rang=range(split_index, len(arraybase)))
            except: 
               pass
           
        
        self.value= np.sum(self.arraybase)
    
    def query(self,init, fin):
        pass
    
    def update(self,pos, value):
        pass

    def printt(self):
        #s= ""
        #for i in self.range: s+="\t"
        print("["+ str(self.value)+ "]", end="[")
        if(self.left!=None):self.left.printt()
        print("][", end="")
        if(self.right!=None):self.right.printt()
        print("[", end="")


import os
import random
import re
import sys
import heapq

# Complete the prims function below.
def prims(n, edges, start):
    un = unionfind_structure(n)

    edgess = []
    W= 0
    for a in edges:
        edgess.append((a[2], a[0], a[1]))
    heapq.heapify(edgess)  
    while(True):
        try:
            (w, i,j) = heapq.heappop(edgess)
            if(un.same_set(i,j)): continue
            else:
                W+=w
        except: break
    return W


def prims(n, edges, start):
    un = unionfind_structure(n)
    edgess = []
    W= 0
    for a in edges:
        edgess.append((a[2], a[0], a[1]))
    heapq.heapify(edgess)  
    while(True):
        try:
            (w, i,j) = heapq.heappop(edgess)
        except:
            break
        if(un.same_set(i,j)): continue
        else:
            W+=w
            un.union(i,j)
    return W


class Unionfind:
    def __init__(self,n):
        self.arr= list(range(0,n))
        self.rank= [0]*n

    def union(self,a, b):
        if(self.same_set(self,a,b)==False):
            x= self.find(a)
            y= self.find(b)
            if(self.rank[x] > self.rank[y]): self.arr[y] = x 
            else: 
                self.arr[x] = y
                if(self.rank[x] == self.rank[y]): self.rank[y]+=1
            

    def find(self,a):
        if(i==self.arr[i]):
            return i
        else:
            r = self.find(self.arr[i])
            self.arr[i]=r
            return r
            
    def same_set(self,a,b):
        return self.find(a)==self.find(b)




if __name__ == '__main__':  
  '''
  student = namedtuple("Student", "name, age, description")
  a= student(2,4,5)
  a= student(2,"kkkk",5)
  j= set([2,3,4])
  
  li = [5, 7, 9, 1, 3] 
  # using heapify to convert list into heap 
  heapq.heapify(li) 
  print ("The created heap is : ",end="") 
  heapq.heappush(li,4) 
  print (list(li)) 
  print ("The popped and smallest element is : ",end="") 
  print (heapq.heappop(li))   

  d= SinglyLinkedListNode(3,SinglyLinkedListNode(5,SinglyLinkedListNode(7,SinglyLinkedListNode(4,None))))
  while(d.value!=None):
      print(d.data, end=" -> ")
      d= d.value
  '''
  test = segmentTree(np.array([1,3,5,7, 9, 11]))
  test.printt()

  


#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'countPalindromes' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#


def countPalindromes(s):
    n= len(s) 
    T=[]  
    for i in range(0,n):
        T.append([-1]*n)

    for i in range(0,n-1):
        T[i][i+1]=1

    print(T)
    
    def subs(i,j):
        print(i,j)
        if(T[i][j]==-1):
            d= 0
            if(i>=j): return 0
            if(s[j]==s[i]):
                d= subs(i+1,j)+ 2* subs(i+1,j-1)
            else:  
                d= subs(i+1,j)+ subs(i+1,j-1)
            T[i][j]=d
            return d
        else: return T[i][j]
    
    quant= (subs(0,n-1))
    print(T)
    return quant


#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'countPalindromes' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#


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
                    else: d= 1+ 2* subs(i+1,j-1)
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

