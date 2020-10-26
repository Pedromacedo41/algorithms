import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import scipy.stats as sps
from time import time

def function():
    x=np.linspace(-2,2,1000)
    y= np.sin(x)
    plt.plot(x,y)
    
class nova:
    variable = "string"
    def novo(x): return x
        
    
class Vehicle(object):
    name = ""
    kind = "car"
    color = ""
    value = 100.00
    def description(self):
        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)
        return desc_str
    def __init__(self):
        print("iniciando")
        
def main():
    print("Execução Obrigaatoria")
    car = Vehicle() 
    print(car.description())

    
if __name__== '__main__':
    main()
    
def retorna2matrizes(A):
    return [(A+ np.ones(A.shape)),A]




    
    