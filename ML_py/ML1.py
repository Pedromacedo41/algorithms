
import math
import os
import random
import re
import sys
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def load_data():
    PATH = r"trainingdata.txt"
    df = pd.read_csv(PATH, header = None)
    #df_h = df.set_index(['charge', 'duration'])
    return df


if __name__ == '__main__':
    F,N = map(int,input().split())
    values = []
    for i in range(0,N):
        g = map(float,input().split())
        values.append(g)
    df= pd.DataFrame(values)
    X= np.array(df[[0,1]])
    Y=  np.array(df[[2]])
    #print(X)
    
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(x_poly,Y)
    
    z= model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(Y,z.reshape(1,-1)[0]))
    r2 = r2_score(Y,z.reshape(1,-1)[0])
    print(rmse)
    print(r2)
    #print(z.reshape(1,-1))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0], X[:,1], Y, c="g")
    ax.plot_trisurf(X[:,0], X[:,1], z.reshape(1,-1)[0])
    plt.show()

    
   
    

    
