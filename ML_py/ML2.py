import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    F,N = map(int,input().split())
    values = []
    for i in range(0,N):
        g = map(float,input().split())
        values.append(g)
    df= pd.DataFrame(values)
    t = max(0, len(df.columns)-2)
    X= np.array(df.loc[:, 0:t])
    Y=  np.array(df.loc[:, len(df.columns)-1])
    #print(X)
    
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(x_poly,Y)
    
    
    test = input(int())
    values2=[]
    for i in range(0, int(test)):
        g = map(float,input().split())
        values2.append(g)
    df2= pd.DataFrame(values2)
    #print(df2)
    jj= np.array(df2.loc[:, 0:t])
    
    z= model.predict(polynomial_features.fit_transform(jj))
    z= z.reshape(1,-1)[0]
    for a in z:
        print(a)