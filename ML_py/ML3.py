import math
import numpy as np
import matplotlib.pyplot as plt

P = np.array([15,12,8,8,7,7,7,6,5,3]).reshape(-1,1)
H = np.array([10,25,17,11,13,17,20,13,9,15])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(P,H)
print(model.get_params())

plt.scatter(P,H)
plt.plot(P, model.predict(P))
plt.grid()
plt.show()
