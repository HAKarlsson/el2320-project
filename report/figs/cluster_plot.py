import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as mnormal

np.random.seed(0)
A = mnormal([0,0], [[.1,0],[0,.1]], 80)
B = mnormal([2,1], [[.1,0],[0,.1]], 20)

plt.scatter(A[:,0], A[:,1])
plt.scatter(B[:,0], B[:,1])
plt.axis('equal')
plt.axis([-1, 3, -1, 2])
plt.show()
