# from math import exp
# import numpy as np
# import matplotlib.pyplot as plt

# def rbf_kernel(x1, x2, variance = 1):
#     return exp(-1 * ((x1-x2) ** 2) / (2*variance))

# def gram_matrix(xs):
#     return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

# xs = np.arange(-1, 1, 0.01)
# mean = [0 for x in xs]
# gram = gram_matrix(xs)

# plt_vals = []
# for i in range(0, 5):
#     ys = np.random.multivariate_normal(mean, gram)
#     plt_vals.extend([xs, ys, "k"])
# plt.plot(*plt_vals)
# plt.show()
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel1(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = 0
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 10         # number of training points.
n = 100         # number of test points.
s = 0    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

K = kernel1(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel1(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel1(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# # draw samples from the prior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
# f_prior = np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(2)
# pl.clf()
# pl.plot(Xtest, f_prior)
# pl.title('Ten samples from the GP prior')
# pl.axis([-5, 5, -3, 3])
# pl.savefig('prior.png', bbox_inches='tight')

# Generate with:
# np.random.normal(size=(n,2))
random_weights = np.array([[-1.09943538, -0.43042637],
       [-0.87377456, -0.89471246],
       [ 1.31662549, -0.05844004],
       [-1.15203785,  0.19147873],
       [ 0.75453576,  0.29473933],
       [-1.17758322, -1.0734176 ],
       [ 1.32139288, -0.52635073],
       [ 1.09859581,  1.06008598],
       [-0.0916233 ,  1.07495867],
       [-0.05122012, -0.32537299],
       [-1.01397733, -0.66642977],
       [ 1.39995821, -1.15611127],
       [ 2.42223523, -0.01831005],
       [ 1.15163288, -0.66988221],
       [-1.54311863, -1.97453546],
       [-0.32853178, -1.57687387],
       [-0.74597992,  0.10464471],
       [-0.96938289, -0.638026  ],
       [-0.45134995,  0.15087415],
       [ 0.26454364,  0.87612791],
       [ 0.60375788, -0.35646722],
       [ 0.32128788, -0.49827165],
       [ 0.27362713, -0.29432819],
       [-0.06448389,  0.25624826],
       [-0.2950606 , -0.19180562],
       [ 0.40818322,  0.57611983],
       [ 1.51206315,  1.91983126],
       [ 1.81266516, -0.04031515],
       [ 0.50863365,  0.53199776],
       [-1.08142783, -2.3091708 ],
       [-0.28005132, -0.98805463],
       [ 0.5467018 , -1.83568952],
       [-1.04806737,  0.34104958],
       [ 0.4565613 , -0.78076757],
       [-0.63492401,  1.60908789],
       [ 0.91794746,  1.3008254 ],
       [ 0.62949172, -0.17838201],
       [ 0.54852361,  1.00804392],
       [ 0.48601501,  1.7734863 ],
       [-2.07129593, -0.29602392],
       [ 0.12206627, -0.05473431],
       [ 0.27400657,  0.39634876],
       [-0.34955375,  0.0384706 ],
       [ 0.50568763,  0.78560502],
       [ 0.86280847, -1.2326723 ],
       [ 1.68448296,  0.66297679],
       [ 0.84872376,  1.22687078],
       [-0.29838175,  0.08594228],
       [ 0.57881882, -0.32095791],
       [ 1.30307069,  1.16919226],
       [ 0.02283966, -1.73290082],
       [ 1.1672812 , -1.10946571],
       [-0.97043771,  2.91518353],
       [-1.08565559, -0.27841872],
       [ 0.07123566, -0.26338109],
       [ 0.0070268 ,  0.43231919],
       [-0.88437392,  0.08119566],
       [-0.38835786, -1.09604007],
       [-0.30205403,  0.54456139],
       [-0.33366651,  0.6478121 ],
       [-0.18487545, -0.11144078],
       [-0.88928697, -0.6631311 ],
       [ 1.02505294,  0.8366363 ],
       [-0.69343475, -0.74093212],
       [ 1.76380968,  3.08398737],
       [-0.04274573,  1.06692284],
       [-1.07382748, -0.20909913],
       [ 1.79707129,  1.32238172],
       [ 0.24372007,  0.3505784 ],
       [-0.56419797, -0.33380045],
       [ 0.12883139, -0.59700755],
       [-1.24247838, -0.45829083],
       [-1.15131295,  0.59439388],
       [-0.07975814, -0.34732859],
       [ 1.30839788, -1.37452232],
       [-0.89490732,  0.97430895],
       [ 0.61539253,  0.57028098],
       [-0.16551751, -1.07195079],
       [ 1.29963083, -0.81629738],
       [-2.31139968,  1.74770916],
       [ 0.58871252,  1.23349476],
       [ 1.22107315, -0.9985554 ],
       [-0.14749804, -0.49900578],
       [-0.20822277,  0.70058435],
       [ 0.28030468, -0.14802876],
       [ 1.54770699,  0.46436449],
       [ 0.84280203, -1.24344193],
       [ 1.03113565, -0.54605822],
       [ 1.48438713, -0.12720555],
       [ 0.20348384, -0.13599503],
       [-0.44477986,  1.37177436],
       [ 0.3912288 , -0.16307241],
       [-1.57756726,  1.55183126],
       [ 0.65442448, -0.65695858],
       [ 0.48630086,  0.94230863],
       [-0.65734645, -1.07202268],
       [ 1.15250744,  0.3731517 ],
       [ 1.19529133, -0.39644431],
       [ 1.83465329,  0.37779279],
       [-1.59290535, -1.328244  ]])
append_random_weights = np.vstack([random_weights,random_weights])

# draw samples from the posterior at our test points.
L_post = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L_post, random_weights)
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Two samples from the GP posterior, uniform X_test')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

# pl.show()

def sort_indices(Xtest,f_post):
    arr1inds = Xtest.argsort(0)
    sorted_arr1 = Xtest[arr1inds[::-1]]
    sorted_arr2 = f_post[arr1inds[::-1]]
    return sorted_arr1.reshape(-1,1), sorted_arr2.reshape(-1,2)

# points we're going to make predictions at.
Xtest2 = np.random.uniform(-5, 5, size=(n,1)).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel1(X, Xtest2))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel1(Xtest2, Xtest2)
# s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
# s = np.sqrt(s2)

# draw samples from the posterior at our other test points.
L_post = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post2 = mu.reshape(-1,1) + np.dot(L_post, random_weights)
pl.figure(4)
pl.clf()
X2, f2 = sort_indices(Xtest2,f_post2)
pl.plot(X2, f2)
pl.title('Two samples from the GP posterior, non-uniform X_test')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')


# combine the plots:
# compute the mean at our test points.
Xtest3 = np.vstack([Xtest,Xtest2])
Lk = np.linalg.solve(L, kernel1(X, Xtest3))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel1(Xtest3, Xtest3)
# s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
# s = np.sqrt(s2)

# draw samples from the posterior at our other test points.
L_post = np.linalg.cholesky(K_ + 1e-6*np.eye(2*n) - np.dot(Lk.T, Lk))
f_post3 = mu.reshape(-1,1) + np.dot(L_post, append_random_weights)

pl.figure(5)
pl.clf()
X3, f3 = sort_indices(Xtest3,f_post3)
pl.plot(X3, f3)
pl.title('combined sample')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()


