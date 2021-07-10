import numpy as np

sizes = [2, 3, 1]

biases = [np.random.randn(y, 1) for y in sizes[1:]]

print(biases)

"""
[array([[-0.4651621 ], [ 0.8158959 ], [ 0.54096477]]), array([[-0.22998989]])]
"""

weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print(weights)

"""
[
    array([[-0.72992396,  0.46994942], [ 1.58703312,  1.04549954], [ 0.14324044, -0.26521768]]), 
    array([[ 0.91158753, -0.47684287, -0.2038082 ]])
]
"""

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))