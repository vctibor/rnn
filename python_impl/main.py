from network import Network
from numpy import array

network = Network([3, 2, 1])
network.biases = [array([[0.5],[0.4]]), array([[0.9]])]
network.weights = [
    array([[0.2, 0.1, 0.2], [0.1, 0.2, 0.2]]),
    array([[0.5, 0.7]])
]


net_input = array([[0.22], [0.41], [0.3]])

res = network.feedforward(net_input)

b, w = network.backprop(net_input, 1)


print b
print w
