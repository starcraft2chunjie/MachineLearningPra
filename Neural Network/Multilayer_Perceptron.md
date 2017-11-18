# The practice of Neural Network

## The overall method of Neural network
* define the struct of neural network(the number of input neurons, the number of hidden neurons)
* randomly initialize the model parameter
* step in train loop
  * foreward propagate
  * compute the cost function
  * backpropagate to get the gradient
  * update the parameter

## Some brief introduction to Neural Network
### activation function and it's derivatives
sigmoid function:

g(z) = 1/(1 + e^-z)(d = g(z) * (1 - g(z)))

tanh function:

g(z) = sinhz/coshz = (e^z - e^-z)/(e^z + e^-z)(d = 1 - g(z)^2)

ReLU function:

g(z) = max(0, z)

Leaky ReLU function:

g(z) = max(0.01z, z)




