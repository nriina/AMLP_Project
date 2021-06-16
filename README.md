# AMLP_Project

this is an implementation of a computational astrocytic-neural network under the framework published by Peter Gergel and Igor Farkas (2017)

Paper here:

Gergeľ, P., Farkaš I. (2018). Investigating the Role of Astrocyte Units in a Feedforward
Neural Network. In International Conference on Artificial Neural Networks, pages 73–83.
Springer.


The neural network is a Feed-Forward multi-layer perceptron with adjustable parameters
The astrocytic model adds one astrocyte to each neuron
The model is trained with traditional backpropogation

The tasks included are N-Parity and 2-Spirals
N-parity is a complex logic gate, with a binary input space (N length vector of 1's and 0's). 

The correct output is a 1 (True or Even) if the number of 1's is even, and 0 if the number of inputted 1's is odd.

2-Spirals is a collection of (x,y) points that are pre-labeled as belonging to one spiral or the other. The network attempts to assign the two spirals

