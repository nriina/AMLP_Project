AMLP Readme

This is a file for implementing the ANN implemented by Peter Gergel and Igor Farkas 

See paper:

https://doi.org/10.1007/978-3-030-01424-7 _ 8


Astrocytes are part of the central nervous system that have been recently been gaining computational ground. They seem to send signals to each other using calcium waves that move in a slower timescale than neural electric communication. This model implements a multi-layer perceptron trained with gradient descent and back propogation. The astrocytic addition adds one astrocyte to each neuron that is activated slowely, over the training set. 

The following code includes a file for running the AMLP, and several datasets that were included in the paper listed above including:
  N-Parity and 2 Spirals
  
This was done as a part of a school project, so the following files are also included: The AMLP implemented on the datasets, a gridsearch to find optimal parameters, and a "batches" file that runs the AMLP numerous times to collect average performance.

Overall it seemed like the Astrocytic implementation improved the performance of the network, especially by giving the network a jolt whenever it seemed to be stuck in a local minimum.

