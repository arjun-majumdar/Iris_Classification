# Iris_Classification
Classification for the classical "Iris" dataset using Neural Networks, more specifically, a Multi-Layer Perceptron (MLP).

The MLP has the following structure-

4 input nodes -> 9 hidden nodes -> 3 output nodes

'Rectified Linear Unit' or 'ReLU' activation function is used for the hidden neurons. While a 'softmax' activation function is used in the output layer. The 'softmax' ensures that the output values are in the range of 0 and 1 and may be used as predicted probabilities.

The Neural Network (Multi-Layer Perceptron) uses the 'Adam' gradient descent version algorithm. The loss/error function is the 'categorical_crossentropy' (which is a logarithmic loss function).

To execute the program, just download:
1.) 'iris.csv' file
2.) 'Iris_Classification.py' file and execute it (after making sure that the import packages are satisfied)
