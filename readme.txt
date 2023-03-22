# Instruction
# This function creates and trains a multi-layer neural Network
# X_train: Array of input for training [input_dimensions,nof_train_samples]

# Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
# X_test: Array of input for testing [input_dimensions,nof_test_samples]
# Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
# layers: array of integers representing number of nodes in each layer
# alpha: learning rate
# epochs: number of epochs for training.
# h: step size
# seed: random number generator seed for initializing the weights.
# return: This function should return a list containing 3 elements:
# The first element of the return list should be a list of weight matrices.
# Each element of the list corresponds to the weight matrix of the corresponding layer.

# The second element should be a one dimensional array of numbers
# representing the average mse error after each epoch. Each error should
# be calculated by using the X_test array while the network is frozen.
# This means that the weights should not be adjusted while calculating the error.

# The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
# representing the actual output of network when X_test is used as input.

# Notes:
# DO NOT use any other package other than numpy
# Bias should be included in the weight matrix in the first column.
# Assume that the activation functions for all the layers are sigmoid.
# Use MSE to calculate error.
# Use gradient descent for adjusting the weights.
# use centered difference approximation to calculate partial derivatives.
# (f(x + h)-f(x - h))/2*h
# Reseed the random number generator when initializing weights for each layer.
# i.e., Initialize the weights for each layer by:
# np.random.seed(seed)
# np.random.randn()
