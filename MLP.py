import numpy as np

# Forward Propogation
def forward_pass(x,w_):
    activations = x
    for i in range(len(w_)):
        activations[-1] = np.append(1,activations[-1])
        z = np.dot(w_[i], activations[-1].T)
        a = 1 / (1 + np.exp(-z))
        activations.append(a)
    return activations[-1]

# Calculate MSE
def mse(predicted,actual):
    delta = predicted - actual
    error = np.mean(delta ** 2)
    return error

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    np.random.seed(seed)
    # array to hold actual outputs for test
    outputs_final = []
    # list to hold MSE errors for each epoch
    errors_final = []

    weights = []
    errors = []

    # Initialize weights for each layer
    # print("layers", layers)
    for i in range(len(layers)):
        np.random.seed(seed)
        if i == 0:
            weights.append(np.random.randn(layers[i], X_train.shape[0] + 1))
        else:
            weights.append(np.random.randn(layers[i], layers[i-1] + 1))
    # print("X_train", X_train.shape)
    # print("Y_train",Y_train.shape)
    # print("X_test", X_test.shape)
    # print("Y_test",Y_test.shape)
    # print("epochs", epochs)
    # print("weights", weights)
    
    for epoch in range(epochs):
        # Iterate over training samples
        for i in range(X_train.shape[1]):            
            w = weights
            # to hold new partial derivatives
            dw = []
            for each in w:
                dw.append(np.zeros_like(each))

            for l in range(len(weights)):
                for j in range(w[l].shape[1]):
                    for k in range(w[l].shape[0]):
                        # calculate f(x+h)
                        w_copy = []
                        for each in w:
                            a = np.copy(each)
                            w_copy.append(a)
                        w_copy[l][k,j] += h
                        y_pred_try = forward_pass([X_train[:, i]],w_copy)
                        error1 = mse(y_pred_try, Y_train[:, i])
                        # calculate f(x-h)
                        w_copy = []
                        for each in w:
                            a = np.copy(each)
                            w_copy.append(a)
                        w_copy[l][k,j] -= h
                        y_pred_try = forward_pass([X_train[:, i]],w_copy)
                        error2 = mse(y_pred_try, Y_train[:, i])
                        # Calculate partial derivative
                        dw[l][k,j] = (error1 - error2) / (2*h)

            for each_layer in range(len(dw)):
                dw[each_layer] = alpha * dw[each_layer]
            weights = [a - b for a, b in zip(weights,dw)]
        
        #Iterate over testing sample
        test_error = []

        for i in range(X_test.shape[1]):
            activations = [X_test[:, i]]
            predicted = forward_pass(activations,weights)
            test_error.append(mse(predicted,Y_test[:, i]))  

        final_error = sum(test_error)/X_test.shape[1]
        # print(" mean error epoch ---->", epoch, final_error)
        errors_final.append(final_error)
        
    #Calculate final predictions on test data
    for i in range(X_test.shape[1]):
        activations = [X_test[:, i]]
        outputs = forward_pass(activations,weights)
        outputs_final.append(outputs)

    outputs_final = np.array(outputs_final).T
    # print("weights final", weights)
    # print("errors_final",errors_final)
    # print("outputs_final",outputs_final,outputs_final.shape)
    # print("Y_test",Y_test,Y_test.shape)
    return weights, errors_final, outputs_final


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
