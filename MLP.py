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
    for i in range(len(layers)):
        np.random.seed(seed)
        if i == 0:
            weights.append(np.random.randn(layers[i], X_train.shape[0] + 1))
        else:
            weights.append(np.random.randn(layers[i], layers[i-1] + 1))
    
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
    return weights, errors_final, outputs_final

