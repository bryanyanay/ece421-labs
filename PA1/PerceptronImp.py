import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

# xtrain is (n, d), n is total number, d is the dimen of feature, y_train is n x 1 dim vector, having value of +1 and -1
# return w, which represents the coef of the line computed by the pocket algorithm that best separates the two classes, dimension is d + 1
def fit_perceptron(X_train, y_train):
    num_data = y_train.size 

    # add the dummy 1 column
    bias = np.ones((num_data, 1)) 
    X_train = np.concatenate((bias, X_train), axis=1) # concatenate along column, X_cat.shape = (n, d + 1), the first column is the bias
    
    w = np.random.uniform(-0.1, 0.1, size=(X_train.shape[1],)) # w.shape = (d + 1,)

    best_w = w
    best_err = errorPer(X_train, y_train, w)
    for i in range(5000): # we stop if all points correctly classified or if we reach 5000 iterations
        datapoint_idx = 0
        while datapoint_idx < num_data:
            if y_train[datapoint_idx] * np.dot(X_train[datapoint_idx], w) <= 0: # we use <= and not < so that points directly on the decision boundary are considered misclassified
                w = w + y_train[datapoint_idx] * X_train[datapoint_idx]
                break
            datapoint_idx += 1
        
        err = errorPer(X_train, y_train, w)
        if err < best_err:
            best_w = w
        if err == 0: # we correctly classify all points
            return best_w


    return best_w

# x, y train has a feature dim of d + 1
# Output: avgError -- average number of points that are misclassified by the plane defined by w.
def errorPer(X_train, y_train, w):
    n = len(y_train)
    misclass_cnt = 0
    for i in range(n):
        if np.dot(X_train[i], w) * y_train[i] <= 0: # we consider 0 to be misclassified
            misclass_cnt += 1
    return misclass_cnt / n

# xtrain is (n, d), n is total number, d is the dimen of feature, y_train is n x 1 dim vector, having value of +1 and -1
# w, which represents the coef of the line computed by the pocket algorithm that best separates the two classes, dimension is d + 1
# return a matrix with shape = (2, 2)
def confMatrix(X_train, y_train, w):
    n = len(y_train)  
    bias = np.ones((n, 1))
    trainingSample = np.concatenate((bias, X_train), axis=1) # now trainingSample is (n, d + 1)
    cf = np.zeros((2, 2))
    for i, xsample in enumerate(trainingSample):
        if pred(xsample, w) == 1:
            if y_train[i] == 1: # True positive
                cf[1][1] += 1
            else:               # False positive
                cf[0][1] += 1
        else:
            if y_train[i] == 1: # False negative
                cf[1][0] += 1
            else:
                cf[0][0] += 1  # True negative
    return cf

# X_i is a vector of d + 1 dimension, it is a single data point 
# X_i and w are both 1D arrays, they're not row or col vectors (which are 2D arrays)
def pred(X_i, w):
    if np.dot(X_i, w) > 0:
        return 1
    else:
        return -1

# X_train.shape = N x d; X_test.shape = M x D, X is the number of testing samples.
# Y_train.shape = N x 1 (N dimensional vector). Y_test.shape = M x 1
def test_SciKit(X_train, X_test, Y_train, Y_test):    
    # create the Perceptron model with max_iter and tol set to specific values
    model = Perceptron(max_iter=5000, tol=1e-3)

    # Train
    model.fit(X_train, Y_train)
    
    # Predict 
    Y_pred = model.predict(X_test)
    
    # Compute and return the confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
    return cm

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
