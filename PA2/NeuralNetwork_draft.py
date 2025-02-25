import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    # Initialize the epoch errors
    err = np.zeros((epochs, 1))
    
    # Initialize parameters
    N, d = X_train.shape            # N = num data points, d = num features in each data point (not including dummy 1)
    X0 = np.ones((N, 1))            # dummy column of 1s
    X_train = np.hstack((X0, X_train))
    d = d + 1                       # now update d to account for the dummy column
    L = len(hidden_layer_sizes) + 2 # number of layers (including input & output)
    
    # Initializing the weights 
    # for weight matrix from layer l going to l+1, row k will be the weights from node k in layer l to all nodes in layer l+1
    weights = [] # list where each element is a weight matrix

    # input layer going to first hidden layer weights
    weight_layer = np.random.normal(0, 0.1, (d, hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights.append(weight_layer) #append(0.1*weight_layer)
    
    # initializing the weights for rest of the hidden layers
    for l in range(L-3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l] + 1, hidden_layer_sizes[l+1])) 
        weights.append(weight_layer) 

    # initializing the weights for output layer
    weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l+1] + 1, 1)) 
    weights.append(weight_layer) 
    
    for e in range(epochs):
        choiceArray = np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN = 0
        for n in range(N):
            index = choiceArray[n]
            x = np.transpose(X_train[index])

            recordX, recordS = forwardPropagation(x, weights)
            gradients = backPropagation(recordX, y_train[index], recordS, weights)
            weights = updateWeights(weights, gradients, alpha)
            errN += errorPerSample(recordX, y_train[index])

        err[e] = errN/N 
    return err, weights
    
def forwardPropagation(x, weights):
    # first element of x is 1, x has d+1 elements
    # weights is a list of length L-1, where each element is a 2D weight matrix
    # L is the number of layers including the initial input layer

    # retS is list of L-1 elements, each element is a vector storing the inputs to the next layer
    # e.g., vector at index 0 stores inputs to layer 1 (first hidden layer)
    # by inputs, we mean the linear combination of previous layer's outputs (and a bias node), before applying activiation function

    # retX is list of L elements, each element is vector storing the outputs of given layer
    # for input layer it's just the input, for others it is after relu is applied
    # each vector includes the dummy 1 (bias node)

    l = len(weights) + 1
    currX = x
    retS = []
    retX = []
    retX.append(currX)

    for i in range(l-1): # for each layer        
        currS = np.dot(currX, weights[i]) # must be in this order
        retS.append(currS)
        currX = currS
        if i != l - 2: # if we're not at the last layer
            for j in range(len(currS)): # for each entry in the layer
                currX[j] = activation(currS[j])
            currX = np.hstack((1,currX))
        else:
            currX = outputf(currS)
            if (currX == 1):
                print("HERE: ", currX, currS, x, retX[-1], weights[-1])
                print("actual s: ", np.dot(retX[-1], weights[-1]))
        retX.append(currX)
    return retX, retS

def errorPerSample(X,y_n):
    # not sure if X[L-1] is a scalar or a np array with 1 element?
    return errorf(X[-1], y_n)

def backPropagation(X,y_n,s,weights):
    # X and s same as retX and retS in forwardPropagation
    # y_n is a single label 

    # the comments from the hint file are wrong. These are the correct indices:
    # X: 0, ..., L-1
    # s: 0, ..., L-2
    # weights: 0, ..., L-2

    l = len(X) # number of layers
    delL = [] # seems to store partial derivatives of loss wrt s (output of neuron before activiation)
    # each entry in delL is a vector, storing the partial derivatives for a given layer

    # this computes the partial derivative vector for the very last layer
    # To be able to complete this function, you need to understand this line below
    # In this line, we are computing the derivative of the Loss function w.r.t the 
    # output layer (without activation). This is dL/dS[l-2]
    # By chain rule, dL/dS[l-2] = dL/dy * dy/dS[l-2] . Now dL/dy is the derivative Error and 
    # dy/dS[l-2]  is the derivative output.

    # print(X[l-1], y_n)
    delL.insert(0, derivativeError(X[l-1], y_n) * derivativeOutput(s[l-2]))
    curr = 0
    
    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(l-2, 0, -1): #L-2,...,1 
        delNextLayer = delL[curr]
        WeightsNextLayer = weights[i]
        sCurrLayer = s[i-1]
        
        # Init this to 0s vector
        delN = np.zeros((len(s[i-1]), 1)) # interesting that this is 2D col vector, not 1D vector?

        #Now we calculate the gradient backward
        #Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i-1])): # number of nodes in layer i - 1
            for k in range(len(s[i])): # number of nodes in layer i
                delN[j] = delN[j] + derivativeActivation(sCurrLayer[j]) * WeightsNextLayer[j][k] * delNextLayer[k]
        
        delL.insert(0, delN)
    
    # now delL has indices: 0, ..., L-2

    # our final return value g will have same dimension as weights

    # We have all the deltas we need. Now, we need to find dL/dW.
    # It's very simple now, dL/dW = dL/dS * dS/dW = dL/dS * X
    g = []
    for i in range(len(delL)): # 0, ..., L-2
        rows, cols = weights[i].shape
        gL = np.zeros((rows, cols))
        currX = X[i] 
        currdelL = delL[i]
        for j in range(rows):
            for k in range(cols):
                gL[j, k] = currX[j].item() * currdelL[k].item()
        g.append(gL)
    return g

def updateWeights(weights, g, alpha):
    nW = []
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight = weights[i]
        currG = g[i]
        for j in range(rows):
            for k in range(cols):
                currWeight[j, k] = currWeight[j, k] - alpha * currG[j, k]  
        nW.append(currWeight)
    return nW

def activation(s):
    if s <= 0: # here we just define derivative to b 0 at 0
        return 0
    else:
        return s

def derivativeActivation(s):
    if s < 0:
        return 0
    else:
        return 1

def outputf(s):
    return 1.0 / (1 + np.exp(-s))

def derivativeOutput(s):
    return np.exp(-s) / ( (1 + np.exp(-2)) ** 2 )

def errorf(x_L,y):
    if y == 1:
        return -1 * np.log(x_L)
    elif y == -1:
        return -1 * np.log(1 - x_L)
    
def derivativeError(x_L,y):
    if y == 1:
        return -1 / x_L
    elif y == -1:
        return 1 / (1 - x_L)

def pred(x_n, weights):
    retX, retS = forwardPropagation(x_n, weights) # x_n includes the dummy 1

    # threshold at 0.5
    if retX[-1] < 0.5:
        return -1
    else:
        return 1
    
def confMatrix(X_train,y_train,w):
    eCount = np.zeros((2,2))
    row, col = X_train.shape

    # add dummy column of 1s
    X0 = np.ones((row,1))
    X_train = np.hstack((X0,X_train))

    for j in range(row):
        if (pred(X_train[j], w) == -1) and (y_train[j] == -1):
            eCount[0,0] += 1
        elif (pred(X_train[j], w) == 1) and (y_train[j] == -1): 
            eCount[0,1] += 1
        elif (pred(X_train[j], w) == 1) and (y_train[j] == 1):
            eCount[1,1] += 1
        else:
            eCount[1,0] += 1

    return eCount

def plotErr(e,epochs):
    plt.plot(range(epochs), e, linewidth=2.0) 
    plt.xlabel("Epoch Index")  
    plt.ylabel("Avg Error")   
    plt.title("Training Error over Epochs") 
    plt.show() 
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    cfMatrix = confusion_matrix(Y_test, Y_pred, labels=[-1, 1])
    return cfMatrix


def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()
