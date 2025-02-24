import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    #Enter implementation here
    return
    
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
        retX.append(currX)
    return retX, retS

def errorPerSample(X,y_n):
    #Enter implementation here
    return

def backPropagation(X,y_n,s,weights):
    # grad has same dimension as weights
    # X and s same as retX and retS in forwardPropagation
    
    #x:0,1,...,L
    #S:1,...,L
    #weights: 1,...,L
    l=len(X)
    delL=[]

    # To be able to complete this function, you need to understand this line below
    # In this line, we are computing the derivative of the Loss function w.r.t the 
    # output layer (without activation). This is dL/dS[l-2]
    # By chain rule, dL/dS[l-2] = dL/dy * dy/dS[l-2] . Now dL/dy is the derivative Error and 
    # dy/dS[l-2]  is the derivative output.
    delL.insert(0,derivativeError(X[l-1],y_n)*derivativeOutput(s[l-2]))
    curr=0
    
    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(len(X)-2, 0, -1): #L-1,...,0
        delNextLayer=delL[curr]
        WeightsNextLayer=weights[i]
        sCurrLayer=s[i-1]
        
        #Init this to 0s vector
        delN=np.zeros((len(s[i-1]),1))

        #Now we calculate the gradient backward
        #Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i-1])): #number of nodes in layer i - 1
            for k in range(len(s[i])): #number of nodes in layer i
                #TODO: calculate delta at node j
                delN[j]=delN[j]+ # Fill in the rest
        
        delL.insert(0,delN)
    
    # We have all the deltas we need. Now, we need to find dL/dW.
    # It's very simple now, dL/dW = dL/dS * dS/dW = dL/dS * X
    g=[]
    for i in range(len(delL)):
        rows,cols=weights[i].shape
        gL=np.zeros((rows,cols))
        currX=X[i]
        currdelL=delL[i]
        for j in range(rows):
            for k in range(cols):
                #TODO: Calculate the gradient using currX and currdelL
                gL[j,k]= # Fill in here
        g.append(gL)
    return g

def updateWeights(weights,g,alpha):
    #Enter implementation here
    return

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

def pred(x_n,weights):
    #Enter implementation here
    return
    
def confMatrix(X_train,y_train,w):
    #Enter implementation here
    return

def plotErr(e,epochs):
    #Enter implementation here
    return
    
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
        
    # err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    # plotErr(err,100)
    
    # cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    # print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()
