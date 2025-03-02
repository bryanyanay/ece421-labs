import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    # Initialize the epoch errors
    err=np.zeros((epochs,1))
    
    # Initialize the architecture
    N, d = X_train.shape
    X0 = np.ones((N,1))
    X_train = np.hstack((X0,X_train))
    d = d + 1
    L = len(hidden_layer_sizes)
    L = L + 2 # L is the total layer = hidden layer + 2
    
    #Initializing the weights for input layer, separately initialize this one
    weight_layer = np.random.normal(0, 0.1, (d,hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer) #append(0.1*weight_layer)
    
    #Initializing the weights for hidden layers
    for l in range(L-3):
        # first hidden[l] + 1 means we account for bias, 
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l]+1,hidden_layer_sizes[l+1])) 
        weights.append(weight_layer) 

    #Initializing the weights for output layers
    weight_layer= np.random.normal(0, 0.1, (hidden_layer_sizes[l+1]+1,1)) 
    weights.append(weight_layer) 
    
    for e in range(epochs):
        choiceArray=np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN=0
        for n in range(N):
            index=choiceArray[n]
            x=np.transpose(X_train[index])
            #TODO: Model Update: Forward Propagation, Backpropagation
            X, S = forwardPropagation(x, weights)
            gradList = backPropagation(X, y_train[index], S, weights)

            # update the weight and calculate the error
            weights = updateWeights(weights ,gradList, alpha)
            errN += errorPerSample(X, y_train[index])
        err[e]=errN/N 
    return err, weights

def activation(s):
    return s if s > 0 else 0

def derivativeActivation(s):
    return 1 if s > 0 else 0

def derivativeError(x_L,y):
    if y == 1:
        return -1 / x_L
    elif y == -1:
        return 1 / (1 - x_L)
    
def derivativeOutput(s):
    es = np.exp(-s)
    ret = es / ((1 + es) ** 2)
    return ret

def outputf(s):
    ret = 1 / (1 + np.exp(-s))
    return ret

def errorf(x_L,y):
    if y == 1:
        return -1 * np.log(x_L)
    elif y == -1:
        return -1 * np.log(1 - x_L)


def forwardPropagation(x, weights):
    l=len(weights)+1 # l now is layer number 
    currX = x
    retS=[]
    retX=[]
    retX.append(currX)

    for i in range(l-1): # we only loop l - 1 layer since the last layer 
        
        currS= np.matmul(np.transpose(weights[i]), currX) # weights[i].shape = (a, b), curX.shape = a, c
        retS.append(currS)
        currX=currS
        if i != len(weights)-1:
            for j in range(len(currS)):
                currX[j] = activation(currX[j])
            currX = np.hstack((1,currX))
        else:
            currX = outputf(currX)
        retX.append(currX)
    return retX,retS


def backPropagation(X,y_n,s,weights): 
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
    delL.insert(0,derivativeError(X[l-1],y_n)*derivativeOutput(s[l-2])) # We use s[l - 2] is bc X[i] is always after activation of previous s[i - 1]
    curr=0
    
    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(len(X)-2, 0, -1): #L-1,...,0
        delNextLayer=delL[curr]
        WeightsNextLayer=weights[i]
        sCur=s[i-1]
        
        #Init this to 0s vector
        delN=np.zeros((len(s[i-1]),1))

        #Now we calculate the gradient backward
        #Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i-1])): #number of nodes in layer i - 1
            for k in range(len(s[i])): #number of nodes in layer i 
              
                delN[j] = delN[j] + delNextLayer[k] * WeightsNextLayer[j + 1][k]
            delN[j] = delN[j] * derivativeActivation(sCur[j])
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
                gL[j,k] = currX[j].item() * currdelL[k].item() # I think we could just do currdelL[k] * currX[j]
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


def errorPerSample(X,y_n):
    return errorf(X[len(X) - 1], y_n)


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
    x = np.arange(1, epochs + 1)
    y = e
    plt.title("Error vs. Epoch") 
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(x,y,linewidth=2.0)
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