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
    d=d+1
    L = len(hidden_layer_sizes)
    L=L+2
    
    #Initializing the weights for input layer
    weight_layer = np.random.normal(0, 0.1, (d,hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer) #append(0.1*weight_layer)
    
    #Initializing the weights for hidden layers
    for l in range(L-3):
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
            # update the weight and calculate the error
        err[e]=errN/N 
    return err, weights
    
def forwardPropagation(x, weights):
    l=len(weights)+1
    currX = x
    retS=[]
    retX=[]
    retX.append(currX)

    # Forward Propagate for each layer
    for i in range(l-1):
        
        currS= #TODO: Dot product between the layer and the weight matrix
        retS.append(currS)
        currX=currS
        if i != len(weights)-1:
            for j in range(len(currS)):
                currX[j]= # TODO: Apply the activation
            currX= np.hstack((1,currX))
        else:
            currX= #TODO: Apply the output activation
        retX.append(currX)
    return retX,retS

def errorPerSample(X,y_n):
    #TODO: Follow the instruction in the document
    return 

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
    nW=[]
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight=weights[i]
        currG=g[i]
        for j in range(rows):
            for k in range(cols):
                #TODO: Gradient Descent Update
                currWeight[j,k]= # 
        nW.append(currWeight)
    return nW

def activation(s):
    #TODO: Follow the instruction

def derivativeActivation(s):
    #TODO: Follow the instruction

def outputf(s):
    #TODO: Follow the instruction

def derivativeOutput(s):
    #TODO: Follow the instruction

def errorf(x_L,y):
    #TODO: Fill in the return values
    if y==1:
        return 
    else:
        return 

def derivativeError(x_L,y):
    #TODO: Fill in the return values
    if y==1:
        return 
    else:
        return 

def pred(x_n,weights):
    # TODO: prediction using the forwardPropagation function
    retX,retS= # Fill in here
    l=len(retX)

    # Return -1 if probability lesser than 0.5
    # Else return 1
    if retX[l-1]<0.5:
        return 
    else:
        return     
    
def confMatrix(X_train,y_train,w):
    #This is a copy from PA1
    eCount=np.zeros((2,2))
    j=0
    row, col = X_train.shape
    X0 = np.ones((row,1))
    X_train = np.hstack((X0,X_train))
    for j in range(row):
        if (pred(X_train[j],w)==-1 and y_train[j]==-1):
            eCount[0,0]=eCount[0,0]+1
        elif (pred(X_train[j],w)==1 and y_train[j]==-1): 
            eCount[0,1]=eCount[0,1]+1
        elif (pred(X_train[j],w)==1 and y_train[j]==1):
            eCount[1,1]=eCount[1,1]+1
        else:
            eCount[1,0]=eCount[1,0]+1
    return eCount

def plotErr(e,epochs):
    #TODO Plot the function using plt.plot(...,...,linewidth=2.0)
    plt.show() 
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #TODO - Follow the tutorial in class
    return 

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
