import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# X_train is (N, d) where N is number of data points and d is number of features
# y_train is (N,) so it's a 1D numpy array
def fit_LinRegr(X_train, y_train):

    # add dummy column of 1s
    ones_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((ones_column, X_train))

    # compute weight vector, will have shape (11,1)
    w = (np.linalg.pinv(X_train.T @ X_train) @ X_train.T) @ y_train.reshape(-1, 1)
    
    # convert it's shape to (11,)
    w = w.squeeze()

    return w

# X_train y_train same dimensions as in fit_LinRegy
# w has dimensions d+1
def mse(X_train,y_train,w):
    num_data = X_train.shape[0]
   
    # add dummy column of 1s
    ones_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((ones_column, X_train))

    avgError = 0
    for i in range(num_data):
       avgError += (y_train[i] - pred(X_train[i], w)) ** 2
    avgError = avgError / num_data

    return avgError

# X_i shape is (d+1,), w is same
def pred(X_i,w):
   return np.dot(X_i, w)

# here the matrices have d columns, not d+1
def test_SciKit(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_squared_error(predictions, y_test)
    return error

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
