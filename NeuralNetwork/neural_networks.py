#import all the libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

### PART 1: Training of a neural network with a simple architecture
###generates n random training inputs on the line y=3x+2
def simplest_training_data(n):
  w = 3
  b = 2
  x = np.random.uniform(0,1,n)
  y = 3*x+b+0.3*np.random.normal(0,1,n)
  return (x,y)

#this is the training function
#parameters:
# n - size of training data
# k - total number of iterations to use during training.
# eta - is the learning rate
def simplest_training(n, k, eta):
  #get the data
  X, Y = simplest_training_data(n)
  weight = np.random.normal(0,1,1)
  bias = 0

  #iterate over all the inputs k times
  for i in range(0,k):
      delta_w = 0
      delta_b = 0

      #train every input
      for x, y in zip(X, Y):
           a = weight*x  + bias
           delta_w = delta_w + (2 * (a - y) * x)
           delta_b = delta_b + (2 * (a - y))

      #adjust the weight and bias after training
      weight = weight - (eta * delta_w/n)
      bias = bias - (eta * delta_b/n)

  theta = (weight, bias)
  return theta

##testing function
"""
parameters:
theta - represent newtork parameters
x - is the training data"""
def simplest_testing(theta, x):
  x = np.array(x)
  y = (theta[0] * x) + theta[1]
  return y


###########################################
#PART 2
## Part 2: Training a neural network with just an input and output layer architecture with no hidden layers

### Provided function to create training data
"""function returns X and y. This provides two different training sets. When the input, trainset, is 1, the
function produces a simple, linearly separable training set. Half the points are near (0, 0) and half are
near (10, 10). X is a matrix in which each row contains one of these points, so it is n × 2, where n is
the number of points. y is a vector of class labels, which have the value 1 for the points near (0, 0)
and 0 for the points near (1, 1).
When trainset is 2, we generate a different training set that is not linearly separable, but that
corresponds to the Xor problem. Points from class 1 are either near (0, 0) or (10, 10), while points in
class 0 are near either (10, 0) or (0, 10)."""
def single_layer_training_data(trainset):
  n = 10
  if trainset == 1:
    # Linearly separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(n), np.zeros(n)),axis=0)

  elif trainset == 2:
    # Not Linearly Separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2)), np.random.normal((10,0),1,(n,2)), np.random.normal((0,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(2*n), np.zeros(2*n)), axis=0)

  else:
    #print ("function single_layer_training_data undefined for input", trainset
    sys.exit()

  return (X,y)

#training function
"""
parameter:
    k - number of iterations
    eta - learning rate
    trainset - is the training data
returns network parameters"""
def single_layer_training(k, eta, trainset):
  X, Y = single_layer_training_data(trainset)
  n = len(Y)


  #iterate over all the inputs k times
  for i in range(0,k):
      weight = [np.random.normal(0,1,2)]
      bias = 0

      #delta variables represent the total loss of all points.
      delta_w = [0,0]
      delta_b = 0

      #train every input
      for x, y in zip(X, Y):
          z = np.dot(weight,x)
          a = sigmoid(z)
          delta_w = delta_w + ((a - y) * x)
          delta_b = delta_b + (a - y)


      #adjust the weight and bias after training
      weight = weight - (eta/n * delta_w)
      bias = bias - (eta/n * delta_b)

  theta = (weight, bias)

  return theta

"""
paramaters:
    theta - network parameters
    X - test data
return predicted class
"""
def single_layer_testing(theta, X):
  X = np.array(X)
  y = sigmoid(np.dot(X, np.transpose(theta[0])) + theta[1])
  return y

def sigmoid(z):
    return 1.0/(1.0 * np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)(1- sigmoid(z))


#############################################################
#PART 3
#Neural network with a single hidden layer
###function to create training data
"""parameters:
    n - number of random training inputs
    sigma - level of noise added"""
def pca_training_data(n, sigma):
  m = 1
  b = 1
  x1 = np.random.uniform(0,10,n)
  x2 = m*x1+b
  X = [x1,x2]
  X += np.random.normal(0,sigma,(2,n))
  return X


"""
parameter:
    k - number of iterations
    eta - learning rate
    n - number of points in the trainset
    trainset - is the training data
returns network parameters"""
def pca_training(k, eta, n, sigma):
    train_data = np.transpose(pca_training_data(n, sigma))
    weights = np.random.normal(0,1,(2,2))
    bias = np.array([[0],[0,0]])

    for i in range(0,k):
        delta_weights = np.array([[0.0,0.0],[0.0,0.0]])
        delta_bias1 = np.array([0.0])
        delta_bias2 = np.array([0.0 , 0.0])

        #iterate over all inputs
        for x in train_data:
            #get the hidden input
            h = (weights[0,0]*x[0] + weights[0,1]*x[1]) + bias[0]

            #get the outputs
            z = (h * weights[1]) + bias[1]

            #get the delta wieghts and biases for layer 2
            w_delta2 = [(2 * (z[0]-x[0]) * h)[0] , (2 * (z[1]-x[1]) * h)[0]]
            b_delta2 = [2 * (z[0] - x[0]) , 2 * (z[1]-x[0]) ]

            #get the delta wieghts and biases for layer 1
            b_delta1 = ((z[0] - x[0]) * weights[1,0]) + ((z[1] - x[1]) * weights[1,1])
            w_delta1 = [(2*b_delta1* x[0]), (2*b_delta1 * x[1])]

            delta_weights = delta_weights + [w_delta1,w_delta2]
            delta_bias1 = delta_bias1 + b_delta1
            delta_bias2 = delta_bias2 + b_delta2

        delta_bias = np.array( [delta_bias1, delta_bias2] )
        weights = weights - ( delta_weights * (eta/n))
        bias = bias - ( delta_bias * (eta/n))


    return (weights,bias)

def pca_test(theta, X):
    X = np.array(X)
    if (X.shape[1] != 2):
        X = np.transpose(X)

    h = np.dot(X, np.transpose(theta[0][0])) + theta[1][0]
    z = [h * theta[0][1][0] + theta[1][1][0], h * theta[0][1][1] + theta[1][1][1] ]
    return z
