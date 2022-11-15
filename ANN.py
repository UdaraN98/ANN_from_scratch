import numpy as np
import pandas as pd

# creating a loss file to save losses
lossFile = open("SumSquaredLossList.csv", "w")


class ANN(object):
    # initializing nodes  for each layers
    def __init__(self):
        self.inlayers = 11
        self.h1layers = 12
        self.h2layers = 12
        self.outlayers = 1

        # weight matrices for each layers
        # instead of zeros , ones matrices random matrices is preferred
        self.W1 = np.random.randn(self.inlayers, self.h1layers)  # 11*12 matrix
        self.W2 = np.random.randn(self.h1layers, self.h2layers)  # 12*12 matrix
        self.W3 = np.random.randn(self.h2layers, self.outlayers)  # 12* 1 matrix

    # defining feedforward function for forward propergation weights are dot producted then driven through an sigmid

    def feedforward(self, X):
        # input to Hidden layer 1
        self.o1 = np.dot(X, self.W1)
        self.o2 = self.sigmoidact(self.o1)

        # HL1 to HL2
        self.o3 = np.dot(self.o2, self.W2)
        self.o4 = self.sigmoidact(self.o3)

        # HL2 to output
        self.o5 = np.dot(self.o4, self.W3)
        o6 = self.sigmoidact(self.o5)

        return o6

    # defining the sigmoid function

    def sigmoidact(self, x):
        return 1 / (np.exp(-x) + 1)

    # defining the sigmoid functions first derivative for backpropergations

    def sigmoidprime(self, x):
        # returns the first derivative of the sigmoid functions result(x)
        return x * (1 - x)

    # backpropegation method

    def backpropegation(self, X, Y, o6):
        self.error = Y - o6  # calculating error

        # applying the derivative of the Sigmoidactivation to the error

        self.o6delta = self.error * self.sigmoidprime(o6)

        # finding the contribution of the last layer to the error

        self.o4error = self.o6delta.dot(self.W3.T)

        # applying the derivative of the o4 to o4 error

        self.o4delta = self.o4error * self.sigmoidprime(self.o4)

        # finding the contribution of the layer to the error

        self.o2error = self.o4delta.dot(self.W2.T)

        # applying the derivative of the o4 to o4 error

        self.o2delta = self.o2error * self.sigmoidprime(self.o2)

        # adjusting first set (inputLayer --> h1) weights
        self.W1 += X.T.dot(self.o2delta)
        # adjusting second set (h1 --> h2) weights
        self.W2 += self.o2.T.dot(self.o4delta)
        # adjusting third set (h2 --> output) weights
        self.W3 += self.o4.T.dot(self.o6delta)

    # train network method

    def trainNN(self, X, Y):
        # forward feeding
        o6 = self.feedforward(X)
        # back propergation
        self.backpropegation(X, Y, o6)

    def saveSumSquaredLossList(self, i, error):
        lossFile.write(str(i) + "," + str(error.tolist()) + '\n')

    def saveWeights(self):
        # save this in order to reproduce our cool network
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self):
        print("Predicted outputs data based on trained weights: ")

        print("Expected : \n" + str(xPredict))
        print("Output : \n" + str(self.feedforward(xPredict)))


sampleNN = ANN()
trainingEpochs = 100

for i in range(trainingEpochs):  # iterate for 100 times (100 epochs)
    print("Epoch # " + str(i) + "\n")
    print("Network Input : \n" + str(X))
    print("Expected Output : \n" + str(Y))
    print("Actual Output : \n" + str(sampleNN.feedforward(X)))

    # mean sum squared loss
    Loss = np.mean(np.square(Y - sampleNN.feedforward(X)))
    sampleNN.saveSumSquaredLossList(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    sampleNN.trainNN(X, Y)
sampleNN.saveWeights()
#sampleNN.predictOutput()
