# loading the relevant modules
import numpy as np
import pandas as pd

# importing the data
# Function : Y = 1*(x1)+2*(x2)+3*(x3)+4*(x4)+5*(x5)+6*(x6)+7*(x7)+8*(x8)+9*(x9)+10*(x10) + random noise
data = pd.read_csv("dataset_with_randnoise.csv", header=None)
# target vector
Y = np.array(data.iloc[:, -1:])
# feature matrix
X = np.array(data.iloc[:, :-1])

# assuming no preprocessing was done so the basis function is Identity
# the Weight equation reduces to  W= inv((X.T)(X))(X.T)*Y

# Weight vector calculation as follows

W = np.matmul(np.matmul(np.linalg.inv(np.matmul((X.transpose()),X)),(X.transpose())),Y)
#predicted values
Y_pred = np.matmul(X,W)
#print(Y_pred)

#final weight vector
print("Weight Vector :", W.transpose())
#calculating the mean squared error
print("Mean Squared Error:",np.mean(np.square(Y-Y_pred)))
