import numpy as np
import csv
import sys
import scipy
from sklearn.preprocessing import normalize
np.set_printoptions(precision=2)

lambdaa = float(sys.argv[1])
sigma2 = float(sys.argv[2])
X = sys.argv[3]
y = sys.argv[4]
testData = sys.argv[5]

X_train = np.loadtxt(X, delimiter=",")
#X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
y_train = np.loadtxt(y, delimiter=",")
X_test = np.loadtxt(testData, delimiter=",")
#X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

d = X_train.shape[1]
identityMatrix = np.eye(d)

w_rr = np.matmul(np.matmul(np.linalg.inv(lambdaa * identityMatrix + np.matmul(X_train.T, X_train)),X_train.T), y_train)

np.savetxt("wRR" +"_"+sys.argv[1] + ".csv", w_rr,delimiter=",")


## Part 2
capSigma = np.linalg.inv(lambdaa * identityMatrix + np.matmul(X_train.T, X_train)/sigma2)
mu = np.matmul(np.matmul(np.linalg.inv(lambdaa * sigma2 * identityMatrix + np.matmul(X_train.T, X_train)), X_train.T),y_train)

# Form predictive distribution p(y0|x0, y, X) for all unmeasured x0 element of D
mu_new = np.matmul(X_test, mu)
sigma2_new = np.zeros((len(X_test),1))
for i in xrange(len(X_test)):
    sigma2_new[i,0] = np.matmul(np.matmul(X_test[i,:], capSigma), X_test[i,:].T)
# find Data point with maximum variance 
index = np.argsort(-1 * sigma2_new,0)[0:10]
indexList = index.flatten().tolist()

with open("active" + "_" + sys.argv[1] + "_" +sys.argv[2] + ".csv" , 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(indexList)

                                
                                
