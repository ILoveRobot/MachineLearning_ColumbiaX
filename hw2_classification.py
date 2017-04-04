import sys
import numpy as np

if __name__ == '__main__':
    trainData = np.genfromtxt(sys.argv[1], delimiter=',')
    trainLabelTmp = np.genfromtxt(sys.argv[2],delimiter=',')
    
    # lets combine two arrays
    
    
    # number of classes is 6
    k = 10
    nRecord = trainData.shape[0]
    
    # creting data for labeled data
    trainLabel = np.zeros((nRecord,k))
    
    for j,i in enumerate(trainLabelTmp):
        trainLabel[j,i] = 1
        
    
    # calculating class priors
    classPriorMean = np.mean(trainLabel,axis=0)
    print classPriorMean
    
    # creating dictionary for each class
    classDict = {}
    for i in range(k):
        classDict[i] = []
    
    # calculating the mean for each class
    for j,i in enumerate(trainLabelTmp):
        if len(classDict[i]) > 0: 
            classDict[i] = np.vstack((classDict[i],trainData[j]))
        else:
            classDict[i] = trainData[j]
    
    # calculale mean of x for each class 
    featureMean = {}
    for i in range(k):
        featureMean[i] = np.mean(classDict[i],axis=0)
    
    #  now we have feature mean noe calculate variance
    featureStd = {}
    for i in range(k):
        # number of observation for class i
        n = classDict[i].shape[0]
        d = classDict[i].shape[1]
        temp = classDict[i]
        covMat = np.cov(temp.T)
        featureStd[i] = covMat
        
    # print featureStd[3]
    # lets make prediction based on this information
    # get the testData information
    testData = np.genfromtxt(sys.argv[3], delimiter=',')
    y_test = np.zeros((testData.shape[0],k))
    for j in range(testData.shape[0]):
        for i in range(k):
            probSum = 0
            row = np.matrix(testData[j])
            deter = pow(np.linalg.det(np.matrix(featureStd[i])),-0.5)
            prior = classPriorMean[i]
            mean = np.matrix(featureMean[i])
            covMat = np.matrix(featureStd[i])
            diff = row - mean
            inv = np.linalg.inv(covMat)
            prob = prior * deter * np.exp(-1.0/2*np.dot(np.dot(diff,inv),diff.T))
            probSum+=prob
            y_test[j,i] = prob

    y_test = y_test/ y_test.sum(axis=1,keepdims=True)
    
    np.savetxt("probs_test.csv", y_test, delimiter=",")