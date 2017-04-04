from __future__ import division
import numpy as np
import pandas as pd
import sys

def update_u(u, v, d, n1, userDict):
    for i in range(n1):
        sumu1 = np.matrix(np.zeros((d,d)))
        sumu2 = np.matrix(np.zeros((d,1)))
        for j in userDict[i]:
            sumu1 += np.matmul(np.matrix(v[j]).T, np.matrix(v[j]))
            sumu2 += ratings[i,j] * np.matrix(v[j]).T
        u[i] = np.matmul(np.linalg.inv(l * var * np.matrix(np.eye(d)) + sumu1), sumu2).T
    return u

def update_v(u,v,d,n2, prodDict):
    for j in range(n2):
        sumv1 = np.zeros((d,d))
        sumv2 = np.zeros((d,1))
        for i in prodDict[j]:
            sumv1 += np.matmul(np.matrix(u[i]).T, np.matrix(u[i]))
            sumv2 += np.matrix(ratings[i,j] * u[i]).T
        
        v[j] = np.matmul(np.linalg.inv(l * var * np.matrix(np.eye(d)) + sumv1), sumv2).T
        return v

def calculateMaxLikelihood(data, u, v, nonZeroindexes):
    sumObj1 = 0
    sumObj2 = 0
    sumObj3 = 0
    
    for [i,j] in nonZeroindexes:
        sumObj1 += -1/(2*var) * np.power((ratings[i,j] - np.dot(u[i],v[j])),2)
    
    for i in range(n1):
        sumObj2 +=  -1* l/2 * np.dot(u[i],u[i])
        
    for j in range(n2):
        sumObj3 += -1*l/2 * np.dot(v[j],v[j])
        
    return sumObj1 + sumObj2 + sumObj3
    

if __name__ == '__main__':
    
    d = 5
    var = 1/10
    l = 2
    
    data = pd.read_csv(sys.argv[1],delimiter=',',header=None)
    data.columns = ['user', 'prod', 'rating']
    
    reshapedData = pd.pivot_table(data,index='user',columns='prod',values='rating',fill_value=0.0).reset_index()
    
    # initialize ui for all users
    n1 = data['user'].unique().size
    
    # number of products
    n2 = data['prod'].unique().size
    
    ratings = reshapedData.as_matrix()[:,1:]
    nonZeroindexes = np.transpose(np.nonzero(ratings))
    
    # creating the dictionary for each user and product
    # creating dictionalry for user
    user = {}
    for [i,j] in nonZeroindexes:
        if i in user.keys():
            user[i].append(j)
        else:
            user[i] = [j]
    
    # creating distionary for prod
    prod = {}
    for [i,j] in nonZeroindexes:
        if j in prod.keys():
            prod[j].append(i)
        else:
            prod[j] = [i]
    
    # Initialise u and v
    u = np.zeros((n1,d))
    v = np.random.multivariate_normal(np.array([0 for i in range(d)]), (1/l) * np.eye(d,d), (n2))
    
    objFunc = np.zeros((50,1))
    maxIter = 50
    for counter in range(maxIter):
        # update user location
        u = update_u(u, v, d, n1, user)
        # update object location
        v = update_v(u, v, d, n2, prod)
        
        if counter == 9 or counter == 24 or counter == 49:
            np.savetxt('U-' + str(counter+1) + '.csv', u, delimiter=',')
            np.savetxt('V-' + str(counter+1) + '.csv', v, delimiter=',')
        
        objFunc[counter] = calculateMaxLikelihood(ratings, u, v, nonZeroindexes)
        
        
        
    np.savetxt('objective.csv', objFunc, delimiter=',')   