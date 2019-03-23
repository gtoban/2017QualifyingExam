import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################################################
#
#                DATA PREP
#
##################################################

##IMPORTANT VARIABLES
learningRate = 10e-5 #learning rate
trainIterations = 100
trainPrintStep = 10
trainRandomStart = 1
writeStartingGuess = 0
normalMax = 1 ## used in place of maximum to normalize input
def getTrainData():
    #
    DATA = pd.read_csv("input001.csv",header=None)
    data = np.array(DATA.as_matrix())
    Xprep = data[:,1:]
    lenXprep = len(Xprep[:,0])
    YName = data[:,0]
    zeroList = np.array([0 for j in range(len(Xprep[0]))])
    #X = np.ones(lenXprep)
    X=[]
    Y = []
    X.append(np.concatenate((zeroList,Xprep[0],Xprep[1])))
    Y.append(1 if YName[0] == "W" else 0)
    for i in range(1,lenXprep-1):        
        X.append(np.concatenate((Xprep[i-1],Xprep[i],Xprep[i+1])))
        Y.append(1 if YName[i] == "W" else 0)
        
    X.append(np.concatenate((Xprep[lenXprep-2],Xprep[lenXprep-1],zeroList)))
    Y.append(1 if YName[lenXprep-1] == "W" else 0)
    
    X = np.array(X).astype(float)/1000
    Y = np.array(Y).astype(float)
    return X,Y

def getTrainDataSoft():
    ###########
    #
    # Y = [REM, NOT REM]
    #
    ###########
    DATA = pd.read_csv("input001.csv",header=None)
    data = np.array(DATA.as_matrix())
    Xprep = data[:,1:]
    lenXprep = len(Xprep[:,0])
    YName = data[:,0]
    zeroList = np.array([0 for j in range(len(Xprep[0]))])
    #X = np.ones(lenXprep)
    X=[]
    Y = []
    X.append(np.concatenate((zeroList,Xprep[0],Xprep[1])))
    Y.append([1,0] if YName[0] == "R" else [0,1])
    for i in range(1,lenXprep-1):        
        X.append(np.concatenate((Xprep[i-1],Xprep[i],Xprep[i+1])))
        Y.append(1 if YName[i] == "R" else 0)
        
    X.append(np.concatenate((Xprep[lenXprep-2],Xprep[lenXprep-1],zeroList)))
    Y.append(1 if YName[lenXprep-1] == "R" else 0)
    
    X = np.array(X).astype(float)/1000
    Y = np.array(Y).astype(float)
    return X,Y

def writeStartGuess(W1,b1,W2,b2,D,M):
    w1f = open("w1f.csv","w")
    for row in W1:
        w1f.write(','.join(str(item) for item in row) + "\n")
    w1f.close()
    
    b1f = open("b1f.csv","w")
    b1f.write(','.join(str(item) for item in b1) + "\n")
    b1f.close()
    
    w2f = open("w2f.csv","w")
    w2f.write(','.join(str(item) for item in W2) + "\n")
    w2f.close()
    
    b2f = open("b2f.csv","w")
    b2f.write(','.join(str(item) for item in b2) + "\n")
    b2f.close()

def readStartGuess():
    w1f = pd.read_csv("w1f.csv",header=None)
    W1 = np.array(w1f.as_matrix())
    
    b1f = pd.read_csv("b1f.csv",header=None)
    b1 = np.array(b1f.as_matrix())
    b1 = b1.T
    b1.shape = (len(b1),)

    w2f = pd.read_csv("w2f.csv",header=None)
    W2 = np.array(w2f.as_matrix())
    W2 = W2.T
    W2.shape = (len(W2),)

    b2f = pd.read_csv("b2f.csv",header=None)
    b2 = np.array(b2f.as_matrix())
    b2 = b2.T
    b2.shape = (len(b2),)
    
    return W1,b1,W2,b2
##################################################
#
#                FUNCTIONS
#
##################################################

def forward(X, W1, b1, W2, b2):
    Z = 1/ (1 + np.exp(-X.dot(W1) - b1))    
    Y = 1/ (1 + np.exp(-Z.dot(W2) - b2))
    return Y,Z

def classification_rate(Y,P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
            
    return float(n_correct) / n_total

def costFunc(Y,P):
    tot = Y*np.log(P)
    return tot.sum()

def derivative_w2(Y,P,Z):
    #N = # of samples
    #K = number of output nodes = 1**
    #M = number of hidden nodes
    N = Y.shape[0]
    M = Z.shape[1]    
    dw2 = np.dot(Y*(1-P),(Z))
    #
    #print(dw2.max())
    
    #TESTING
    #dw22 = np.zeros(M)
    #
    #for n in range(N):
    #    for m in range(M):
    #        dw22[m] += Y[n]*(1-P[n])*Z[n,m]
    #
    #assert(np.abs(dw22-dw2).sum() < 1e-10)
    ##print(dw2[:20])
    ##print(dw22[:20])
    return dw2

def derivative_b2(Y,P):
    return ( Y*(1-P)).sum(axis=0)
 

def derivative_w1(Y,P,W2,Z,X):
    N = Y.shape[0]
    M = Z.shape[1]
    D = X.shape[1]
    #dw1 = np.zeros([D,M])
    dw1n = np.zeros([D,M])
    #for n in range(N):
    #    for m in range(M):
    #        for d in range(D):
    #            dw1[d,m] += Y[n]*(1-P[n])*W2[m]*Z[n,m]*(1-Z[n,m])*X[n,d]
    #assert(np.abs(dw1-dw1n).sum() < 1e-10)
    YP = (Y*(1-P))
    YP.shape = (len(YP),1)
    W2.shape = (len(W2),1)
    temp = (YP.dot(W2.T)*Z*(1-Z))
    return X.T.dot(temp)
    

def derivative_b1(Y,P,W2,Z):
    #N = Y.shape[0]
    #M = Z.shape[1]
    #D = X.shape[1]
    ##db1 = np.zeros([D,M])
    #db1n = np.zeros([D,M])
    
    #for n in range(N):
    #    for m in range(M):
    #        for d in range(D):
    #            db1[d,m] += Y[n]*(1-P[n])*W2[m]*Z[n,m]*(1-Z[n,m])
    YP = (Y*(1-P))
    YP.shape = (len(YP),1)
    W2.shape = (len(W2),1)
    db1n = (YP.dot(W2.T)*Z*(1-Z)).sum(axis=0)
    
    #assert(np.abs(db1-db1n).sum() < 1e-10)
    return db1n

def train():
    #
    lr = learningRate
    X,Y = getTrainData()
    D = len(X[0]) # number of input parameters
    M = D #hidden layer size
    K = 1 # of classes (number of output parameters)
    if trainRandomStart == 1:
        W1 = np.random.randn(D,M)
        b1 = np.random.randn(M)
        W2 = np.random.randn(M)
        b2 = np.random.randn(K)
    else:
        W1,b1,W2,b2 = readStartGuess()
    
    cost=[]
    probPrev = Y
    W1Prev = W1
    b1Prev = b1
    W2Prev = W2
    b2Prev = b2
    for iteration in range(trainIterations):
        Prob, Z = forward(X, W1, b1, W2, b2) #Predictions or Probability
        
        if iteration%trainPrintStep == 0:
            C = costFunc(Y,Prob)
            P = np.round(Prob)
            r = classification_rate(Y,P)
            print("\n----------\niteration:",iteration,"\ncost:",C,"\nclassifcation:",r,"\n")
            cost.append(C)
        W2 += lr*derivative_w2(Y,Prob,Z)
        b2 += lr*derivative_b2(Y,Prob)
        W1 += lr*derivative_w1(Y,Prob,W2,Z,X)
        b1 += lr*derivative_b1(Y,Prob,W2,Z)
        W2.shape = (len(W2),)
        if (np.abs(probPrev - Prob).sum() < 1):
            print ("\nNO CHANGE")
        probPrev = Prob
        if (np.abs(W1Prev - W1).sum() < 1):
            print ("\nW1 NO CHANGE")
        if (np.abs(W2Prev - W2).sum() < 1):
            print ("\nW2 NO CHANGE")
        if (np.abs(b1Prev - b1).sum() < 1):
            print ("b1 NO CHANGE")
        if (np.abs(b2Prev - b2).sum() < 1):
            print ("b2 NO CHANGE")
        W1Prev = W1
        b1Prev = b1
        W2Prev = W2
        b2Prev = b2
    if writeStartingGuess == 1:
        writeStartGuess(W1,b1,W2,b2,D,M)
        
    plt.plot(cost)
    plt.show()

def predict():
    X,Y = getTrainData()
    W1,b1,W2,b2 = getTrainedWeights()
    P_Y_given_X, Z = forward(X, W1, b1, W2, b2) #Predictions or Probability
    
    P = np.round(P_Y_given_X[:,0])
    #assert(len(P) == len(Y))
    #
    print("Classification rate for randomly chosen weights:", classification_rate(Y,P))


train()
