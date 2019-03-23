import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

##################################################
#
#                DATA PREP
#
##################################################

##IMPORTANT VARIABLES
learningRate = 10e-7 #learning rate
trainIterations = 50 #100000
trainPrintStep = 10
trainPrint = 0
trainRandomStart = 1
writeStartingGuess = 0
normalMax = 1 ## used in place of maximum to normalize input
def main():
    
    if (len(sys.argv) < 2):
        print("Train or Predict?")
        return
    elif (len(sys.argv) <= 3):
        fname = sys.argv[2] #Predict Test Data file
        if (sys.argv[1].lower() == "t"):
            print("Missing Overfit Test Data File")
            return
        predict(fname)
    elif (len(sys.argv) > 3):
        fname = sys.argv[2] #predict data
        oname = sys.argv[3] #overfit Test data File
        train(fname,oname)
    else:
        tfunc("?",sys.argv[1])
        

def tfunc(fname,tf):
    if fname == "?":
        if (tf.lower() == "t"):
            train("001")
        else:
            predict("002")
    else:
        print (tf.lower())
        if (tf.lower() == "t"):
            train(fname)
        else:
            predict(fname)
    #print("TALLY")
def getTrainData(fname):
    ###########
    #
    # Y = [REM, NOT REM]
    #
    ###########
    DATA = pd.read_csv("input"+fname+".csv",header=None)
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
        Y.append([1,0] if YName[i] == "R" else [0,1])
        
    X.append(np.concatenate((Xprep[lenXprep-2],Xprep[lenXprep-1],zeroList)))
    Y.append([1,0] if YName[lenXprep-1] == "R" else [0,1])
    
    X = np.array(X).astype(float)/1000
    Y = np.array(Y).astype(float)
    return X,Y

def getPredictData(fname):
    ###########
    #
    # Y = [REM, NOT REM]
    #
    ###########
    DATA = pd.read_csv("input"+fname+".csv",header=None)
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
        Y.append([1,0] if YName[i] == "R" else [0,1])
        
    X.append(np.concatenate((Xprep[lenXprep-2],Xprep[lenXprep-1],zeroList)))
    Y.append([1,0] if YName[lenXprep-1] == "R" else [0,1])
    
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
    for row in W2:
        w2f.write(','.join(str(item) for item in row) + "\n")
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

    b2f = pd.read_csv("b2f.csv",header=None)
    b2 = np.array(b2f.as_matrix())
    b2 = b2.T
    b2.shape = (len(b2),)
    
    return W1,b1,W2,b2

def writeTrainedWeights(W1,b1,W2,b2):
    w1f = open("w1t.csv","w")
    for row in W1:
        w1f.write(','.join(str(item) for item in row) + "\n")
    w1f.close()
    
    b1f = open("b1t.csv","w")
    b1f.write(','.join(str(item) for item in b1) + "\n")
    b1f.close()
    
    w2f = open("w2t.csv","w")
    for row in W2:
        w2f.write(','.join(str(item) for item in row) + "\n")
    w2f.close()
    
    b2f = open("b2t.csv","w")
    b2f.write(','.join(str(item) for item in b2) + "\n")
    b2f.close()

def readTrainedWeights():
    w1f = pd.read_csv("w1t.csv",header=None)
    W1 = np.array(w1f.as_matrix())
    
    b1f = pd.read_csv("b1t.csv",header=None)
    b1 = np.array(b1f.as_matrix())
    b1 = b1.T
    b1.shape = (len(b1),)

    w2f = pd.read_csv("w2t.csv",header=None)
    W2 = np.array(w2f.as_matrix())

    b2f = pd.read_csv("b2t.csv",header=None)
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
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y,Z

def classification_rate(Y,P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
            
    return float(n_correct) / n_total

def allStats(Y,P):
    TP = 0 #TRUE POSITIVE (CORRECTLY IDENTIFIED REM)
    FP = 0 #FALSE POSITIVE (INCORRECTLY IDENTIFIED REM)
    TN = 0 #TRUE NEGATIVE (CORRECTLY IDENTIFIED NOT REM)
    FN = 0 #FALSE NEGATIVE (INCORRECTLY IDENTIFIED NOT REM)
    for i in range(len(Y)):
        
        if Y[i] == P[i]:
            if P[i] == 0:
                TP += 1
            else:
                TN += 1
        else:
            if P[i] == 0:
                FP += 1
            else:
                FN += 1
    acc = (TP+TN)/(TP+FP+TN+FN) #accuracy
    sens = TP/(TP+FN) #sensitivity
    spec = TN/(TN+FP) #specificity

    return acc,sens,spec
            
            
    #return float(n_correct) / n_total

def costFunc(Y,P):
    tot = Y*np.log(P)
    return tot.sum()

def derivative_w2(Y,P,Z):
    #N = # of samples
    #K = number of output nodes = 1**
    #M = number of hidden nodes
    N = Y.shape[0]
    M = Z.shape[1]
    K = Y.shape[1]
    dw2 = Z.T.dot(Y-P)
    
    
    #TESTING
    #dw22 = np.zeros((M,K))
    #for n in range(N):
    #    for m in range(M):
    #        for k in range(K):
    #            dw22[m,k] += (Y[n,k]-P[n,k])*Z[n,m]
    #assert(np.abs(dw22-dw2).sum() < 1e-10)
    ##print(dw2[:20])
    ##print(dw22[:20])
    return dw2

def derivative_b2(Y,P):
    return (Y-P).sum(axis=0)
 

def derivative_w1(Y,P,W2,Z,X):
    N = Y.shape[0]
    M = Z.shape[1]
    D = X.shape[1]
    K = Y.shape[1]
    dw1 = X.T.dot((Y-P).dot(W2.T)*Z*(1-Z))
    #dw1n = np.zeros([D,M])
    #for n in range(N):
    #    for k in range(K):
    #        for m in range(M):
    #            for d in range(D):
    #                dw1n[d,m] += (Y[n,k]-P[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
            
    #assert(np.abs(dw1-dw1n).sum() < 1e-10)
    return dw1
    

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
    return ((Y-P).dot(W2.T)*Z*(1-Z)).sum(axis=0)

def train(fname,oname):
    #
    lr = learningRate
    X,Y = getTrainData(fname)
    YT = np.argmax(Y,axis=1)
    OX,OY = getTrainData(oname)
    OYT = np.argmax(OY,axis=1)
    D = len(X[0]) # number of input parameters
    M = D #hidden layer size
    K = int(Y.shape[1]) # of classes (number of output parameters)
    if trainRandomStart == 1:
        W1 = np.random.randn(D,M)
        b1 = np.random.randn(M)
        W2 = np.random.randn(M,K)
        b2 = np.random.randn(K)
    else:
        W1,b1,W2,b2 = readStartGuess()
    
    cost=[]
    YTrate = []
    OYTrate = []
    #probPrev = Y
    #W1Prev = W1
    #b1Prev = b1
    #W2Prev = W2
    #b2Prev = b2
    for iteration in range(trainIterations):
        Prob, Z = forward(X, W1, b1, W2, b2)#Predictions or Probability
        Probo,Zo = forward(OX, W1, b1, W2, b2) #For OverFit Test
        if iteration%trainPrintStep == 0:
            C = costFunc(Y,Prob)
            P = np.argmax(Prob,axis=1)
            Po = np.argmax(Probo,axis=1)
            r = classification_rate(YT,P)
            ro = classification_rate(OYT,Po)
            if r == 1:
            #if np.abs(C) < 10e-1:
                break
            print("\n----------\niteration:",iteration,"\ncost:",C,"\nclassifcation:",r,"\n")
            cost.append(C)
            YTrate.append(r)
            OYTrate.append(ro)
        W2 += lr*derivative_w2(Y,Prob,Z)
        b2 += lr*derivative_b2(Y,Prob)
        W1 += lr*derivative_w1(Y,Prob,W2,Z,X)
        b1 += lr*derivative_b1(Y,Prob,W2,Z)
        #if (np.abs(probPrev - Prob).sum() < 1):
        #    print ("\nNO CHANGE")
        #probPrev = Prob
        #if (np.abs(W1Prev - W1).sum() < 1):
        #    print ("\nW1 NO CHANGE")
        #if (np.abs(W2Prev - W2).sum() < 1):
        #    print ("\nW2 NO CHANGE")
        #if (np.abs(b1Prev - b1).sum() < 1):
        #    print ("b1 NO CHANGE")
        #if (np.abs(b2Prev - b2).sum() < 1):
        #    print ("b2 NO CHANGE")
        #W1Prev = W1
        #b1Prev = b1
        #W2Prev = W2
        #b2Prev = b2
    if writeStartingGuess == 1:
        writeStartGuess(W1,b1,W2,b2,D,M)
    writeTrainedWeights(W1,b1,W2,b2)
    C = costFunc(Y,Prob)
    P = np.argmax(Prob,axis=1)
    Po = np.argmax(Probo,axis=1)
    r = classification_rate(YT,P)
    ro = classification_rate(OYT,Po)
    cost.append(C)
    YTrate.append(r)
    OYTrate.append(ro)
    #print("\n----------\niteration:",iteration,"\ncost:",C,"\nclassifcation:",r,"\n")
    file = open("trainStats.txt", "w")
    file.write("\n----------\niteration:" + str(iteration)+"\ncost:"+str(C)+"\nclassifcation:"+str(r)+"\noverfit classification:"+str(ro)+"\n")
    file.close()
    
    plt.figure()
    plt.plot(cost)
    plt.savefig("cost.png")
    
    plt.figure()
    plt.plot(YTrate)
    plt.savefig("realClassRate.png")

    plt.figure()
    plt.plot(OYTrate)
    plt.savefig("overClassRate.png")

    plt.figure()
    plt.plot(YTrate)
    plt.plot(OYTrate)
    plt.savefig("classRate.png")
    

def predict(fname):
    X,Y = getPredictData(fname)
    YT = np.argmax(Y,axis=1)
    W1,b1,W2,b2 = readTrainedWeights()
    Prob, Z = forward(X, W1, b1, W2, b2)
    P = np.argmax(Prob,axis=1)

    acc,sens,spec=allStats(YT,P)
    print("Accuracy:",acc)
    print("Sensitivity:",sens)
    print("Specificity:",spec)
    print("Classification Rate",classification_rate(YT,P))


if __name__ == "__main__":
    main()
