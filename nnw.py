#Michael Groff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import mlrose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

def split(df):
    array = df.values
    m,n = array.shape
    ind = np.random.choice(m, int(0.7*m), replace = False)
    all = set(range(0,m))
    left = all - set(ind)
    left = list(left)
    m = len(left)
    indt = np.random.choice(m, int(0.3*m),replace = False)
    left = np.take(left,indt,axis=0)
    test = np.take(array,ind,axis=0)
    trial = np.take(array,left,axis=0)
    return test,trial

def readin():
    #returns two dataframes with data to be used, last row being the result
    df1 = pd.read_csv('winequality-white.csv', sep=";")
    df2 = pd.read_csv('adult.txt', sep=",", header=None)
    df2.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
    stacked = df2[["1","3","5","6","7","8","9","13","14"]].stack()
    df2[["1","3","5","6","7","8","9","13","14"]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return df1,df2

if __name__=="__main__":
    print ("NN Weights")

    df1,df2 = readin()
    X = df1[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']].values
    Y = df1[['quality']].values
    array = df2.values
    X = array[:,:-2]
    Y = array[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    one_hot = OneHotEncoder(categories='auto')

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


    t=time.clock()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [13], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100)
    nn_model1.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model1.predict(X_train_scaled)
    e = time.clock()-t
    print('random_hill_climb')
    print(e)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    print(y_train_accuracy)
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print(y_test_accuracy)
    print

    t=time.clock()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [13], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100)
    nn_model1.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model1.predict(X_train_scaled)
    e = time.clock()-t
    print('simulated_annealing')
    print(e)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    print(y_train_accuracy)
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print(y_test_accuracy)
    print
    t=time.clock()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [13], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100)
    nn_model1.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model1.predict(X_train_scaled)
    e = time.clock()-t
    print('genetic_alg')
    print(e)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    print(y_train_accuracy)
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print(y_test_accuracy)
    print
