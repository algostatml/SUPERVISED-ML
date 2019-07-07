#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:40:10 2019

@author: kenneth
"""

import numpy as np
class Regression(object):
    def __init__(self):
        return
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        #--Closed form
        self.beta = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y))
        return self
        
    def predict(self, X):
        Y_hat = X.dot(self.beta)
        return Y_hat
    
    #-Root Mean Square Error
    def RMSE(self, yh, y):
        return np.sqrt(self.MSE(yh, y))
    #-Mean Square Error
    def MSE(self, yh, y):
        return np.square(yh - y).mean()
    #-Mean Absolute Error
    def MAE(self, yh, y):
        return np.abs(yh - y).mean()
    #-Median Absolute Error
    def MDAE(self, yh, y):
        return np.median(np.abs(yh - y))
    #-Mean Squared Log Error
    def MSLE(self, yh, y):
        return np.mean(np.square((np.log(y + 1)-np.log(yh + 1))))
    #-R-squared Error
    def R_squared(self, yh, y):
        #-- R_square = 1 - (SS[reg]/SS[total])
        return (1 -(np.sum(np.square(y - yh))/np.sum(np.square(y - y.mean()))))
    def Huber(self, yh, y, delta=None):
        loss = []
        if not delta:
            delta = 1.0
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        else:
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        return np.array(loss).mean()
    #--Explained Variance score
    def explainedVariance(self, yh, y):
        e = y - yh
        return (1 - ((np.sum(np.square(e - np.mean(e))))/(np.sum(np.square(y - y.mean())))))
    
    def summary(self, y, y_hat):
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('RMSE: %s'%(self.RMSE(y_hat,  y)))
        print('*'*40)
        print('MSE: %s'%(self.MSE(y_hat,  y)))
        print('*'*40)
        print('MAE: %s'%(self.MAE(y_hat,  y)))
        print('*'*40)
        print('MDAE: %s'%(self.MDAE(y_hat,  y)))
        print('*'*40)
        print('R_squared = %s'%(self.R_squared(y_hat,  y)))
        print('*'*40)
        print('Huber = %s'%(self.Huber(y_hat,  y)))
        print('*'*40)
        print('Explained Variance = %s'%(self.explainedVariance(y_hat,  y)))
        print('*'*40)
    
    def plot(self, X, Y, y_hat):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(X.shape[0]), Y)
        plt.plot(np.arange(X.shape[0]), y_hat)
        plt.legend(loc = 2)
        plt.title('True vlaue vs Predicted value')
        plt.xlabel('Data point')
        plt.ylabel('True vlaue vs Predicted value')


class Ridge(Regression):
    def __init__(self, lamda = None):
        super().__init__()
        self.lamda = lamda
        return
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        #--Closed form
        self.beta = np.linalg.solve(self.lamda*np.eye(self.X.shape[1]) + self.X.T.dot(self.X), self.X.T.dot(self.Y))
        return self
        
    def predict(self, X):
        Y_hat = X.dot(self.beta)
        return Y_hat

class GradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def GD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X_train = NxD feature matrix
        param: Y_train = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        self.beta_rec = np.zeros((iterations, X.shape[1]))
        if early_stopping:
            for self.iter in range(iterations):
                #compute gradient
                self.beta = beta - (1/len(Y_train)) *(alpha) * (np.dot(X_train.T, (np.dot(X_train, beta) - Y_train)))
                self.beta_rec[self.iter, :] = self.beta.T
                self.cost_rec[self.iter] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            return self
        else:
            for ii in range(iterations):
                #compute gradient
                self.beta = self.beta - (1/len(Y)) *(alpha) * (np.dot(X.T, (np.dot(X,self.beta) - Y)))
                self.beta_rec[ii, :] = beta.T
                self.cost_rec[ii] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(ii, self.cost_rec[ii]))
            print('*'*40)
            return self
        
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

class RidgeGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self, lamda):
        self.lamda = 0.01
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * (np.sum(np.square(X.dot(beta) - Y)) + (self.lamda*np.sum(np.square(beta))))
    
    def RGD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        self.beta_rec = np.zeros((iterations, X.shape[1]))
        if early_stopping:
            for self.iter in range(iterations):
                #compute gradient
                self.beta = self.beta - (1/len(Y)) *(alpha) * ((np.dot(X.T, (np.dot(X, self.beta) - Y))) + (self.lamda*self.beta))
                self.beta_rec[self.iter, :] = self.beta.T
                self.cost_rec[self.iter] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            return self
        else:
            for self.iter in range(iterations):
                #compute gradient
                self.beta = self.beta - (1/len(Y)) *(alpha) * ((np.dot(X.T, (np.dot(X, self.beta) - Y))) + (self.lamda*self.beta))
                self.beta_rec[self.iter, :] = self.beta.T
                self.cost_rec[self.iter] = self.cost(X, Y, self.beta)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))
            print('*'*40)
            return self
            
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
        
class StochasticGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1(1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def StochGD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        len_y = len(Y)
        if early_stopping:
            for self.iter in range(iterations):
                #compute gradient
                cost_val = []
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    cost_val.append(self.cost(X_samp, Y_samp, self.beta))
                    if cost_val[ij] == cost_val[ij -1]:
                        break
                    else:
                        continue
                self.cost_rec[self.iter] = np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            print('*'*40)
            return self
        else:
            for self.iter in range(iterations):
                #compute gradient
                cost_val = 0.0
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    self.beta = beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, self.beta)
                self.cost_rec[self.iter] = cost_val
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
            print('*'*40)
            return self
        
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
class RidgeStochasticGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self, lamda):
        self.lamda = 0.01
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1(1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
        Return: cost
        '''
        return (1/2*len(Y)) * (np.sum(np.square(X.dot(beta) - Y)) + (self.lamda*np.sum(np.square(beta))))
    
    def RStochGD(self, X, Y, beta, alpha, iterations, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        len_y = len(Y)
        if early_stopping:
            for self.iter in range(iterations):
                #compute gradient
                cost_val = []
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * ((np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp))) + (self.lamda*self.beta))
                    self.cost_val.append(self.cost(X_samp, Y_samp, self.beta))
                    if self.cost_val[ij] == self.cost_val[ij -1]:
                        break
                    else:
                        continue
                self.cost_rec[self.iter] = np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            print('*'*40)
            return self
        else:
            for self.iter in range(iterations):
                #compute gradient
                cost_val = 0.0
                for ij in range(len_y):
                    random_samples = np.random.randint(1, len_y)
                    X_samp = X[:random_samples]
                    Y_samp = Y[:random_samples]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * ((np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp))) + (self.lamda*self.beta))
                    cost_val += self.cost(X_samp, Y_samp, self.beta)
                self.cost_rec[self.iter] = cost_val
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
            print('*'*40)
            return self
        
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
class MinibatchGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self):
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * np.sum(np.square(X.dot(beta) - Y))
    
    def minbatchGD(self, X, Y, beta, alpha, iterations, batch_size = None, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        len_y = len(Y)
        self.number_batches = int(len_y/batch_size)
        if early_stopping:
            for self.iter in range(iterations):
                cost_val = []
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, self.number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    cost_val.append(self.cost(X_samp, Y_samp, self.beta))
                    if cost_val[ij] == cost_val[ij -1]:
                        break
                    else:
                        continue
                self.cost_rec[self.iter] = cost_val #np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            print('*'*40)
            return self
        else:
            for self.iter in range(iterations):
                cost_val = 0
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, self.number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, self.beta)
                self.cost_rec[self.iter] = cost_val 
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
            print('*'*40)
            return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return X.dot(self.beta)
    
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        

class RidgeMinibatchGradientDescent(Regression):
    '''
    Inherits Regression class
    '''
    def __init__(self, lamda):
        self.lamda = 0.01
        super().__init__()
        return
    
    def cost(self, X, Y, beta):
        '''
        param: X = training examples/data. column vector <x1, x2, ...., xn | x E R^D>
        param: Y = target. vector  <y | y E R^DX1>
        param: beta = coefficients, e.g b0, b1
        Return: cost
        '''
        return (1/2*len(Y)) * (np.sum(np.square(X.dot(beta) - Y)) + (self.lamda*np.sum(np.square(beta))))
    
    def ridgeminbatchGD(self, X, Y, beta, alpha, iterations, batch_size = None, early_stopping = None):
        '''
        param: X = NxD feature matrix
        param: Y = Dx1 column vector
        param: beta = Dx1 beta vector coefficients
        param: alpha = learning rate. Default 1e-2
        param: iterations = Number of times to run. Default 1000
        
        Return type; final beta/coefficients, cost and bata iterations
        '''
        self.beta = beta
        self.cost_rec = np.zeros(iterations)
        len_y = len(Y)
        self.number_batches = int(len_y/batch_size)
        if early_stopping:
            for self.iter in range(iterations):
                cost_val = 0
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, self.number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    self.cost_val += self.cost(X_samp, Y_samp, self.beta)
                    if self.cost_val[ij] == self.cost_val[ij -1]:
                        break
                    else:
                        continue
                self.cost_rec[self.iter] = cost_val #np.average(cost_val)
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
                #--compare last and previous value. stop if they are the same
                if not self.cost_rec[self.iter] == self.cost_rec[self.iter -1]:
                    continue
                else:
                    break
            print('*'*40)
            return self
        else:
            for self.iter in range(iterations):
                cost_val = 0
                #randomize dataset using permutation
                random_samples = np.random.permutation(len_y)
                X_random = X[random_samples]
                Y_random = Y[random_samples]
                for ij in range(0, len_y, self.number_batches):
                    #split into batches
                    X_samp = X_random[ij:ij+batch_size]
                    Y_samp = Y_random[ij:ij+batch_size]
                    self.beta = self.beta - (1/len(Y_samp)) *(alpha) * (np.dot(X_samp.T, (np.dot(X_samp, self.beta) - Y_samp)))
                    cost_val += self.cost(X_samp, Y_samp, self.beta)
                self.cost_rec[self.iter] = cost_val 
                print('*'*40)
                print('%s iteratiion, cost = %s'%(self.iter, self.cost_rec[self.iter]))    
            print('*'*40)
            return self
        
    def plot_cost(self, cost, iter_):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iter_), cost)
        plt.title('Cost vs Number of iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
#%%

from collections import Counter
def extractFeatures(model, k_features, fscore = None):
    features = Counter(model.get_booster().get_score())
    features = features.most_common(k_features)
    features = [x[0] for x in features if x[1] >= fscore]
    return features

features = extractFeatures(estimator, 20, 1)
features = ['Area m2',
             'District_id',
             'Street Width',
             'Driver Room',
             'Extra Unit',
             'Apartments',
             'Bed Rooms',
             'WC',
             'With Stairs',
             'Living Rooms',
             'Servant Room']

X = df_standard_no_out[features]
X = np.c_[np.ones((X.shape[0], 1)), X]   
Y = df_standard_no_out[['Price']].values

#--Multivariant regression
lm = Regression().fit(X_train, Y_train)
yhat = lm.predict(X_test)
lm.summary(Y_test, yhat)
gd.plot(X_test[:200], Y_test[:200], yhat[:200])

#--Gradient descent
iterations = 100
gd = GradientDescent().GD(X_train, Y_train, beta = np.zeros(X.shape[1]), alpha = 0.1, iterations = iterations, early_stopping=False)
yhat = gd.predict(X_test)
gd.summary(Y_test, yhat)

#--stochastic gradient descent
stgrad = StochasticGradientDescent().StochGD(X_train, Y_train, beta = np.zeros(X.shape[1]), alpha = 0.8, iterations = iterations, early_stopping=True)
yhat = stgrad.predict(X_test)
stgrad.summary(Y_test, yhat)

#--minibatch gradient descent
minibatch = MinibatchGradientDescent().minbatchGD(X_train, Y_train, beta = np.zeros(X.shape[1]), alpha = 0.01, iterations = iterations, batch_size = 20, early_stopping=True)
yhat = minibatch.predict(X_test)
minibatch.summary(Y, yhat)





















