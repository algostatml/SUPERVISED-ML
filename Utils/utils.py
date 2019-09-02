#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:14:33 2019

@author: kenneth
"""
import numpy as np

class EvalC():
    def __init__(self):
        return
    
    #classification metrics
    '''
                            Actual
                        +ve       -ve
                    ---------------------
                +ve |   TP    |     FP  | +---> Precision
                    --------------------- |
    predicted   -ve |   FN    |   TN    | v
                    --------------------- Recall
    '''
    def TP(self, A, P):
        '''Docstring
        when actual is 1 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 1))
    
    def FP(self, A, P):
        '''Docstring
        when actual is 0 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 1))
    
    def FN(self, A, P):
        '''Docstring
        when actual is 1 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 0))
    
    def TN(self, A, P):
        '''Docstring
        when actual is 0 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 0))
    
    def confusionMatrix(self, A, P):
        '''Docstring
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        TP, FP, FN, TN = (self.TP(A, P),\
                          self.FP(A, P),\
                          self.FN(A, P),\
                          self.TN(A, P))
        return np.array([[TP, FP], [FN, TN]])
    
    def accuracy(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        
        Also: Accuracy np.mean(Y == model.predict(X))
        '''
        return (self.TP(A, P) + self.TN(A, P))/(self.TP(A, P) + self.FP(A, P) +\
                                                self.FN(A, P) + self.TN(A, P))
    
    def precision(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.TP(A, P)/(self.TP(A, P) + self.FP(A, P))
    
    def recall(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.TP(A, P)/(self.TP(A, P) + self.FN(A, P))
    
    def TPR(self, A, P):
        '''Docstring
        True Positive rate:
            True Positive Rate corresponds to the 
            proportion of positive data points that 
            are correctly considered as positive, 
            with respect to all positive data points.
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.recall(A, P)
    
    def FPR(self, A, P):
        '''Docstring
        False Positive rate:
            False Positive Rate corresponds to the 
            proportion of negative data points that 
            are mistakenly considered as positive, 
            with respect to all negative data points.

        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return self.FP(A, P)/(self.FP(A, P) + self.TN(A, P))
    
    def TNR(self, A, P):
        '''Docstring
        True Negative Rate
        '''
        return self.TN(A, P)/(self.TN(A, P) + self.FP(A, P))
       
    def f1(self, A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return (2 * (self.precision(A, P) * self.recall(A, P)))/(self.precision(A, P) + self.recall(A, P))
    
    def summary(self, A, P):
        '''
        :params: A: Actual label
        :params: P: predicted labels
        :return: classification summary
        '''
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('>> Accuracy: %s'%self.accuracy(A, P))
        print('>> Precision: %s'%self.precision(A, P))
        print('>> Recall: %s'%self.recall(A, P))
        print('>> F1-score: %s'%self.f1(A, P))
        print('>> True positive rate: %s'%self.TPR(A, P))
        print('>> False positive rate: %s'%self.FPR(A, P))
        print('*'*40)
        
class EvalR(object):
    def __init__(self):
        return
    
    #-Root Mean Square Error
    def RMSE(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: square root of mean square error
        '''
        return np.sqrt(self.MSE(yh, y))
    #-Mean Square Error
    def MSE(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean square error = average((yh - y)^2)
        '''
        return np.square(yh - y).mean()
    
    #-Mean Absolute Error
    def MAE(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean absolute error = average(|yh - y|)
        '''
        return np.abs(yh - y).mean()
    
    #-Median Absolute Error
    def MDAE(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean absolute error = median(|yh - y|)
        '''
        return np.median(np.abs(yh - y))
    
    #-Mean Squared Log Error
    def MSLE(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean square log error
        '''
        return np.mean(np.square((np.log(y + 1)-np.log(yh + 1))))
    
    #-R-squared Error
    def R_squared(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: R-squared error = 1 - (SS[reg]/SS[total])
        '''
        #-- R_square = 1 - (SS[reg]/SS[total])
        return (1 -(np.sum(np.square(y - yh))/np.sum(np.square(y - y.mean()))))
    
    #--HUber loss
    def Huber(self, yh, y, delta=None):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :param: delta
        :return: Huber loss
        '''
        loss = []
        if not delta:
            delta = 1.0
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        else:
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        return np.array(loss).mean()
    
    #--Explained Variance score
    def explainedVariance(self, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: Explained variance
        '''
        e = y - yh
        return (1 - ((np.sum(np.square(e - np.mean(e))))/(np.sum(np.square(y - y.mean())))))
    
    def summary(self, y, y_hat):
        '''
        :param: y_hat: predicted target
        :param: y: actual target
        :return: Loss summary
        '''
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
        
        
        
